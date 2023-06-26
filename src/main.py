from threading import local
import numpy as np
import torch
import logging
from tqdm.auto import tqdm
import torch.optim as optim
import torch.distributed as dist
import os
from pathlib import Path
import random
from torch.utils.data import DataLoader
import importlib
import subprocess
from metrics import roc_auc_score, ndcg_score, hit_score
import pytorch_lightning as pl
import pickle

import utils
from parameters import parse_args
from preprocess import read_news, get_doc_input, get_news_input_matrix
from prepare_data import prepare_training_data, prepare_testing_data, generate_bpemb_embeddings
import discuss_utils
from dataset import DatasetTrain, DatasetTest, NewsDataset


def get_mean(arr):
    return [np.array(i).mean() for i in arr]

def get_sum(arr):
    return [np.array(i).sum() for i in arr]

def print_metrics(rank, cnt, sample_num, x):
    logging.info("[{}][{}] {} samples: {}".format(rank, cnt, sample_num, '\t'.join(["{:0.2f}".format(i * 100) for i in x])))

def save_chkpt(model, ckpt_path, is_distributed, category_dict, authorid_dict, entity_dict, word_dict, skip_news_encoder_embedding_matric=False):
    model_state_dict = {k: v for k, v in model.state_dict().items()}
    if skip_news_encoder_embedding_matric:
        key = 'news_encoder.embedding_matrix.weight'
        if key in model_state_dict:
            model_state_dict.pop(key)

    ckpt_dict = {
            'model_state_dict':
                {'.'.join(k.split('.')[1:]): v for k, v in model_state_dict.items()}
                if is_distributed else model_state_dict,
            'category_dict': category_dict,
            'authorid_dict': authorid_dict,
            'entity_dict': entity_dict,
            'word_dict': word_dict,
    }
    torch.save(ckpt_dict, ckpt_path)


from pytorch_lightning.callbacks import ModelCheckpoint

def train(rank, args):
    if rank is None:
        rank = 0
        
    news, news_index, category_dict, authorid_dict, word_dict, entity_dict = read_news(
    os.path.join(args.train_data_dir, 'news.tsv'), args, mode='train')

    news_combined = get_news_input_matrix(args, news, news_index, category_dict, authorid_dict, word_dict, entity_dict)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename='discuss-{epoch:02d}'
    )

    module = importlib.import_module(f'model.{args.model}')
    data_module = module.TrainDataModule(
        args=args,
        rank=rank,
        news_index=news_index,
        news_combined=news_combined
    )
    
    trainer = pl.Trainer(max_epochs=args.epochs,
                         callbacks=[checkpoint_callback])
    model = module.Model(args=args,
                         num_category=len(category_dict), 
                         num_authorid=len(authorid_dict), 
                         num_entity=len(entity_dict),
                         word_dict=word_dict)
    
    trainer.fit(model, datamodule=data_module)
    
    extra_state = dict(
            category_dict=category_dict,
            authorid_dict=authorid_dict,
            entity_dict=entity_dict,
            word_dict=word_dict
        )
    with open(os.path.join(args.model_dir, 'extra_state.pkl'), 'wb') as f:
        pickle.dump(extra_state, f)
        
    logging.info(f'Model saved to {checkpoint_callback.best_model_path}')


def train_org(rank, args):
    rank, is_distributed = torch_setup(rank, args)

    news, news_index, category_dict, authorid_dict, word_dict, entity_dict = read_news(
        os.path.join(args.train_data_dir, 'news.tsv'), args, mode='train')

    news_combined = get_news_input_matrix(args, news, news_index, category_dict, authorid_dict, word_dict, entity_dict)

    if args.skip_title:
        embedding_matrix = None
    else:
        if rank == 0:
            logging.info('Initializing word embedding matrix...')

        embedding_matrix, have_word = utils.load_matrix(args.bpemb_embedding_path,
                                                        word_dict,
                                                        args.word_embedding_dim)
        if rank == 0:
            logging.info(f'Word dict length: {len(word_dict)}')
            logging.info(f'Have words: {len(have_word)}')
            logging.info(f'Missing rate: {(len(word_dict) - len(have_word)) / len(word_dict)}')

    module = importlib.import_module(f'model.{args.model}')
    model = module.Model(args, embedding_matrix, num_category=len(category_dict), num_authorid=len(authorid_dict), num_entity=len(entity_dict), rank=rank)

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from {ckpt_path}.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.enable_gpu:
        model = model.cuda(rank)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # if rank == 0:
    #     print(model)
    #     for name, param in model.named_parameters():
    #         print(name, param.requires_grad)

    data_file_path = os.path.join(args.train_data_dir, f'behaviors_np{args.npratio}_{rank}.tsv.gz')

    dataset = DatasetTrain(data_file_path, news_index, news_combined, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    logging.info('Training...')
    for ep in range(args.start_epoch, args.epochs):
        loss = 0.0
        accuracy = 0.0
        for cnt, (log_ids, log_mask, input_ids, targets) in enumerate(dataloader):
            if args.enable_gpu:
                log_ids = log_ids.cuda(rank, non_blocking=True)
                log_mask = log_mask.cuda(rank, non_blocking=True)
                input_ids = input_ids.cuda(rank, non_blocking=True)
                targets = targets.cuda(rank, non_blocking=True)

            bz_loss, y_hat = model(log_ids, log_mask, input_ids, targets)
            loss += bz_loss.data.float()
            accuracy += utils.acc(targets, y_hat)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            if cnt % args.log_steps == 0:
                logging.info(
                    '[{}][{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                        rank, ep, cnt * args.batch_size, loss.data / cnt, accuracy / cnt)
                )

            if rank == 0 and cnt != 0 and cnt % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}-{cnt}.pt')
                save_chkpt(model, ckpt_path, is_distributed, category_dict, authorid_dict, entity_dict, word_dict)
                logging.info(f"Model saved to {ckpt_path}.")

        logging.info('Training finish.')

        if rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}.pt')
            save_chkpt(model, ckpt_path, is_distributed, category_dict, authorid_dict, entity_dict, word_dict)
            logging.info(f"Model saved to {ckpt_path}.")


def torch_setup(rank, args):
    if rank is None:
        is_distributed = False
        rank = 0
    else:
        is_distributed = True

    if is_distributed:
        utils.setuplogger()
        dist.init_process_group('nccl', world_size=args.nGPU, init_method='env://', rank=rank)

    if args.enable_gpu:
        torch.cuda.set_device(rank)
        
    return rank, is_distributed


def get_test_behavior_path(rank, args):
    data_file_path = os.path.join(args.test_data_dir, f'behaviors_{rank}.tsv.gz')
    if args.test_users == 'seen':
        data_file_path = data_file_path.replace('.tsv', '.seen_users.tsv')
    elif args.test_users == 'unseen':
        data_file_path = data_file_path.replace('.tsv', '.unseen_users.tsv')
        
    return data_file_path


def test(rank, args):
    if rank is None:
        rank = 0
        
    with open(os.path.join(args.model_dir, 'extra_state.pkl'), 'rb') as f:
        extra_state = pickle.load(f)
        
    category_dict = extra_state['category_dict']
    authorid_dict = extra_state['authorid_dict']
    entity_dict = extra_state['entity_dict']
    word_dict = extra_state['word_dict']

    ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    
    module = importlib.import_module(f'model.{args.model}')
    model = module.Model.load_from_checkpoint(ckpt_path, args=args,
                                              num_category=len(category_dict), 
                                              num_authorid=len(authorid_dict),
                                              num_entity=len(entity_dict), 
                                              word_dict=word_dict)
    
    data_module = module.TestDataModule(
        args=args,
        model=model,
        rank=rank,
        category_dict=category_dict,
        authorid_dict=authorid_dict,
        entity_dict=entity_dict,
        word_dict=word_dict
    )
    
    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.test(model, datamodule=data_module)


def test_org(rank, args):
    rank, is_distributed = torch_setup(rank, args)

    ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    model, authorid_dict, entity_dict, category_dict, word_dict = load_checkpoint_for_inference(rank, args, ckpt_path)

    test_news_vecs, news_index = gen_vecs_from_news_encoder(
        os.path.join(args.test_data_dir, 'news.tsv'),
        model, rank, category_dict, authorid_dict, entity_dict, word_dict, args)
    
    logging.info("news scoring num: {}".format(test_news_vecs.shape[0]))

    if args.show_news_doc_sim and rank == 0:
        doc_sim = 0
        for _ in tqdm(range(1000000)):
            i = random.randrange(1, len(test_news_vecs))
            j = random.randrange(1, len(test_news_vecs))
            if i != j:
                doc_sim += np.dot(test_news_vecs[i], test_news_vecs[j]) / (np.linalg.norm(test_news_vecs[i]) * np.linalg.norm(test_news_vecs[j]))
        logging.info(f'News doc-sim: {doc_sim / 1000000}')

    data_file_path = get_test_behavior_path(rank, args)
    logging.info(f'Behavior file: {data_file_path}')

    def collate_fn(tuple_list):  # len(tuple_list) = batch_size
        log_vecs = torch.FloatTensor(np.array([x[0] for x in tuple_list]))
        log_mask = torch.FloatTensor(np.array([x[1] for x in tuple_list]))
        news_vecs = [x[2] for x in tuple_list]
        labels = [x[3] for x in tuple_list]
        return (log_vecs, log_mask, news_vecs, labels)

    dataset = DatasetTest(data_file_path, news_index, test_news_vecs, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    AUC = []
    HIT5 = []
    HIT10 = []
    nDCG5 = []
    nDCG10 = []

    local_sample_num = 0
    
    # log_vecs: (batch_size, 50, 400), user feature based on clicked doc history
    # log_mask: log_vecs are padded; log_mask tells which items are padded (0) or not (1)
    # candidate_news_vecs: len=batch_size; vectors (from news_encoder) of all candidate news (from that sample row)
    # label_vecs:len=batch_size; each item is a list and contain the label (0/1) of each candidate news
    for cnt, (log_vecs, log_mask, candidate_news_vecs, label_vecs) in enumerate(dataloader):
        local_sample_num += log_vecs.shape[0]

        if args.enable_gpu:
            log_vecs = log_vecs.cuda(rank, non_blocking=True)
            log_mask = log_mask.cuda(rank, non_blocking=True)

        user_vecs = model.user_encoder(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

        for inner_cnt, user_vec, candidate_news_vec, label_vec in zip(range(len(user_vecs)), user_vecs, candidate_news_vecs, label_vecs):
            if label_vec.mean() == 0 or label_vec.mean() == 1:
                continue

            score_vec = infer_score_vec(user_vec, candidate_news_vec, args)
            if cnt == 0 and inner_cnt == 1:
                print(f"score of 2nd row: {score_vec}")

            update_metrics(label_vec, score_vec, AUC, HIT5, HIT10, nDCG5, nDCG10)

        if cnt % args.log_steps == 0:
            print_metrics(rank, cnt, local_sample_num, get_mean([AUC, HIT5, HIT10, nDCG5, nDCG10]))

    logging.info('[{}] local_sample_num: {}'.format(rank, local_sample_num))
    if is_distributed:
        local_sample_num = torch.tensor(local_sample_num).cuda(rank)
        dist.reduce(local_sample_num, dst=0, op=dist.ReduceOp.SUM)
        local_metrics_sum = torch.FloatTensor(get_sum([AUC, HIT5, HIT10, nDCG5, nDCG10])).cuda(rank)
        dist.reduce(local_metrics_sum, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            print_metrics('*', cnt, local_sample_num, local_metrics_sum / local_sample_num)
    else:
        print('Metrics: AUC, HIT5, HIT10, nDCG5, nDCG10')
        print_metrics('*', '*', local_sample_num, get_mean([AUC, HIT5, HIT10, nDCG5, nDCG10]))


def test_one(rank, args):
    rank, _ = torch_setup(rank, args)

    ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    model, authorid_dict, entity_dict, category_dict, word_dict = load_checkpoint_for_inference(rank, args, ckpt_path)

    test_news_vecs, news_index = gen_vecs_from_news_encoder(
        os.path.join(args.test_data_dir, 'news.tsv'),
        model, rank, category_dict, authorid_dict, entity_dict, word_dict, args)
    
    logging.info("news scoring num: {}".format(test_news_vecs.shape[0]))

    dataset = DatasetTest(None, news_index, test_news_vecs, args)

    from datetime import datetime
    line = '598275\t6962008\t2023-05-08 00:00:01\t31096359 31096125 31097543 31097540 31097665 31097417 31097079 31097554 31097055 31097381 31097627 31094645 31078513 31097381 31098942 31098958 31098811 31098748 31098766 31095668 31098280 31099028 31098599 31099159 31099095 31098670 31098033 31098576 31099376 31100193 31098497 31099683 31099262 31100779 31100427 31099711 31102009 31101819 31102598 31102377 31101466 31109635 31109929 31109978 31108756 31111142 31111086 31109556 31111261 31112297 31111526 31108273 31112794 31112968 31111883 31109898 31109157 31113337 31112828 31113654 31114513 31114053 31114251 31113601 31112725 31114746 31115493 31114918 31110366 31112805 31116202 31116653\t31116653-0 31112408-0 31117213-0 31117164-0 31117107-0 31117173-1 31116622-0 31116941-0 31116175-0 31043439-0 31115992-0 31115202-0 31117234-0 31116394-0 31112802-0 31116306-0 31115937-0 31116902-0 31116787-0 31113097-0'

    start = datetime.now()
    times = 1
    for i in range(times):
        log_vecs, log_mask, candidate_news_vecs, _ = dataset.line_mapper(line)
        
        # To similate the type and shape as if it's loaded from DataLoader, which is what's expected by the model.
        log_vecs = torch.from_numpy(log_vecs).unsqueeze(0)
        log_mask = torch.from_numpy(log_mask).unsqueeze(0)
        candidate_news_vecs = torch.from_numpy(candidate_news_vecs).unsqueeze(0)
        
        if args.enable_gpu:
            log_vecs = log_vecs.cuda(rank, non_blocking=True)
            log_mask = log_mask.cuda(rank, non_blocking=True)

        user_vecs = model.user_encoder(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

        user_vec = user_vecs[0]
        candidate_news_vec = candidate_news_vecs[0]

        score_vec = infer_score_vec(user_vec, candidate_news_vec, args)
        
    print(f"score: {score_vec}")
    time_taken = datetime.now() - start
    print(f'Time take for each sample = {time_taken.seconds / times}s')


def update_metrics(label_vec, score_vec, AUC, HIT5, HIT10, nDCG5, nDCG10):
    auc = roc_auc_score(label_vec, score_vec)
    hit5 = hit_score(label_vec, score_vec, k=5)
    hit10 = hit_score(label_vec, score_vec, k=10)
    ndcg5 = ndcg_score(label_vec, score_vec, k=5)
    ndcg10 = ndcg_score(label_vec, score_vec, k=10)

    AUC.append(auc)
    HIT5.append(hit5/100)
    HIT10.append(hit10/100)
    nDCG5.append(ndcg5)
    nDCG10.append(ndcg10)


def infer_score_vec(user_vec, candidate_news_vec, args):
    score_vec = np.dot(candidate_news_vec, user_vec)  # candidate_news_vec: (20, 400); user_vec: (400,); score: (20,)
    if args.jitao_score_method:
        score_sorted_idx = np.flip(np.argsort(score_vec))  # Idx of item if we sort it in desc order
        score_by_candidate_news = list(range(candidate_news_vec.shape[0], 0, -1))
        for idx in score_sorted_idx[:args.jitao_topn]:
            score_by_candidate_news[idx] += args.jitao_boost
                    
        score_vec = score_by_candidate_news
        
    return score_vec


def load_checkpoint_for_inference(rank, args, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    entity_dict = authorid_dict = dict()
    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']

    if args.use_authorid:
        authorid_dict = checkpoint['authorid_dict']
    if args.use_entity:
        entity_dict = checkpoint['entity_dict']

    dummy_embedding_matrix = np.zeros((len(word_dict) + 1, args.word_embedding_dim))
    module = importlib.import_module(f'model.{args.model}')
    model = module.Model(args, dummy_embedding_matrix, num_category=len(category_dict), num_authorid=len(authorid_dict), num_entity=len(entity_dict))
    
    # Note:
    # The embedding matrix is also loaded from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_gpu:
        model.cuda(rank)

    model.eval()
    torch.set_grad_enabled(False)
    return model, authorid_dict, entity_dict, category_dict, word_dict


def gen_vecs_from_news_encoder(news_path, model, rank, category_dict, authorid_dict, entity_dict, word_dict, args):
    # news = {}  # Dict: key='news_id, e.g. N1235', value=[ list_of_tokens, cat, authorid, entity ]
    # news_index = {}  # Dict: key=news_id, value=idx
    news, news_index = read_news(news_path, args, mode='test')
    
    # Note:
    # All the *_dict lookups were read from the saved model checkpoint 
    news_combined = get_news_input_matrix(args, news, news_index, category_dict, authorid_dict, word_dict, entity_dict)
    
    news_dataset = NewsDataset(news_combined)  # news_combined: (num_news, max_num_tokens + 3)
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=4)
    news_vecs = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):  # news_dataloader loads directly from news_combined
            if args.enable_gpu:
                input_ids = input_ids.cuda(rank)
            # input_ids: shape = (128, 22)    
            candidate_news_vec = model.news_encoder(input_ids)
            candidate_news_vec = candidate_news_vec.to(torch.device("cpu")).detach().numpy()
            news_vecs.extend(candidate_news_vec)  # news_vec: (128, 400)

    news_vecs = np.array(news_vecs)  # shape:  (num_of_news, 400)
    return news_vecs, news_index


def test_baseline(args):
    rank = 0

    data_file_path = get_test_behavior_path(rank, args)
    logging.info(f'Behavior file: {data_file_path}')


    def collate_fn(tuple_list):
        labels = [x[3] for x in tuple_list]
        return (labels)

    dataset = DatasetTest(data_file_path, None, None, args, baseline_eval=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    AUC = []
    HIT5 = []
    HIT10 = []
    nDCG5 = []
    nDCG10 = []

    local_sample_num = 0
    
    # log_vecs: user feature based on clicked doc history
    # log_mask: log_vecs are padded; log_mask tells which items are padded (0) or not (1)
    # candidate_news_vecs: vectors (from news_encoder) of all candidate news (from that sample row)
    # labels: whether the candidate news is clicked or not
    for cnt, (labels) in enumerate(dataloader):
        local_sample_num += len(labels)
        for label_vec in labels:
            if label_vec.mean() == 0 or label_vec.mean() == 1:
                continue

            score_vec = list(range(label_vec.shape[0], 0, -1))  # score: (22,)
            update_metrics(label_vec, score_vec, AUC, HIT5, HIT10, nDCG5, nDCG10)

        if cnt % args.log_steps == 0:
            print_metrics(rank, cnt, local_sample_num, get_mean([AUC, HIT5, HIT10, nDCG5, nDCG10]))

    logging.info('[{}] local_sample_num: {}'.format(rank, local_sample_num))
    print('Metrics: AUC, HIT5, HIT10, nDCG5, nDCG10')
    print_metrics('*', '*', local_sample_num, get_mean([AUC, HIT5, HIT10, nDCG5, nDCG10]))

if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    utils.dump_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        if args.prepare:
            logging.info('Preparing training data...')
            total_sample_num = prepare_training_data(args.train_data_dir, args.nGPU, args.npratio, args.seed)
        else:
            if args.skip_count_sample:
                total_sample_num = -1
            else:
                total_sample_num = 0
                for i in range(args.nGPU):
                    data_file_path = os.path.join(args.train_data_dir, f'behaviors_np{args.npratio}_{i}.tsv.gz')
                    if not os.path.exists(data_file_path):
                        logging.error(f'Splited training data {data_file_path} for GPU {i} does not exist. Please set the parameter --prepare as True and rerun the code.')
                        exit()
                    if data_file_path.endswith('.gz'):
                        result = subprocess.getoutput(f'zcat {data_file_path} | wc -l')
                    else:               
                        result = subprocess.getoutput(f'wc -l {data_file_path}')
                    total_sample_num += int(result.split(' ')[0])
            logging.info('Skip training data preparation.')
        logging.info(f'{total_sample_num} training samples, {total_sample_num // args.batch_size // args.nGPU} batches in total.')

        if args.nGPU == 1:
            if args.model == 'NAMLv1':
                train_org(None, args)
            else:
                train(None, args)

        else:
            torch.multiprocessing.spawn(train, nprocs=args.nGPU, args=(args,))

    elif args.mode == 'test':
        if args.prepare:
            logging.info('Preparing testing data...')
            total_sample_num = prepare_testing_data(args.test_data_dir, args.nGPU)
        else:
            if args.skip_count_sample:
                total_sample_num = -1
            else:
                total_sample_num = 0
                for i in range(args.nGPU):
                    data_file_path = os.path.join(args.test_data_dir, f'behaviors_{i}.tsv.gz')
                    if not os.path.exists(data_file_path):
                        logging.error(f'Splited testing data {data_file_path} for GPU {i} does not exist. Please set the parameter --prepare as True and rerun the code.')
                        exit()
                    if data_file_path.endswith('.gz'):
                        result = subprocess.getoutput(f'zcat {data_file_path} | wc -l')
                    else:               
                        result = subprocess.getoutput(f'wc -l {data_file_path}')
                    total_sample_num += int(result.split(' ')[0])
            logging.info('Skip testing data preparation.')
        logging.info(f'{total_sample_num} testing samples in total.')

        if args.nGPU == 1:
            if args.model == 'NAMLv1':
                test_org(None, args)
            else:
                test(None, args)
        else:
            torch.multiprocessing.spawn(test, nprocs=args.nGPU, args=(args,))

    elif args.mode == 'test_one':
        test_one(None, args)
            
    elif args.mode == 'test_baseline':
        test_baseline(args)
            
    elif args.mode == 'create_embeddings':
        generate_bpemb_embeddings(args.bpemb_embedding_path)
        
    elif args.mode == 'gen_discuss_data':
        discuss_utils.gen_discuss_data(args.data_dir, nrows=args.nrows)
        
    elif args.mode == 'split_data':
        discuss_utils.split_data(args.start_date, args.test_date, args.data_dir, args.train_data_dir, args.test_data_dir)
        
    elif args.mode == 'ad_hoc':
        #discuss_utils.regen_test_dev_news_tsv(args.data_dir, args.train_data_dir)
        #discuss_utils.regen_test_dev_news_tsv(args.data_dir, args.test_data_dir)
        
        #discuss_utils.split_dev_behaviors(args.train_data_dir, args.test_data_dir)
        
        discuss_utils.regen_test_dev_news_tsv_for_authorid(args.data_dir, args.train_data_dir)
        discuss_utils.regen_test_dev_news_tsv_for_authorid(args.data_dir, args.test_data_dir)

    elif args.mode == 'gen_entity_lookup':
        discuss_utils.gen_entity_lookup(args.data_dir, update_news_tsv=args.gen_entity_update_news_tsv)
