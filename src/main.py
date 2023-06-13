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
from metrics import roc_auc_score, ndcg_score, mrr_score

import utils
from parameters import parse_args
from preprocess import read_news, get_doc_input
from prepare_data import prepare_training_data, prepare_testing_data, generate_bpemb_embeddings
from dataset import DatasetTrain, DatasetTest, NewsDataset
import discuss_utils


def get_mean(arr):
    return [np.array(i).mean() for i in arr]

def get_sum(arr):
    return [np.array(i).sum() for i in arr]

def print_metrics(rank, cnt, x):
    logging.info("[{}] {} samples: {}".format(rank, cnt, '\t'.join(["{:0.2f}".format(i * 100) for i in x])))


def train(rank, args):
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

    news, news_index, category_dict, authorid_dict, word_dict = read_news(
        os.path.join(args.train_data_dir, 'news.tsv'), args, mode='train')

    news_title, news_category, news_authorid = get_doc_input(
        news, news_index, category_dict, authorid_dict, word_dict, args)
    news_combined = np.concatenate([x for x in [news_category, news_authorid] if x is not None], axis=-1)

    module = importlib.import_module(f'model.{args.model}')
    model = module.Model(args, len(category_dict), len(authorid_dict))

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

    dataset = DatasetTrain(data_file_path, news_index, news_title, news_combined, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    logging.info('Training...')
    for ep in range(args.start_epoch, args.epochs):
        loss = 0.0
        accuary = 0.0
        for cnt, (log_ids, log_mask, input_ids, targets) in enumerate(dataloader):
            if args.enable_gpu:
                log_ids = log_ids.cuda(rank, non_blocking=True)
                log_mask = log_mask.cuda(rank, non_blocking=True)
                input_ids = input_ids.cuda(rank, non_blocking=True)
                targets = targets.cuda(rank, non_blocking=True)

            bz_loss, y_hat = model(log_ids, log_mask, input_ids, targets)
            loss += bz_loss.data.float()
            accuary += utils.acc(targets, y_hat)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            if cnt % args.log_steps == 0:
                logging.info(
                    '[{}][{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                        rank, ep, cnt * args.batch_size, loss.data / cnt, accuary / cnt)
                )

            if rank == 0 and cnt != 0 and cnt % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}-{cnt}.pt')
                torch.save(
                    {
                        'model_state_dict':
                            {'.'.join(k.split('.')[1:]): v for k, v in model.state_dict().items()}
                            if is_distributed else model.state_dict(),
                        'category_dict': category_dict,
                        'word_dict': word_dict,
                        'authorid_dict': authorid_dict
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}.")

        logging.info('Training finish.')

        if rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}.pt')
            torch.save(
                {
                    'model_state_dict':
                        {'.'.join(k.split('.')[1:]): v for k, v in model.state_dict().items()}
                        if is_distributed else model.state_dict(),
                    'category_dict': category_dict,
                    'authorid_dict': authorid_dict,
                    'word_dict': word_dict,
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}.")


def get_test_behavior_path(rank, args):
    data_file_path = os.path.join(args.test_data_dir, f'behaviors_{rank}.tsv.gz')
    if args.test_users == 'seen':
        data_file_path = data_file_path.replace('.tsv', '.seen_users.tsv')
    elif args.test_users == 'unseen':
        data_file_path = data_file_path.replace('.tsv', '.unseen_users.tsv')
        
    return data_file_path

def test(rank, args):
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

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)

    assert ckpt_path is not None, 'No checkpoint found.'
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    authorid_dict = checkpoint['authorid_dict']
    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']

    dummy_embedding_matrix = np.zeros((len(word_dict) + 1, args.word_embedding_dim))
    module = importlib.import_module(f'model.{args.model}')
    model = module.Model(args, dummy_embedding_matrix, len(category_dict), len(authorid_dict))
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_gpu:
        model.cuda(rank)

    model.eval()
    torch.set_grad_enabled(False)

    # news = {}  # Dict: key='news_id, e.g. N1235', value=[ list_of_tokens, cat, authorid ]
    # news_index = {}  # Dict: key=news_id, value=idx
    news, news_index = read_news(os.path.join(args.test_data_dir, 'news.tsv'), args, mode='test')
    
    # news = {}  # Dict: key='news_id, e.g. N1235', value=[ list_of_tokens, cat, authorid ]
    # category_dict = {}  # Dict: key=cat_name, value=idx
    # authorid_dict = {}  # Dict: key=authorid, value=idx
    news_title, news_category, news_authorid = get_doc_input(
        news, news_index, category_dict, authorid_dict, word_dict, args)
    news_combined = np.concatenate([x for x in [news_title, news_category, news_authorid] if x is not None], axis=-1)

    news_dataset = NewsDataset(news_combined)  # news_combined: (num_news, max_num_tokens + 1 + 1) (e.g. )
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=4)

    test_news_vecs = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            if args.enable_gpu:
                input_ids = input_ids.cuda(rank)
            candidate_news_vec = model.news_encoder(input_ids)
            candidate_news_vec = candidate_news_vec.to(torch.device("cpu")).detach().numpy()
            test_news_vecs.extend(candidate_news_vec)  # news_vec: (128, 400)

    test_news_vecs = np.array(test_news_vecs)  # shape:  (num_of_news, 400)
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

    def collate_fn(tuple_list):
        log_vecs = torch.FloatTensor(np.array([x[0] for x in tuple_list]))
        log_mask = torch.FloatTensor(np.array([x[1] for x in tuple_list]))
        news_vecs = [x[2] for x in tuple_list]
        labels = [x[3] for x in tuple_list]
        return (log_vecs, log_mask, news_vecs, labels)

    dataset = DatasetTest(data_file_path, news_index, test_news_vecs, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    local_sample_num = 0
    
    # log_vecs: user feature based on clicked doc history
    # log_mask: log_vecs are padded; log_mask tells which items are padded (0) or not (1)
    # candidate_news_vecs: vectors (from news_encoder) of all candidate news (from that sample row)
    # labels: whether the candidate news is clicked or not
    for cnt, (log_vecs, log_mask, candidate_news_vecs, labels) in enumerate(dataloader):
        local_sample_num += log_vecs.shape[0]

        if args.enable_gpu:
            log_vecs = log_vecs.cuda(rank, non_blocking=True)
            log_mask = log_mask.cuda(rank, non_blocking=True)

        user_vecs = model.user_encoder(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

        for user_vec, candidate_news_vec, label in zip(user_vecs, candidate_news_vecs, labels):
            if label.mean() == 0 or label.mean() == 1:
                continue

            score = np.dot(candidate_news_vec, user_vec)  # candidate_news_vec: (20, 400); user_vec: (400,); score: (20,)
            if args.jitao_score_method:
                score_sorted_idx = np.flip(np.argsort(score))  # Idx of item if we sort it in desc order
                score_by_candidate_news = list(range(candidate_news_vec.shape[0], 0, -1))
                for idx in score_sorted_idx[:args.jitao_topn]:
                    score_by_candidate_news[idx] += args.jitao_boost
                    
                score = score_by_candidate_news

            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        if cnt % args.log_steps == 0:
            print_metrics(rank, local_sample_num, get_mean([AUC, MRR, nDCG5, nDCG10]))

    logging.info('[{}] local_sample_num: {}'.format(rank, local_sample_num))
    if is_distributed:
        local_sample_num = torch.tensor(local_sample_num).cuda(rank)
        dist.reduce(local_sample_num, dst=0, op=dist.ReduceOp.SUM)
        local_metrics_sum = torch.FloatTensor(get_sum([AUC, MRR, nDCG5, nDCG10])).cuda(rank)
        dist.reduce(local_metrics_sum, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            print_metrics('*', local_sample_num, local_metrics_sum / local_sample_num)
    else:
        print('Metrics: AUC, MRR, nDCG5, nDCG10')
        print_metrics('*', local_sample_num, get_mean([AUC, MRR, nDCG5, nDCG10]))


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
    MRR = []
    nDCG5 = []
    nDCG10 = []

    local_sample_num = 0
    
    # log_vecs: user feature based on clicked doc history
    # log_mask: log_vecs are padded; log_mask tells which items are padded (0) or not (1)
    # candidate_news_vecs: vectors (from news_encoder) of all candidate news (from that sample row)
    # labels: whether the candidate news is clicked or not
    for cnt, (labels) in enumerate(dataloader):
        local_sample_num += len(labels)
        for label in labels:
            if label.mean() == 0 or label.mean() == 1:
                continue

            score = list(range(label.shape[0], 0, -1))  # score: (22,)

            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        if cnt % args.log_steps == 0:
            print_metrics(rank, local_sample_num, get_mean([AUC, MRR, nDCG5, nDCG10]))

    logging.info('[{}] local_sample_num: {}'.format(rank, local_sample_num))
    print('Metrics: AUC, MRR, nDCG5, nDCG10')
    print_metrics('*', local_sample_num, get_mean([AUC, MRR, nDCG5, nDCG10]))

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
            test(None, args)
        else:
            torch.multiprocessing.spawn(test, nprocs=args.nGPU, args=(args,))
            
    elif args.mode == 'test_baseline':
        test_baseline(args)
            
    elif args.mode == 'create_embeddings':
        generate_bpemb_embeddings(args.bpemb_embedding_path)
        
    elif args.mode == 'gen_discuss_data':
        discuss_utils.gen_discuss_data(args.data_dir, nrows=args.nrows)
        
    elif args.mode == 'split_data':
        discuss_utils.split_data(args.start_date, args.test_date, args.data_dir, args.train_data_dir, args.test_data_dir, args.frac)
        
    elif args.mode == 'create_bert_embeddings':
        discuss_utils.create_bert_embeddings_file(args.train_data_dir)
        discuss_utils.create_bert_embeddings_file(args.test_data_dir)
        
    elif args.mode == 'ad_hoc':
        #discuss_utils.regen_test_dev_news_tsv(args.data_dir, args.train_data_dir)
        #discuss_utils.regen_test_dev_news_tsv(args.data_dir, args.test_data_dir)
        
        #discuss_utils.plit_dev_behaviors(args.train_data_dir, args.test_data_dir)
        
        discuss_utils.regen_test_dev_news_tsv_for_authorid(args.data_dir, args.train_data_dir)
        discuss_utils.regen_test_dev_news_tsv_for_authorid(args.data_dir, args.test_data_dir)
