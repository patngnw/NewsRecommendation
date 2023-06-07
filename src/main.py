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

import utils
from parameters import parse_args
from preprocess import read_news, get_doc_input, create_news_embeddings, read_news_embeddings
from prepare_data import prepare_training_data, prepare_testing_data
from dataset import DatasetTrain, DatasetTest, NewsDataset


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
        
    # news: dict: key=doc_id, value=[cat, subcat]
    # news_index: dict: key=doc_id, value=idx (1-based)
    news, news_index, category_dict, subcategory_dict = read_news(
        os.path.join(args.train_data_dir, 'news.tsv'), args, mode='train')
    
    # news_title: shape=(num_news+1, 1), value=idx of doc_id
    # news_category: shape=(num_news+1, 1), value=idx of that news item's category
    # news_subcategory: shape=(num_news+1, 1), value=idx of that news item's subcategory
    news_title, news_category, news_subcategory = get_doc_input(
        news, news_index, category_dict, subcategory_dict, args)
    
    # news_combined: shape=(num_news+1, 3)
    news_combined = np.concatenate([x for x in [news_title, news_category, news_subcategory] if x is not None], axis=-1)  # news_combined.shape: (51283, 22)

    if rank == 0:
        logging.info('Initializing word embedding matrix...')
        
    # embedding_matrix: np.array, shape=(12507, 300); 12507 refers to the number of words found in news, and is equal to len(word_dict)
    #embedding_matrix, have_word = utils.load_matrix(args.glove_embedding_path,
    #                                                word_dict,  # dict: key=word (read from news), value=index
    #                                                args.word_embedding_dim)
    #if rank == 0:
    #    logging.info(f'Word dict length: {len(word_dict)}')
    #    logging.info(f'Have words: {len(have_word)}')
    #    logging.info(f'Missing rate: {(len(word_dict) - len(have_word)) / len(word_dict)}')
    
    embedding_matrix = read_news_embeddings(args.train_data_dir)
    module = importlib.import_module(f'model.{args.model}')
    model = module.Model(args, embedding_matrix, len(category_dict), len(subcategory_dict))
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

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

    data_file_path = os.path.join(args.train_data_dir, f'behaviors_np{args.npratio}_{rank}.tsv')

    dataset = DatasetTrain(data_file_path, news_index, news_combined, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    logging.info('Training...')
    for ep in range(args.start_epoch, args.epochs):
        loss = 0.0
        accuary = 0.0
        for cnt, (history, history_mask, candidate, label) in enumerate(dataloader):
            if args.enable_gpu:
                history = history.cuda(rank, non_blocking=True)
                history_mask = history_mask.cuda(rank, non_blocking=True)
                candidate = candidate.cuda(rank, non_blocking=True)
                label = label.cuda(rank, non_blocking=True)

            bz_loss, y_hat = model(history, history_mask, candidate, label)
            loss += bz_loss.data.float()
            accuary += utils.acc(label, y_hat)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            if cnt % args.log_steps == 0:
                logging.info(
                    '[{}][{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                        ep, rank, cnt * args.batch_size, loss.data / cnt, accuary / cnt)
                )

            if rank == 0 and cnt != 0 and cnt % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}-{cnt}.pt')
                torch.save(
                    {
                        'model_state_dict':
                            {'.'.join(k.split('.')[1:]): v for k, v in model.state_dict().items() if k != 'news_encoder.title_embeddings.weight'}
                            if is_distributed else model.state_dict(),
                        'category_dict': category_dict,
                        'subcategory_dict': subcategory_dict
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}.")

        logging.info('Training finish.')

        if rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}.pt')
            torch.save(
                {
                    'model_state_dict':
                        {'.'.join(k.split('.')[1:]): v for k, v in model.state_dict().items() if k != 'news_encoder.title_embeddings.weight'}
                        if is_distributed else model.state_dict(),
                    'category_dict': category_dict,
                    'subcategory_dict': subcategory_dict,
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}.")


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

    subcategory_dict = checkpoint['subcategory_dict']
    category_dict = checkpoint['category_dict']

    embedding_matrix = read_news_embeddings(args.test_data_dir, args.news_dim)
    module = importlib.import_module(f'model.{args.model}')
    model = module.Model(args, embedding_matrix, len(category_dict), len(subcategory_dict))
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_gpu:
        model.cuda(rank)

    model.eval()
    torch.set_grad_enabled(False)

    news, news_index = read_news(os.path.join(args.test_data_dir, 'news.tsv'), args, mode='test')
    news_title, news_category, news_subcategory = get_doc_input(
        news, news_index, category_dict, subcategory_dict, args)
    news_combined = np.concatenate([x for x in [news_title, news_category, news_subcategory] if x is not None], axis=-1)

    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=4)

    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            input_ids = input_ids.cuda(rank)
            news_vec = model.news_encoder(input_ids)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)

    news_scoring = np.array(news_scoring)
    logging.info("news scoring num: {}".format(news_scoring.shape[0]))

    if rank == 0:
        doc_sim = 0
        for _ in tqdm(range(1000000)):
            i = random.randrange(1, len(news_scoring))
            j = random.randrange(1, len(news_scoring))
            if i != j:
                doc_sim += np.dot(news_scoring[i], news_scoring[j]) / (np.linalg.norm(news_scoring[i]) * np.linalg.norm(news_scoring[j]))
        logging.info(f'News doc-sim: {doc_sim / 1000000}')

    data_file_path = os.path.join(args.test_data_dir, f'behaviors_{rank}.tsv')

    def collate_fn(tuple_list):
        log_vecs = torch.FloatTensor([x[0] for x in tuple_list])
        log_mask = torch.FloatTensor([x[1] for x in tuple_list])
        news_vecs = [x[2] for x in tuple_list]
        labels = [x[3] for x in tuple_list]
        return (log_vecs, log_mask, news_vecs, labels)

    dataset = DatasetTest(data_file_path, news_index, news_scoring, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    from metrics import roc_auc_score, ndcg_score, mrr_score

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    def print_metrics(rank, cnt, x):
        logging.info("[{}] {} samples: {}".format(rank, cnt, '\t'.join(["{:0.2f}".format(i * 100) for i in x])))

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    def get_sum(arr):
        return [np.array(i).sum() for i in arr]

    local_sample_num = 0

    for cnt, (log_vecs, log_mask, news_vecs, labels) in enumerate(dataloader):
        local_sample_num += log_vecs.shape[0]

        if args.enable_gpu:
            log_vecs = log_vecs.cuda(rank, non_blocking=True)
            log_mask = log_mask.cuda(rank, non_blocking=True)

        user_vecs = model.user_encoder(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

        for user_vec, news_vec, label in zip(user_vecs, news_vecs, labels):
            if label.mean() == 0 or label.mean() == 1:
                continue

            score = np.dot(news_vec, user_vec)

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

    if 'train' in args.mode:
        if args.prepare:
            logging.info('Preparing training data...')
            total_sample_num = prepare_training_data(args.train_data_dir, args.nGPU, args.npratio, args.seed)
        else:
            total_sample_num = 0
            for i in range(args.nGPU):
                data_file_path = os.path.join(args.train_data_dir, f'behaviors_np{args.npratio}_{i}.tsv')
                if not os.path.exists(data_file_path):
                    logging.error(f'Splited training data {data_file_path} for GPU {i} does not exist. Please set the parameter --prepare as True and rerun the code.')
                    exit()
                result = subprocess.getoutput(f'wc -l {data_file_path}')
                total_sample_num += int(result.split(' ')[0])
            logging.info('Skip training data preparation.')
        logging.info(f'{total_sample_num} training samples, {total_sample_num // args.batch_size // args.nGPU} batches in total.')

        if args.nGPU == 1:
            train(None, args)
        else:
            torch.multiprocessing.spawn(train, nprocs=args.nGPU, args=(args,))

    if 'test' in args.mode:
        if args.prepare:
            logging.info('Preparing testing data...')
            total_sample_num = prepare_testing_data(args.test_data_dir, args.nGPU)
        else:
            total_sample_num = 0
            for i in range(args.nGPU):
                data_file_path = os.path.join(args.test_data_dir, f'behaviors_{i}.tsv')
                if not os.path.exists(data_file_path):
                    logging.error(f'Splited testing data {data_file_path} for GPU {i} does not exist. Please set the parameter --prepare as True and rerun the code.')
                    exit()
                result = subprocess.getoutput(f'wc -l {data_file_path}')
                total_sample_num += int(result.split(' ')[0])
            logging.info('Skip testing data preparation.')
        logging.info(f'{total_sample_num} testing samples in total.')

        if args.nGPU == 1:
            test(None, args)
        else:
            torch.multiprocessing.spawn(test, nprocs=args.nGPU, args=(args,))
            
    if 'create_embeddings' == args.mode:
        create_news_embeddings(args.train_data_dir, args.num_words_title)
        create_news_embeddings(args.test_data_dir, args.num_words_title)
        
    if 'read_embeddings' == args.mode:
        read_news_embeddings(args.train_data_dir)
