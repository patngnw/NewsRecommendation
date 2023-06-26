from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import sys
import pathlib
import logging
import numpy as np
from tqdm.auto import tqdm

from .model_utils import AttentionPooling

_srcdir = pathlib.Path(__file__).parent
sys.path.append(str(_srcdir / '..'))
from preprocess import read_news, get_news_input_matrix
from dataset import DatasetTrain, NewsDataset, NewsDataset, DatasetTest
import utils
from metrics import roc_auc_score, ndcg_score, hit_score


class NewsEncoder(torch.nn.Module):
    def __init__(self, args, embedding_matrix, num_category, num_authorid, num_entity):
        super(NewsEncoder, self).__init__()
        self.num_words_title = args.num_words_title
        self.use_category = args.use_category
        self.use_authorid = args.use_authorid
        self.use_entity = args.use_entity

        if args.use_category:
            self.category_emb = nn.Embedding(num_category + 1, args.category_emb_dim, padding_idx=0)
            self.category_dense = nn.Linear(args.category_emb_dim, args.news_dim)

        if args.use_authorid:
            self.authorid_emb = nn.Embedding(num_authorid + 1, args.category_emb_dim, padding_idx=0)
            self.authorid_dense = nn.Linear(args.category_emb_dim, args.news_dim)

        if args.use_entity:
            self.entity_emb = nn.Embedding(num_entity + 1, args.category_emb_dim, padding_idx=0)
            self.entity_dense = nn.Linear(args.category_emb_dim, args.news_dim)

        if args.use_category or args.use_authorid or args.use_entity:
            self.final_attn = AttentionPooling(args.news_dim, args.news_query_vector_dim)

        self.skip_title = args.skip_title
        if not args.skip_title:
            self.embedding_matrix = embedding_matrix
            self.drop_rate = args.drop_rate
            self.cnn = nn.Conv1d(
                in_channels=args.word_embedding_dim,
                out_channels=args.news_dim,
                kernel_size=args.conv1d_kernel_size,
                padding=1
            )
            self.attn = AttentionPooling(args.news_dim, args.news_query_vector_dim)


    def forward(self, x, mask=None):
        '''
            x: batch_size * (1+K), news_feature_length (i.e. 23)
            mask:
        '''
        if self.skip_title:
            all_vecs = []
        else:
            title = torch.narrow(x, -1, 0, self.num_words_title).long()  # shape: 160, word_num
            word_vecs = F.dropout(self.embedding_matrix(title),  # self.embedding_matrix(title): (160, 20, 300)
                                  p=self.drop_rate,
                                  training=self.training)  # word_vecs: (160, 20, 300)
            context_word_vecs = self.cnn(word_vecs.transpose(1, 2)).transpose(1, 2)  # context_word_vecs: (160, 20, 400)
            title_vecs = self.attn(context_word_vecs, mask)  # title_vecs: (160, 400)
            all_vecs = [title_vecs]

        start = self.num_words_title
        if self.use_category:
            category = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()  # category: (160,)
            category_vecs = self.category_dense(self.category_emb(category))  # category_vecs: (160, 400)
            all_vecs.append(category_vecs)
            start += 1

        if self.use_authorid:
            authorid = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            authorid_vecs = self.authorid_dense(self.authorid_emb(authorid))
            all_vecs.append(authorid_vecs)
            start += 1

        if self.use_entity:
            entity = torch.narrow(x, -1, start, 1).long().reshape(-1, 1).squeeze()
            entity_vecs = self.entity_dense(self.entity_emb(entity))
            all_vecs.append(entity_vecs)
            start += 1

        if len(all_vecs) == 1:
            news_vecs = all_vecs[0]
        else:
            all_vecs = torch.stack(all_vecs, dim=1)
            news_vecs = self.final_attn(all_vecs)

        return news_vecs  # shape: 128, 400


class UserEncoder(torch.nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.attn = AttentionPooling(args.news_dim, args.user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
        '''
        if self.args.user_log_mask:
            user_vec = self.attn(news_vecs, log_mask)
        else:
            bz = news_vecs.shape[0]  # batch size
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.args.user_log_length, -1)
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
            user_vec = self.attn(news_vecs)   # batch size, 400

        return user_vec


class TrainDataModule(pl.LightningDataModule):
    def __init__(self, args, rank, news_index, news_combined):
        super().__init__()
        self.args = args
        self.rank = rank
        self.news_index = news_index
        self.news_combined = news_combined
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        args = self.args
        data_file_path = os.path.join(args.train_data_dir, f'behaviors_np{args.npratio}_{self.rank}.tsv.gz')
        dataset = DatasetTrain(data_file_path, self.news_index, self.news_combined, args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        return dataloader


class TestDataModule(pl.LightningDataModule):
    def __init__(self, args, model, rank, category_dict, authorid_dict, entity_dict, word_dict):
        super().__init__()
        self.args = args
        self.model = model
        self.rank = rank
        self.category_dict = category_dict
        self.authorid_dict = authorid_dict
        self.entity_dict = entity_dict
        self.word_dict = word_dict  
            
    def gen_vecs_from_news_encoder(self, news_path, model, rank, category_dict, authorid_dict, entity_dict, word_dict, args):
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
                #if args.enable_gpu:
                #    input_ids = input_ids.cuda(rank)
                # input_ids: shape = (128, 22)    
                candidate_news_vec = model.news_encoder(input_ids)
                candidate_news_vec = candidate_news_vec.to(torch.device("cpu")).detach().numpy()
                news_vecs.extend(candidate_news_vec)  # news_vec: (128, 400)

        news_vecs = np.array(news_vecs)  # shape:  (num_of_news, 400)
        return news_vecs, news_index    
        
        
    def get_test_behavior_path(self):
        data_file_path = os.path.join(self.args.test_data_dir, f'behaviors_{self.rank}.tsv.gz')
        if self.args.test_users == 'seen':
            data_file_path = data_file_path.replace('.tsv', '.seen_users.tsv')
        elif self.args.test_users == 'unseen':
            data_file_path = data_file_path.replace('.tsv', '.unseen_users.tsv')
            
        return data_file_path

        
    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_news_vecs, news_index = self.gen_vecs_from_news_encoder(
            os.path.join(self.args.test_data_dir, 'news.tsv'),
            self.model, self.rank, self.category_dict, self.authorid_dict, 
            self.entity_dict, self.word_dict, self.args)
    
        logging.info("news scoring num: {}".format(test_news_vecs.shape[0]))

        data_file_path = self.get_test_behavior_path()
        logging.info(f'Behavior file: {data_file_path}')

        def collate_fn(tuple_list):  # len(tuple_list) = batch_size
            log_vecs = torch.FloatTensor(np.array([x[0] for x in tuple_list]))
            log_mask = torch.FloatTensor(np.array([x[1] for x in tuple_list]))
            news_vecs = [x[2] for x in tuple_list]
            labels = [x[3] for x in tuple_list]
            return (log_vecs, log_mask, news_vecs, labels)

        dataset = DatasetTest(data_file_path, news_index, test_news_vecs, self.args)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=collate_fn)

        return dataloader


def get_mean(arr):
    return [np.array(i).mean() for i in arr]


class Model(pl.LightningModule):
    def __init__(self, args, num_category, num_authorid, num_entity, word_dict):
        super().__init__()

        self.args = args
                
        if args.skip_title:
            word_embedding = None
        else:
            embedding_matrix, _ = utils.load_matrix(args.bpemb_embedding_path,
                                                            word_dict,
                                                            args.word_embedding_dim)
            pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
            word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                          freeze=args.freeze_embedding,
                                                          padding_idx=0)

        self.news_encoder = NewsEncoder(args, word_embedding, num_category, num_authorid, num_entity)
        self.user_encoder = UserEncoder(args)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.AUC = []
        self.HIT5 = []
        self.HIT10 = []
        self.nDCG5 = []
        self.nDCG10 = []

        
        
    def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.args.lr)


    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        '''
            history: batch_size, history_length, num_word_title + 3
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''
        history, history_mask, candidate, label = batch
        
        news_feature_len = history.shape[-1]
        candidate_news = candidate.reshape(-1, news_feature_len)  # (batch_size * 1+K, news_feature_len)
        candidate_news_vecs = self.news_encoder(candidate_news).reshape(-1, 1 + self.args.npratio, self.args.news_dim)  # batch_size, 1+K, 400

        history_news = history.reshape(-1, news_feature_len)
        history_news_vecs = self.news_encoder(history_news).reshape(-1, self.args.user_log_length, self.args.news_dim)  # batch_size, history_len, 400

        user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        accuracy = utils.acc(label, score)
        self.log("train_loss", loss, prog_bar=True)
        self.log("accuracy", accuracy, prog_bar=True)
        return loss
    
    def infer_score_vec(self, user_vec, candidate_news_vec):
        score_vec = np.dot(candidate_news_vec, user_vec)  # candidate_news_vec: (20, 400); user_vec: (400,); score: (20,)
        if self.args.jitao_score_method:
            score_sorted_idx = np.flip(np.argsort(score_vec))  # Idx of item if we sort it in desc order
            score_by_candidate_news = list(range(candidate_news_vec.shape[0], 0, -1))
            for idx in score_sorted_idx[:self.args.jitao_topn]:
                score_by_candidate_news[idx] += self.args.jitao_boost
                        
            score_vec = score_by_candidate_news
            
        return score_vec
    
    def update_metrics(self, label_vec, score_vec):
        auc = roc_auc_score(label_vec, score_vec)
        hit5 = hit_score(label_vec, score_vec, k=5)
        hit10 = hit_score(label_vec, score_vec, k=10)
        ndcg5 = ndcg_score(label_vec, score_vec, k=5)
        ndcg10 = ndcg_score(label_vec, score_vec, k=10)

        self.AUC.append(auc)
        self.HIT5.append(hit5/100)
        self.HIT10.append(hit10/100)
        self.nDCG5.append(ndcg5)
        self.nDCG10.append(ndcg10)

    
    def test_step(self, batch, batch_idx):
        # log_vecs: (batch_size, 50, 400), user feature based on clicked doc history
        # log_mask: log_vecs are padded; log_mask tells which items are padded (0) or not (1)
        # candidate_news_vecs: len=batch_size; vectors (from news_encoder) of all candidate news (from that sample row)
        # label_vecs:len=batch_size; each item is a list and contain the label (0/1) of each candidate news
        log_vecs, log_mask, candidate_news_vecs, label_vecs = batch
        user_vecs = self.user_encoder(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()
        for inner_cnt, user_vec, candidate_news_vec, label_vec in zip(range(len(user_vecs)), user_vecs, candidate_news_vecs, label_vecs):
            
            if label_vec.mean() == 0 or label_vec.mean() == 1:
                continue

            score_vec = self.infer_score_vec(user_vec, candidate_news_vec)

            self.update_metrics(label_vec, score_vec)

        self.log("avg_AUC", np.array(self.AUC).mean())
        self.log("avg_HIT5", np.array(self.HIT5).mean())
        self.log("avg_HIT10", np.array(self.HIT10).mean())
        self.log("avg_nDCG5", np.array(self.nDCG5).mean())
        self.log("avg_nDCG10", np.array(self.nDCG10).mean())

