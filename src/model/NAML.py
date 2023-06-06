import torch
from torch import nn
import torch.nn.functional as F

from .model_utils import AttentionPooling


class NewsEncoder(nn.Module):
    def __init__(self, args, embedding_matrix, num_category, num_subcategory):
        super(NewsEncoder, self).__init__()
        self.drop_rate = args.drop_rate
        self.num_words_title = args.num_words_title
        self.use_category = args.use_category
        self.use_subcategory = args.use_subcategory
        
        self.title_embeddings = embedding_matrix
        self.title_shorten = nn.Linear(args.bert_emb_dim, args.news_dim)
        
        if args.use_category:
            self.category_emb = nn.Embedding(num_category + 1, args.category_emb_dim, padding_idx=0)
            self.category_dense = nn.Linear(args.category_emb_dim, args.news_dim)
        if args.use_subcategory:
            self.subcategory_emb = nn.Embedding(num_subcategory + 1, args.category_emb_dim, padding_idx=0)
            self.subcategory_dense = nn.Linear(args.category_emb_dim, args.news_dim)
        if args.use_category or args.use_subcategory:
            self.final_attn = AttentionPooling(args.news_dim, args.news_query_vector_dim)
        # self.cnn = nn.Conv1d(
        #     in_channels=args.word_embedding_dim,
        #     out_channels=args.news_dim,
        #     kernel_size=3,
        #     padding=1
        # )
        #self.attn = AttentionPooling(args.news_dim, args.news_query_vector_dim)

    def forward(self, x, mask=None):
        '''
            x: batch_size, hist_feature_num
            mask: batch_size, word_num
        '''
        # title = torch.narrow(x, -1, 0, self.num_words_title).long()  # title.shape: (160, 20)
        # word_vecs = F.dropout(self.embedding_matrix(title),  # word_vecs.shape: (160, 20, 300)
        #                       p=self.drop_rate,
        #                       training=self.training)
        # context_word_vecs = self.cnn(word_vecs.transpose(1, 2)).transpose(1, 2)  # context_word_vecs.shape: (160, 20, 400)
        # title_vecs = self.attn(context_word_vecs, mask)  # title_vecs.shape: (160, 400)
        start = 0
        title_emb = self.title_embeddings(torch.narrow(x, -1, start, 1)).squeeze(dim=1)
        title_vecs = self.title_shorten(title_emb)
        all_vecs = [title_vecs]
        start += 1
        
        if self.use_category:
            category = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            category_vecs = self.category_dense(self.category_emb(category))
            all_vecs.append(category_vecs)
            start += 1
        if self.use_subcategory:
            subcategory = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            subcategory_vecs = self.subcategory_dense(self.subcategory_emb(subcategory))
            all_vecs.append(subcategory_vecs)

        if len(all_vecs) == 1:
            news_vecs = all_vecs[0]
        else:
            all_vecs = torch.stack(all_vecs, dim=1)
            news_vecs = self.final_attn(all_vecs)
        return news_vecs


class UserEncoder(nn.Module):
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
        bz = news_vecs.shape[0]
        if self.args.user_log_mask:
            user_vec = self.attn(news_vecs, log_mask)
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.args.user_log_length, -1)
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
            user_vec = self.attn(news_vecs)
        return user_vec


class Model(torch.nn.Module):
    def __init__(self, args, news_embeddings_weight, num_category, num_subcategory, **kwargs):
        super(Model, self).__init__()
        self.args = args
        pretrained_embedding = torch.from_numpy(news_embeddings_weight).float()
        news_embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                      freeze=args.freeze_embedding,
                                                      padding_idx=0)

        self.news_encoder = NewsEncoder(args, news_embedding, num_category, num_subcategory)
        self.user_encoder = UserEncoder(args)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, history, history_mask, candidate, label):
        '''
            history: batch_size, history_length, hist_feature_len
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, hist_feature_len   (K = npratio = 4)
            label: batch_size
        '''
        hist_feature_len = history.shape[-1]
        candidate_news = candidate.reshape(-1, hist_feature_len) # shape: batch_size * (1+K), hist_feature_len
        candidate_news_vecs = self.news_encoder(candidate_news).reshape(-1, 1 + self.args.npratio, self.args.news_dim) # shape: 32, 5, news_dim

        history_news = history.reshape(-1, hist_feature_len)
        history_news_vecs = self.news_encoder(history_news).reshape(-1, self.args.user_log_length, self.args.news_dim) # shape: 32, 50, news_dim

        user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        return loss, score
