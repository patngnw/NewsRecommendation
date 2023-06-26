import torch
from torch import nn
import torch.nn.functional as F

from .model_utils import AttentionPooling


class NewsEncoder(nn.Module):
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
        if self.args.user_log_mask:
            user_vec = self.attn(news_vecs, log_mask)
        else:
            bz = news_vecs.shape[0]  # batch size
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.args.user_log_length, -1)
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
            user_vec = self.attn(news_vecs)   # batch size, 400

        return user_vec


class Model(torch.nn.Module):
    def __init__(self, args, embedding_matrix, num_category, num_authorid, num_entity, **kwargs):
        super(Model, self).__init__()
        self.args = args
        if args.skip_title:
            word_embedding = None
        else:
            pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
            word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                          freeze=args.freeze_embedding,
                                                          padding_idx=0)

        self.news_encoder = NewsEncoder(args, word_embedding, num_category, num_authorid, num_entity)
        self.user_encoder = UserEncoder(args)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, history, history_mask, candidate, label):
        '''
            history: batch_size, history_length, num_word_title + 3
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''
        news_feature_len = history.shape[-1]
        candidate_news = candidate.reshape(-1, news_feature_len)  # (batch_size * 1+K, news_feature_len)
        candidate_news_vecs = self.news_encoder(candidate_news).reshape(-1, 1 + self.args.npratio, self.args.news_dim)  # batch_size, 1+K, 400

        history_news = history.reshape(-1, news_feature_len)
        history_news_vecs = self.news_encoder(history_news).reshape(-1, self.args.user_log_length, self.args.news_dim)  # batch_size, history_len, 400

        user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        return loss, score