from torch.utils.data import IterableDataset, Dataset
import numpy as np
import random
import gzip


class DatasetTrain(IterableDataset):
    def __init__(self, filename, news_index, news_combined, args):
        super(DatasetTrain).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_combined = news_combined
        self.args = args

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):
        line = line.strip().split('\t')
        click_docs = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs), self.args.user_log_length)
        user_feature = self.news_combined[click_docs]  # shape: (user_log_length, 20+1+1+1)

        pos = self.trans_to_nindex(sess_pos)
        neg = self.trans_to_nindex(sess_neg)

        label = random.randint(0, self.args.npratio)
        sample_news = neg[:label] + pos + neg[label:]
        news_feature = self.news_combined[sample_news]

        return user_feature, log_mask, news_feature, label

    def __iter__(self):
        if self.filename.endswith('.gz'):
            file_iter = gzip.open(self.filename, 'rt', encoding='utf-8')
        else:
            file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class DatasetTest(DatasetTrain):
    def __init__(self, filename, news_index, news_vecs, args, baseline_eval=False):
        super(DatasetTrain).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_vecs = news_vecs  # This is the vectors generated from the model's news encoder
        self.args = args
        self.baseline_eval = baseline_eval

    def line_mapper(self, line):
        line = line.strip().split('\t')
        if self.baseline_eval:
            labels = np.array([int(i.split('-')[1]) for i in line[4].split()])  # len: impression_size (e.g. 22), value: 0 or 1
            return None, None, None, labels
        else:
            click_docs = line[3].split()  # User's historical clicked docs
            click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs), self.args.user_log_length)  # len: user_log_length (50)
            user_feature = self.news_vecs[click_docs]  # shape: (50, 400); it is based on the news vectors of all his historical reading

            candidate_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])  # len: impression_size (e.g. 22)
            labels = np.array([int(i.split('-')[1]) for i in line[4].split()])  # len: impression_size (e.g. 22), value: 0 or 1
            candidates_feature = self.news_vecs[candidate_news]  # shape: (num of candidate news, 400); it's the news vectors of all the candidate news

            return user_feature, log_mask, candidates_feature, labels

    def __iter__(self):
        if self.filename.endswith('.gz'):
            file_iter = gzip.open(self.filename, 'rt', encoding='utf-8')
        else:
            file_iter = open(self.filename)

        if self.args.test_rows:
            rows = [next(file_iter) for _ in range(self.args.test_rows)]
            file_iter.close()
            file_iter = rows
            
        return map(self.line_mapper, file_iter)


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]
