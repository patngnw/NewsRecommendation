from collections import Counter
from tqdm import tqdm
import numpy as np
#from nltk.tokenize import word_tokenize
#from bpemb import BPEmb
from transformers import AutoModel, AutoTokenizer
from utils import update_dict


def read_news(news_path, args, mode='train'):
    news = {}  # Dict: key='news_id, e.g. N1235', value=[ cat, authorid ]
    category_dict = {}  # Dict: key=cat_name, value=idx, 1-based
    authorid_dict = {}  # Dict: key=authorid, value=idx, 1-based
    news_index = {}  # Dict: key=news_id, value=idx, 1-based
    
    with open(news_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, authorid, title, abstract, url, _, _ = splited
            update_dict(news_index, doc_id)

            update_dict(news, doc_id, [category, authorid])
            if mode == 'train':
                if args.use_category:
                    update_dict(category_dict, category)
                if args.use_authorid:
                    update_dict(authorid_dict, authorid)

    if mode == 'train':
        return news, news_index, category_dict, authorid_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'

# news = {}  # Dict: key='news_id, e.g. N1235', value=[ list_of_tokens, cat, authorid ]
# category_dict = {}  # Dict: key=cat_name, value=idx
# authorid_dict = {}  # Dict: key=authorid, value=idx
# news_index = {}  # Dict: key=news_id, value=idx
def get_doc_input(news, news_index, category_dict, authorid_dict, args):
    news_num = len(news) + 1
    news_idx = np.zeros((news_num, 1), dtype='int32')
    news_category = np.zeros((news_num, 1), dtype='int32') if args.use_category else None
    news_authorid = np.zeros((news_num, 1), dtype='int32') if args.use_authorid else None

    for key in tqdm(news):
        category, authorid = news[key]
        doc_index = news_index[key]
        
        news_idx[doc_index, 0] = doc_index

        if args.use_category:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        if args.use_authorid:
            news_authorid[doc_index, 0] = authorid_dict[authorid] if authorid in authorid_dict else 0

    return news_idx, news_category, news_authorid
