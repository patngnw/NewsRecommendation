from collections import Counter
from tqdm import tqdm
import numpy as np
#from nltk.tokenize import word_tokenize
from bpemb import BPEmb
from utils import update_dict


def read_news(news_path, args, mode='train'):
    news = {}  # Dict: key='news_id, e.g. N1235', value=[ list_of_tokens, cat, authorid ]
    category_dict = {}  # Dict: key=cat_name, value=idx
    authorid_dict = {}  # Dict: key=authorid, value=idx
    entity_dict = {}
    news_index = {}  # Dict: key=news_id, value=idx
    word_cnt = Counter()  # Count of appearance of each token in training data
    
    multibpemb = BPEmb(lang="multi", vs=320000, dim=300)

    with open(news_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                splited = line.strip('\n').split('\t')
                doc_id, category, authorid, title, abstract, url, title_entities, _ = splited
            except Exception:
                print(f'line: {line}')
                if args.use_authorid:
                    assert len(authorid) > 0
                raise
            update_dict(news_index, doc_id)

            title = title.lower()
            title = multibpemb.encode(title)  # list: (num of token in title)
            title_entities = [] if len(title_entities) == 0 else title_entities.split(',')
            update_dict(news, doc_id, [title, category, authorid, title_entities])
            if mode == 'train':
                if args.use_category:
                    update_dict(category_dict, category)
                if args.use_authorid:
                    update_dict(authorid_dict, authorid)
                if args.use_entity:
                    for entity in title_entities:
                        update_dict(entity_dict, entity)
                word_cnt.update(title)

    if mode == 'train':
        word = [k for k, v in word_cnt.items() if v > args.filter_num]  # All the words which appears at least args.filter_num times
        word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}  # Dict: key=word, value=idx (self-created)
        return news, news_index, category_dict, authorid_dict, word_dict, entity_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'

# news = {}  # Dict: key='news_id, e.g. N1235', value=[ list_of_tokens, cat, authorid ]
# category_dict = {}  # Dict: key=cat_name, value=idx
# authorid_dict = {}  # Dict: key=authorid, value=idx
# news_index = {}  # Dict: key=news_id, value=idx
# word_cnt = Counter()  # Count of appearance of each token in training data
# word_dict = {}: key=word, value=idx (self-created)
def get_doc_input(news, news_index, category_dict, authorid_dict, entity_dict, word_dict, args):
    news_num = len(news) + 1
    news_title = np.zeros((news_num, args.num_words_title), dtype='int32')  # Value = an idx of that word, read from word_dict
    news_category = np.zeros((news_num, 1), dtype='int32') if args.use_category else None
    news_authorid = np.zeros((news_num, 1), dtype='int32') if args.use_authorid else None
    news_entity = np.zeros((news_num, 1), dtype='int32') if args.use_entity else None

    for key in tqdm(news):
        title, category, authorid, entities = news[key]
        doc_index = news_index[key]

        for word_id in range(min(args.num_words_title, len(title))):
            if title[word_id] in word_dict:
                news_title[doc_index, word_id] = word_dict[title[word_id]]

        if args.use_category:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        if args.use_authorid:
            news_authorid[doc_index, 0] = authorid_dict[authorid] if authorid in authorid_dict else 0
        if args.use_entity:
            # Use the 1st one
            if len(entities) > 0:
                entity = entities[0]
            else:
                entity = 0
            news_entity[doc_index, 0] = entity_dict.get(entity, 0)

    return news_title, news_category, news_authorid, news_entity
