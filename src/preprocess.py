from collections import Counter
from tqdm import tqdm
import numpy as np
from nltk.tokenize import word_tokenize


def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value


def read_news(news_path, args, mode='train'):
    news = {}
    category_dict = {}
    subcategory_dict = {}
    news_index = {}
    #word_cnt = Counter()

    with open(news_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract, url, _, _ = splited
            update_dict(news_index, doc_id)

            #title = title.lower()
            #title = word_tokenize(title)
            update_dict(news, doc_id, [category, subcategory])
            if mode == 'train':
                if args.use_category:
                    update_dict(category_dict, category)
                if args.use_subcategory:
                    update_dict(subcategory_dict, subcategory)
                #word_cnt.update(title)

    if mode == 'train':
        #word = [k for k, v in word_cnt.items() if v > args.filter_num]
        #word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
        return news, news_index, category_dict, subcategory_dict #, word_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'

# news: dict: key=doc_id, value=[cat, subcat]
# news_index: dict: key=doc_id, value=idx (1-based)
def get_doc_input(news, news_index, category_dict, subcategory_dict, args):
    news_num = len(news) + 1  # One extra is for index-0 - the zero embedding for unknown news
    news_title = np.zeros((news_num, 1), dtype='int32')
    news_category = np.zeros((news_num, 1), dtype='int32') if args.use_category else None
    news_subcategory = np.zeros((news_num, 1), dtype='int32') if args.use_subcategory else None

    for doc_id in tqdm(news):
        category, subcategory = news[doc_id]
        doc_index = news_index[doc_id]

        #for word_id in range(min(args.num_words_title, len(title))):
        #    if title[word_id] in word_dict:
        #        news_title[doc_index, word_id] = word_dict[title[word_id]]

        # TODO: It's lookup into the Bert embedding layer; right now it's identical to doc_index
        news_title[doc_index, 0] = doc_index
        
        if args.use_category:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        if args.use_subcategory:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0

    return news_title, news_category, news_subcategory


from transformers import AutoTokenizer, AutoModel
import logging
import os
import pickle

def write_embedding(f, embedding):
    f.write("\t")
    for j, num in enumerate(embedding):
        if j > 0:
            f.write(",")
        f.write("%.4f" % num)

def create_news_embeddings(data_dir):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    
    doc_id_dict = {}

    embeddings_path = os.path.join(data_dir, "title_embeddings.txt")
    news_path = os.path.join(data_dir, 'news.tsv')
    logging.info(f'Read from {news_path}\nWrite embeddings to {embeddings_path}\n')
    
    with open(news_path, 'r', encoding='utf-8') as f_in:
        with open(embeddings_path, 'w') as f:
            for line in tqdm(f_in):
                splited = line.strip('\n').split('\t')
                doc_id, category, subcategory, title, abstract, url, _, _ = splited
                
                # Note:
                # The first doc_id will get an index of 1.
                # We reserve index 0 for unknown news, and when we create nn.Embedding, we will set num_embeddings to num_news+1, and 
                # padding_idx to 0
                update_dict(doc_id_dict, doc_id)
                
                f.write(doc_id)
                tokens = tokenizer(title, return_tensors="pt", max_length=200, truncation=True, padding=True)
                outputs = model(**tokens)
                write_embedding(f, outputs[0].tolist()[0][0])   
                f.write('\n')
                
                #if len(doc_id_dict)>100:
                #    print(f'embedding length: {len(outputs[0].tolist()[0][0])}')
                #    break
                
    output_path = os.path.join(data_dir, 'doc_id_dict.pkl')
    logging.info(f'Writing doc_id_dict to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(doc_id_dict, f)


import torch

def read_news_embeddings(data_dir, news_dim):    
    embeddings_numpy_path = os.path.join(data_dir, "title_embeddings.npy")
    if os.path.exists(embeddings_numpy_path):
        return np.load(embeddings_numpy_path)
    else:
        doc_id_dict = pickle.load(open(os.path.join(data_dir, 'doc_id_dict.pkl'), 'rb'))
        embeddings_path = os.path.join(data_dir, "title_embeddings.txt")
        
        embeddings = read_embeddings_from_file(embeddings_path, doc_id_dict, news_dim)
        np.save(embeddings_numpy_path, embeddings)
        return embeddings


def read_embeddings_from_file(embeddings_path, doc_id_dict, news_dim):
    print(f"Reading data from {embeddings_path}")
    
    # Ref: https://stackoverflow.com/a/1019572/5552903
    with open(embeddings_path, "rbU") as f:
        # First find out the number of lines in the file
        num_lines = sum(1 for _ in f)
        
    embeddings = np.zeros((num_lines + 1, news_dim))        
    with open(embeddings_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(tqdm(f)):
            embeddings_data = line.split("\t")
            doc_id = embeddings_data[0]
            title_embedding = [ float(i) for i in embeddings_data[1].split(",") ]
            news_index = doc_id_dict[doc_id]
            embeddings[news_index] = title_embedding
    
    return embeddings

            

