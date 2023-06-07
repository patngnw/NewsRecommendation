from collections import Counter
from tqdm import tqdm
import numpy as np
#from nltk.tokenize import word_tokenize
import torch
import gzip

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

def get_hidden_states(encoded, num_tokens, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    output = output[1:-1]  # Take out [CLS] and [SEP]
    word_tokens_output = output[:num_tokens]

    return word_tokens_output


def get_word_vector(sent, tokenizer, model, num_tokens, device, layers=[-4, -3, -2, -1]):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt").to(device)
    
    return get_hidden_states(encoded, num_tokens, model, layers)

def write_embedding(f, embedding):
    f.write("\t")
    for j, num in enumerate(embedding):
        if j > 0:
            f.write(",")
        f.write("%.4f" % num)

def create_news_embeddings(data_dir, num_tokens_title):
    from bpemb import BPEmb
    
    # https://github.com/bheinzerling/bpemb
    multibpemb = BPEmb(lang="multi", vs=320000, dim=300)

    doc_id_dict = {}

    embeddings_path = os.path.join(data_dir, "title_embeddings.bpemb.npy.gz")
    news_path = os.path.join(data_dir, 'news.tsv')
    logging.info(f'Read from {news_path}\nWrite embeddings to {embeddings_path}\n')
    
    embeddings_list = []
    embeddings_doc_ids = []
    
    # Add the info for the place holder for Unknown news
    embeddings_list.append(torch.zeros((num_tokens_title, 300)))
    embeddings_doc_ids.append('')
    
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
                
                embeddings_doc_ids.append(doc_id)
                
                # outputs: shape = (no. of tokens in title, 300)
                outputs = multibpemb.embed(title)[:num_tokens_title]
                outputs = np.pad(outputs, ((0, num_tokens_title - outputs.shape[0]), (0, 0)), mode='constant')
                embeddings_list.append(outputs)
                
                if False:
                    if len(embeddings_list) == 50:
                        break
                
    embeddings_all = np.stack(embeddings_list)
    # Flatten the embeddings for each news item, so that we can use it in Torch Embeddings layer
    embeddings_all = embeddings_all.reshape((embeddings_all.shape[0], -1))
    with gzip.GzipFile(embeddings_path, "w") as f:
        np.save(f, embeddings_all)

    output_path = os.path.join(data_dir, 'embeddings_doc_ids.pkl')
    logging.info(f'Writing embeddings_doc_ids to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_doc_ids, f)
    
    output_path = os.path.join(data_dir, 'doc_id_dict.pkl')
    logging.info(f'Writing doc_id_dict to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(doc_id_dict, f)

def create_news_embeddings_bert(data_dir, num_tokens_title):
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device)
    
    doc_id_dict = {}

    embeddings_path = os.path.join(data_dir, "title_embeddings.bert.npy.gz")
    news_path = os.path.join(data_dir, 'news.tsv')
    logging.info(f'Read from {news_path}\nWrite embeddings to {embeddings_path}\n')
    
    embeddings_list = []
    embeddings_doc_ids = []
    
    # Add the info for the place holder for Unknown news
    embeddings_list.append(torch.zeros((1, 768)))
    embeddings_doc_ids.append('')
    
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
                
                embeddings_doc_ids.append(doc_id)
                # outputs: (no. of tokens in title, 768)
                outputs = get_word_vector(title, tokenizer, model, num_tokens_title, device).cpu()
                embeddings_list.append(outputs)
                
                #if len(doc_id_dict) == 100:
                #    break
                
    nt = torch.nested.nested_tensor(embeddings_list)
    embeddings_all = torch.nested.to_padded_tensor(nt, padding=0, output_size=(len(embeddings_list), num_tokens_title, 768))
    embeddings_all = embeddings_all.numpy()
    # Flatten the embeddings for each news item, so that we can use it in Torch Embeddings layer
    embeddings_all = embeddings_all.reshape((embeddings_all.shape[0], -1))
    with gzip.GzipFile(embeddings_path, "w") as f:
        np.save(f, embeddings_all)

    output_path = os.path.join(data_dir, 'embeddings_doc_ids.pkl')
    logging.info(f'Writing embeddings_doc_ids to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_doc_ids, f)
    
    output_path = os.path.join(data_dir, 'doc_id_dict.pkl')
    logging.info(f'Writing doc_id_dict to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(doc_id_dict, f)

def read_news_embeddings(data_dir):    
    embeddings_numpy_path = os.path.join(data_dir, "title_embeddings.bpemb.npy.gz")
    with gzip.GzipFile(embeddings_numpy_path, "r") as f:
        embeddings = np.load(f)
    
    return embeddings

def read_news_embeddings_bert(data_dir):    
    embeddings_numpy_path = os.path.join(data_dir, "title_embeddings.bert.npy.gz")
    with gzip.GzipFile(embeddings_numpy_path, "r") as f:
        embeddings = np.load(f)
    
    return embeddings
