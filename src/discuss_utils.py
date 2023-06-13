from pathlib import Path
import pandas as pd
import sys
import os
import numpy as np
import logging
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


_src_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def get_impressions_long_df(impression_path, nrows=None):
    df = pd.read_csv(impression_path, dtype={'user_id': str}, nrows=nrows)
    df = df.drop_duplicates(subset=['time', 'user_id'])
    df = df[['time', 'user_id', 'tids']]
    
    # time,user_id,nwtc,uid,so,tids
    
    df = df.rename(columns={'tids': 'impressions'})
    df = df.assign(impressions=df.impressions.str.split('_')).explode('impressions')
    df['impressions'] = df.impressions.astype(int)
    
    tids_impressions = df['impressions'].unique()
    
    # remember the order of each news after explosion
    # df['imp_idx'] = df.groupby(['time', 'user_id']).cumcount()
    
    df['date'] = df.time.str.slice(0, 10).astype(str)
    return df, tids_impressions
    
def get_reads(reads_path, nrows=None):
    df = pd.read_csv(reads_path, dtype={'tid': int, 'user_id': str}, nrows=nrows)
    
    # time,"user_id",nwtc,uid,so,tid
    df = df[['time', 'user_id', 'tid']]
    df['date'] = df.time.str.slice(0, 10).astype(str)
    df = df.drop_duplicates(['user_id', 'tid', 'date'])
    
    tids_reads = df['tid'].unique()
    
    return df, tids_reads

# Ref:
# https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md
def get_behaviors_df(df_impressions, df_reads):
    df_reads = df_reads.assign(clicked=1)
    df_impressions =  df_impressions.rename(columns={'impressions': 'tid'})  # for merging
    df = pd.merge(df_impressions, df_reads[['user_id', 'tid', 'date', 'clicked']], on=['user_id', 'date', 'tid'], how='left')
    df.loc[pd.isna(df.clicked), ['clicked']] = 0
    df['clicked'] = df['clicked'].astype(int)
    
    df['impression'] = df['tid'].astype(str) + "-" + df['clicked'].astype(str)
    
    df_wide = df[['time', 'date', 'user_id', 'impression']].groupby(['time', 'date', 'user_id'])['impression'].apply(
        lambda x: ' '.join(x)).reset_index()
    
    # Now we have to calculate the field History
    df_with_hist_list = []
    
    impression_dates = sorted(df_impressions['date'].unique().tolist())
    for imp_date in impression_dates:
        df_with_hist = pd.merge(df_wide.loc[df_wide.date == imp_date], 
                                df_reads.loc[df_reads.date <= imp_date][['time', 'user_id', 'tid']], 
                                on=['user_id'])
        df_with_hist = df_with_hist.query('time_y < time_x')
        df_with_hist = df_with_hist.sort_values(['user_id', 'time_x', 'time_y'])
        df_with_hist = df_with_hist.drop(['time_y', 'date'], axis='columns')
        df_with_hist = df_with_hist.rename(columns={'time_x': 'time'})
        df_with_hist = df_with_hist.groupby(['time', 'user_id', 'impression'])['tid'].apply(lambda x: ' '.join(map(str, x))).reset_index()
        df_with_hist = df_with_hist.rename(columns={'tid': 'history'})
        
        df_with_hist_list.append(df_with_hist)
        
    df_with_hist_all = pd.concat(df_with_hist_list)
    
    df_res = pd.merge(df_wide, df_with_hist_all[['time', 'user_id', 'history']], on=['time', 'user_id'], how='left')
    
    # Drop duplicates for the same day
    df_res['date'] = df_res.time.str.slice(0, 10).astype(str)
    df_res = df_res.drop_duplicates(['date', 'user_id', 'history', 'impression'])
    df_res['history'] = df_res['history'].fillna('')
    df_res = df_res.sort_values(['time']).reset_index()
    
    return df_res


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def get_tids_info(tids):
    connector_dir = Path("/opt/nwdata/data-playground/connector/src")
    sys.path.append(str(connector_dir))
    import mysql_conn

    sql_conn = mysql_conn.mysql_conn(product='discuss').get_conn_sqlalchemy()
    df_list = []

    for chunk in chunks(tids, 500):
        tids_str = ",".join([str(t) for t in chunk])
        
        query = f"""
SELECT t.tid, t.fid, f.name as forum,
        t.authorid,
        date_format(from_unixtime(t.dateline), "%Y-%m-%d") AS create_date,
        t.subject
FROM cdb_threads t join cdb_forums f on t.fid=f.fid

WHERE tid in ({tids_str})
"""
        df = pd.read_sql_query(query, sql_conn)
        df_list.append(df)
        
    df_tids_info = pd.concat(df_list)
    df_tids_info['subject'] = df_tids_info['subject'].str.replace(r'\s+|\n', ' ', regex=True)
    return df_tids_info


_behaviors_tsv = 'behaviors.tsv'
_behaviors_header = ['user_id', 'time', 'history', 'impression']
_news_tsv = 'news.tsv'
_news_header = ['tid', 'forum', 'authorid', 'subject', 'abstract', 'url', 'title_entities', 'abstract_entities']
_tids_info_csv = 'tids_info.csv'

def gen_discuss_data(data_dir, nrows=None):
    data_dir = Path(data_dir)
    
    logging.info('Generating data for behaviors.tsv')
    df_impressions, tids_impressions = get_impressions_long_df(data_dir / 'discuss-rs-tids.csv', nrows=nrows)
    df_reads, tids_reads = get_reads(data_dir / 'discuss-rs-reads.csv', nrows=nrows)

    output_dir = data_dir
    if nrows:
        output_dir = output_dir / f'nrows_{nrows}'
        
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Outputdir = {output_dir}')
        
    df_with_hist = get_behaviors_df(df_impressions, df_reads)    
    df_with_hist[_behaviors_header].to_csv(output_dir / _behaviors_tsv, sep='\t', 
                                                                      index=True,   # Need the Impression ID
                                                                      header=None)
    
    logging.info('Generating data for news.tsv')
    
    logging.info('Saving involved tids to tids_all.csv')
    tids_all = sorted(list(set(tids_impressions.tolist()).union(set(tids_reads.tolist()))))
    pd.DataFrame({'tid': tids_all}).to_csv(output_dir / 'tids_all.csv', index=False)

    output_path = output_dir / _tids_info_csv
    logging.info(f'Saving tids info to {output_path}')
    df_tids_info = get_tids_info(tids_all)
    df_tids_info.to_csv(output_path, index=False, encoding='utf-8')

    # Save to news.tsv
    df_tids_info['abstract'] = ''
    df_tids_info['url'] = ''
    df_tids_info['title_entities'] = ''
    df_tids_info['abstract_entities'] = ''
    df_tids_info[_news_header].to_csv(output_dir / _news_tsv, sep='\t', index=False, header=None, encoding='utf-8')


def get_seen_tids(df_behavior):
    tids_all = set()
    for _, row in df_behavior.iterrows():
        history = row['history']
        if pd.isna(history):
            history = []
        else:
            history = history.split()
            
        impression = [x.split('-')[0] for x in row['impression'].split()]
        
        history.extend(impression)
        tids = [int(x) for x in history]
        tids_all.update(tids)
        
    return sorted(list(tids_all))
        

def split_data(start_date, test_date, data_dir, train_data_dir, test_data_dir, frac=None):
    assert test_date > start_date
    data_dir = Path(data_dir)
    train_data_dir = Path(train_data_dir)
    os.makedirs(train_data_dir, exist_ok=True)
    test_data_dir = Path(test_data_dir)
    os.makedirs(test_data_dir, exist_ok=True)
    
    df_behaviors = pd.read_csv(data_dir / _behaviors_tsv, delimiter='\t', names=['id'] + _behaviors_header)
    if frac:
        df_behaviors = df_behaviors.sample(frac=frac)
    df_behaviors['date'] = df_behaviors.time.str.slice(0, 10).astype(str)
    
    output_path = train_data_dir / (_behaviors_tsv + '.gz')
    logging.info(f'Writing to {output_path}')
    df_behaviors_train = df_behaviors.loc[(df_behaviors.date >= start_date) & (df_behaviors.date < test_date)]
    df_behaviors_train[_behaviors_header].to_csv(output_path, sep='\t', index=True, header=None, encoding='utf-8')
    
    output_path = test_data_dir / (_behaviors_tsv + '.gz')
    logging.info(f'Writing to {output_path}')
    df_behaviors_test = df_behaviors.loc[df_behaviors.date == test_date]
    df_behaviors_test[_behaviors_header].to_csv(output_path, sep='\t', index=True, header=None, encoding='utf-8')

    df_news_all = pd.read_csv(data_dir / _news_tsv, delimiter='\t', names=_news_header, dtype={'tid': int})
    
    logging.info('Splitting the news...')
    for df, output_dir in zip([df_behaviors_train, df_behaviors_test], [train_data_dir, test_data_dir]):
        tids = get_seen_tids(df)
        
        output_path = output_dir / 'seen_tids.csv'
        logging.info(f'Writing to {output_path}')
        pd.DataFrame({'tid': tids}).to_csv(output_path, index=False)

        df_news = df_news_all.loc[df_news_all.tid.isin(tids)]
        
        output_path = output_dir / _news_tsv
        logging.info(f'Writing to {output_path}')
        df_news[_news_header].to_csv(output_path, sep='\t', index=False, header=None, encoding='utf-8')
        
        
_seen_tids_csv = 'seen_tids.csv'
def save_seen_tids(data_dir):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / _news_tsv, delimiter='\t', names=_news_header, dtype={'tid': int})
    output_path = data_dir / _seen_tids_csv
    logging.info(f'Writing to {output_path}')
    df[['tid']].to_csv(output_path, index=False)
    
    
def regen_test_dev_news_tsv(base_data_dir, data_dir):
    base_data_dir = Path(base_data_dir)
    data_dir = Path(data_dir)
    
    df_tids_info = pd.read_csv(base_data_dir / _tids_info_csv)
    
    df_tids_info['abstract'] = ''
    df_tids_info['url'] = ''
    df_tids_info['title_entities'] = ''
    df_tids_info['abstract_entities'] = ''

    df_seen_tids = pd.read_csv(data_dir / _seen_tids_csv)
    
    output_path = data_dir / _news_tsv
    logging.info(f'Writing to {output_path}')    
    df_tids_info.loc[df_tids_info.tid.isin(df_seen_tids.tid) , _news_header]\
        .to_csv(output_path, sep='\t', index=False, header=None, encoding='utf-8')
        
        
def split_dev_behaviors(train_data_dir, test_data_dir):
    train_data_dir = Path(train_data_dir)
    test_data_dir = Path(test_data_dir)
    
    input_path = train_data_dir / 'behaviors_np4_0.tsv.gz'
    logging.info(f'Reading {input_path}')
    df_train = pd.read_csv(input_path, sep='\t', names=['id'] + _behaviors_header + ['dummy'])
    
    input_path = test_data_dir / 'behaviors_0.tsv.gz'
    logging.info(f'Reading {input_path}')
    df_test = pd.read_csv(input_path, sep='\t', names=['id'] + _behaviors_header)
    
    train_user_ids = set(df_train.user_id.unique())
    test_user_ids = set(df_test.user_id.unique())
    
    logging.info(f'Size of test_user_ids = {len(test_user_ids)}')

    output_path = test_data_dir / 'behaviors_0.seen_users.tsv.gz'
    logging.info(f'Writing to {output_path}')  
    seen_user_ids = test_user_ids.intersection(train_user_ids) 
    logging.info(f'Size of seen_user_ids = {len(seen_user_ids)}')
    df_test.loc[df_test.user_id.isin(seen_user_ids)].to_csv(output_path, sep="\t", index=False, header=None, encoding='utf-8')
    
    output_path = test_data_dir / 'behaviors_0.unseen_users.tsv.gz'
    logging.info(f'Writing to {output_path}')  
    unseen_user_ids = test_user_ids.difference(train_user_ids) 
    logging.info(f'Size of unseen_user_ids = {len(unseen_user_ids)}')
    df_test.loc[df_test.user_id.isin(unseen_user_ids)].to_csv(output_path, sep="\t", index=False, header=None, encoding='utf-8')
    

_embeddings_npyz = 'embeddings.npz'
def create_bert_embeddings_file(data_dir):
    data_dir = Path(data_dir)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    bert = AutoModel.from_pretrained('bert-base-chinese')

    df = pd.read_csv(data_dir / _news_tsv, delimiter='\t', names=_news_header, dtype={'tid': int})
    embedding_list = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing rows'):
        title = row['subject']
        title = tokenizer(
                    title,
                    return_tensors='pt',
                    return_token_type_ids=False,
                    truncation=True,
                )
        embedding = bert(**title)[0][:, 0][0, :].detach().numpy()
        embedding_list.append(embedding)

    embeddings = np.stack(embedding_list)
    output_path = data_dir / _embeddings_npyz
    logging.info(f'Writing output to {output_path}')
    np.savez(output_path, embeddings=embeddings)


def load_bert_embeddings_file(data_dir):    
    data_dir = Path(data_dir)
    input_path = data_dir / _embeddings_npyz
    embeddings = np.load(input_path)['embeddings']
    
    df = pd.read_csv(data_dir / _news_tsv, delimiter='\t', names=_news_header, dtype={'tid': int}, usecols=['tid'])
    tid2idx = { v: k for k, v in enumerate(df['tid'].to_list()) }
    
    return embeddings, tid2idx

        
# def regen_test_dev_news_tsv_for_authorid(base_data_dir, data_dir):
#     base_data_dir = Path(base_data_dir)
#     data_dir = Path(data_dir)
    
#     df_tids_info = pd.read_csv(base_data_dir / _tids_info_csv)
    
#     df_tids_info['subcat'] = df_tids_info['authorid']
#     df_tids_info['abstract'] = ''
#     df_tids_info['url'] = ''
#     df_tids_info['title_entities'] = ''
#     df_tids_info['abstract_entities'] = ''

#     df_seen_tids = pd.read_csv(data_dir / _seen_tids_csv)
    
#     output_path = data_dir / _news_tsv
#     logging.info(f'Writing to {output_path}')    
#     df_tids_info.loc[df_tids_info.tid.isin(df_seen_tids.tid) , _news_header]\
#         .to_csv(output_path, sep='\t', index=False, header=None, encoding='utf-8')
