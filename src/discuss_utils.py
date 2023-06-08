from pathlib import Path
import pathlib
import pandas as pd
import sys
import os
import yaml
import numpy as np

connector_dir = Path("/opt/nwdata/data-playground/connector/src")
sys.path.append(str(connector_dir))
import mysql_conn

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
    df_res['history'] = df_res['history'].fillna('')
    
    df_res = df_res.sort_values(['time']).reset_index()
    
    return df_res


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def get_tids_info(tids):
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
    return df_tids_info
    
    
def gen_discuss_data():
    nrows = None
    print('Generating data for behaviors.tsv')
    df_impressions, tids_impressions = get_impressions_long_df(_src_dir / '../data/discuss/discuss-rs-tids.csv', nrows=nrows)
    df_reads, tids_reads = get_reads(_src_dir / '../data/discuss/discuss-rs-reads.csv', nrows=nrows)
    df_with_hist = get_behaviors_df(df_impressions, df_reads)    
    df_with_hist[['user_id', 'time', 'history', 'impression']].to_csv(_src_dir / '../data/discuss/behaviors.tsv', sep='\t', 
                                                                      index=True, 
                                                                      header=None)
    
    print('Generating data for news.tsv')
    
    print('Saving involved tids to tids_all.csv')
    tids_all = sorted(list(set(tids_impressions.tolist()).union(set(tids_reads.tolist()))))
    pd.DataFrame({'tid': tids_all}).to_csv(_src_dir / '../data/discuss/tids_all.csv', index=False)
    
    print('Saving tids info to tids_info.csv')
    df_tids_info = get_tids_info(tids_all)
    df_tids_info.to_csv(_src_dir / '../data/discuss/tids_info.csv', index=False, encoding='utf-8')

    # Save to news.tsv
    df_tids_info['subcat'] = ''
    df_tids_info['abstract'] = ''
    df_tids_info['url'] = ''
    df_tids_info['title_entities'] = ''
    df_tids_info['abstract_entities'] = ''
    df_tids_info[['tid', 'forum', 'subcat', 'subject', 'abstract', 'url', 'title_entities', 'abstract_entities']]\
        .to_csv(_src_dir / '../data/discuss/news.tsv', sep='\t', index=True, header=None, encoding='utf-8')
        
        
if __name__ == "__main__":
    gen_discuss_data()
