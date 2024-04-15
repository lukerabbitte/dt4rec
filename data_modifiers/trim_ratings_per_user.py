import pandas as pd

def keep_first_n_ratings(df, n_ratings=30):
    new_data = pd.DataFrame()
    for user_id, group in df.groupby('user_id'):
        ratings_to_keep = group.iloc[:n_ratings]
        new_data = pd.concat([new_data, ratings_to_keep])

    new_data.to_csv('../data/goodreads_eval_80pc_30_ratings_each.tsv', sep='\t', index=False)
    return new_data

df = pd.read_csv('../data/goodreads_eval_80pc.tsv', sep='\t')
keep_first_n_ratings(df, 30)