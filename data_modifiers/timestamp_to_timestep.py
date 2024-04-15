import pandas as pd

filename = "/home/luke/code/minGPTrecommendations/data/ml-100k/data.tsv"

df = pd.read_csv(filename, sep='\t')

df['timestamp'] = df.groupby('user_id').cumcount() + 1

df = df.rename(columns={'timestamp': 'timestep'})

df.to_csv("/home/luke/code/minGPTrecommendations/data/ml-100k/data.tsv", sep='\t', index=False)