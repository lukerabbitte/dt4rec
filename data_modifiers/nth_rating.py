import pandas as pd

df = pd.read_csv('/home/luke/code/minGPTrecommendations/data/goodreads_eval.tsv', sep='\t')

user_rating_sum = df.groupby('user_id')['rating'].sum()

users_with_10th_highest_sum = user_rating_sum.nlargest(50).index[-1]

print(f"The users with the 10th highest total ratings are {users_with_10th_highest_sum}")