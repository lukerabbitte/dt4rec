import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/home/luke/code/minGPTrecommendations/data/goodreads_data_first_2048_users_4_groups_beta_distribution.tsv", delimiter="\t")

user_rating_counts = data['user_id'].value_counts()

rating_user_counts = user_rating_counts.value_counts()

rating_user_counts = rating_user_counts.sort_index()

plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'monospace'
plt.bar(rating_user_counts.index, rating_user_counts.values)
plt.xlabel('Number of Ratings')
plt.ylabel('User Count')
plt.title('Number of Users For Each Rating Count (Goodreads Synthetic 2048 Users)')

max_value_x = max(rating_user_counts.index)
plt.xlim(0, 730)

max_value_y = max(rating_user_counts.values)
plt.ylim(0, max_value_y * 1.1)


plt.rcParams['font.family'] = 'monospace'
plt.savefig('../figs/visualisations/ratings_per_user_goodreads_data_2048_users_4_groups_beta_distribution.svg')
