import pandas as pd

df = pd.read_csv('/home/luke/code/minGPTrecommendations/data/ml-1m/ratings_timestep.tsv', delimiter='\t')


five_star_ratings = df[df['rating'] == 5]
user_ratings_counts = five_star_ratings.groupby('user_id').size()
average_5_star_ratings = user_ratings_counts.mean()


min_5_star_ratings = user_ratings_counts.min()
max_5_star_ratings = user_ratings_counts.max()

print("Average number of 5-star ratings per user:", average_5_star_ratings)
print("Minimum number of 5-star ratings given by any user:", min_5_star_ratings)
print("Maximum number of 5-star ratings given by any user:", max_5_star_ratings)
