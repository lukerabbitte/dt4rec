import pandas as pd

data = pd.read_csv("/home/luke/code/minGPTrecommendations/data/goodreads_data_first_2048_users_4_groups_beta_distribution.tsv", sep="\t")
# data = pd.read_csv("/home/luke/code/minGPTrecommendations/data/goodreads_data_first_2048_users_4_groups_beta_distribution.tsv", sep='\t')


rating_counts = data['rating'].value_counts()

total_ratings = rating_counts.sum()
percentage_1 = (rating_counts.get(1, 0) / total_ratings) * 100
percentage_2 = (rating_counts.get(2, 0) / total_ratings) * 100
percentage_3 = (rating_counts.get(3, 0) / total_ratings) * 100
percentage_4 = (rating_counts.get(4, 0) / total_ratings) * 100
percentage_5 = (rating_counts.get(5, 0) / total_ratings) * 100

print("Percentage of ratings:")
print(f"1: {percentage_1:.2f}%")
print(f"2: {percentage_2:.2f}%")
print(f"3: {percentage_3:.2f}%")
print(f"4: {percentage_4:.2f}%")
print(f"5: {percentage_5:.2f}%")
