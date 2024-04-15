import pandas as pd

df = pd.read_csv("../data/goodreads_eval_modified.tsv", sep='\t')

count_5 = df[df['rating'] == 5]['rating'].count()
count_1 = df[df['rating'] == 1]['rating'].count()

total_ratings = len(df)
percentage_5 = (count_5 / total_ratings) * 100
percentage_1 = (count_1 / total_ratings) * 100

# Print the percentages
print("Percentage of '5' ratings:", percentage_5)
print("Percentage of '1' ratings:", percentage_1)
