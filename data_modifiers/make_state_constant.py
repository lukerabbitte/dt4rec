import pandas as pd

df = pd.read_csv("../data/goodreads_data_1024_users.tsv", sep="\t")

print(df.head())

df['user_id'] = 1

# Save the modified dataset
df.to_csv("../data/goodreads_data_1024_users_constant_state.tsv", sep="\t", index=False)