import pandas as pd
import numpy as np
import random
from scipy.stats import beta
import os


def create_synthetic_review_dataset(mu_file="../data_generators/mu_goodreads4.csv", sigma_file="../data_generators/sigma_goodreads4.csv",
                                    users_per_group=4096, min_ratings_per_user=30, max_ratings_per_user=30):
    """
    Create a synthetic review dataset based on mean and standard deviation files.

    Parameters:
    - mu_file (str): File path for the mean values CSV file.
    - sigma_file (str): File path for the standard deviation values CSV file.
    - users_per_group (int): Number of users to generate within each group.
    - min_ratings_per_user (int): Minimum number of ratings per user. Follows ml100k baseline to avoid cold start.
    - max_ratings_per_user (int): Maximum number of ratings per user.

    Notes:
    - In ml100k dataset, ratio of users to items is 943/1682. To keep this ratio we have 153 users.
    - In ml100k dataset, least active user had rated 20/1682 items (~1%)
    - In ml100k dataset, most active user had rated 737/1682 items (~44%)
    - In ml100k dataset, average user had rated 106/1682 items (~7%)
    - We try and maintain similar ratios with our data

    Returns:
    - np arrays of states, actions, returns, terminal indices, returns-to-go, timesteps
    """

    # The value at [group_index][item_index] gives us the mean and variance in their respective datasets.
    def get_rating(group_index, item_index):
        mean = -means.iloc[group_index, item_index]
        variance = variances.iloc[group_index, item_index]
        std_dev = np.sqrt(variance)
        rating = np.random.normal(mean, std_dev)
        rounded_rating = round(rating)
        clipped_rating = np.clip(rounded_rating, 1, 5)
        return round(clipped_rating)

    def generate_num_items_to_rate(a=1, b=12):
        return int(np.floor(beta.rvs(a, b, size=1) * (max_ratings_per_user - min_ratings_per_user + 1) + min_ratings_per_user))

    # Set up dataframes and note shape
    means = pd.read_csv(mu_file, header=None)
    variances = pd.read_csv(sigma_file, header=None)
    num_items = means.shape[1]
    groups = means.shape[0]

    user_id = 1
    data = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestep'])
    terminal_index = 0
    terminal_indices = []

    for group in range(groups):
        for _ in range(users_per_group):
            num_items_to_rate = generate_num_items_to_rate()
            item_ids = random.sample(list(range(num_items)), num_items_to_rate)

            for timestep, item_id in enumerate(item_ids, start=1):
                new_row = {'user_id': user_id, 'item_id': item_id + 1,
                           'rating': get_rating(group, item_id), 'timestep': timestep}
                data.loc[len(data)] = new_row

            terminal_index += timestep
            terminal_indices.append(terminal_index)
            user_id += 1

    base_filename = '../data/goodreads_data_4096_users_4_groups.tsv'
    if os.path.exists(base_filename):
        suffix = 1
        while os.path.exists(f"{base_filename}_{suffix}.tsv"):
            suffix += 1
        filename_with_suffix = f"{base_filename}_{suffix}.tsv"
    else:
        filename_with_suffix = base_filename

    ratings_per_user = data.groupby('user_id').size()
    min_ratings_per_user = ratings_per_user.min()
    print(f"min ratings per user: {min_ratings_per_user}")

    # Save the DataFrame to the file with the chosen filename
    data.to_csv(filename_with_suffix, sep='\t', index=False)

    states = np.array(data.iloc[:, 0])
    actions = np.array(data.iloc[:, 1])
    returns = np.array(data.iloc[:, 2])

    start_index = 0
    returns_to_go = np.zeros_like(returns)

    # Generate returns-to-go
    for i in terminal_indices:
        curr_traj_returns = returns[start_index:i]
        for j in range(i - 1, start_index - 1, -1):
            returns_to_go_j = curr_traj_returns[j - start_index:i - start_index]
            returns_to_go[j] = sum(returns_to_go_j)
        start_index = i

    terminal_indices = np.array(terminal_indices)
    returns_to_go = np.array(returns_to_go)
    timesteps = np.array(data.iloc[:, 3])

    return states, actions, returns, terminal_indices, returns_to_go, timesteps


if __name__ == "__main__":
    s, a, r, done_idxs, rtg, t = create_synthetic_review_dataset()
    print(s)
    print(a)
    print(r)
    print(done_idxs)
    print(rtg)
    print(t)