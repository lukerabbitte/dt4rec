import pandas as pd
import matplotlib.pyplot as plt
import os


def calculate_user_stats(dataset_file):
    df = pd.read_csv(dataset_file, sep='\t', header=0)
    user_mean_ratings = df.groupby('user_id')['rating'].mean()
    user_variance = df.groupby('user_id')['rating'].var()
    mean_item_rating = df['rating'].mean()

    return user_mean_ratings, user_variance, mean_item_rating


def plot_stats(user_mean_ratings, user_variance, mean_item_rating, figs_dir):
    plt.rcParams.update({'font.family': 'monospace'})
    os.makedirs(figs_dir, exist_ok=True)

    plt.figure(figsize=(20, 10))
    plt.bar(range(1, 2049), user_mean_ratings, color='#09353d')
    plt.axhline(y=mean_item_rating, color='#e37430', linestyle='--', label='Mean Item Rating for Entire Dataset',
                linewidth=5)
    plt.legend(fontsize=20)

    plt.xlabel(r'User IDs', fontweight='bold', fontsize=24)
    plt.ylabel(r'Mean Ratings', fontweight='bold', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.title(r'Mean Item Rating For Each User in Train Dataset', fontweight='bold', fontsize=28)
    plt.savefig(os.path.join(figs_dir, 'user_mean_ratings.svg'), format='svg')

    plt.close()


dataset_file = '/home/luke/code/minGPTrecommendations/data/goodreads_data_first_2048_users_4_groups_beta_distribution.tsv'
figs_dir = '../figs/visualisations'
user_mean_ratings, user_variance, mean_item_rating = calculate_user_stats(dataset_file)
plot_stats(user_mean_ratings, user_variance, mean_item_rating, figs_dir)
