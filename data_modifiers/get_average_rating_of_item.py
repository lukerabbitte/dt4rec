import pandas as pd
import argparse

def get_average_rating(item_id):
    df = pd.read_csv('/home/luke/code/minGPTrecommender/data/goodreads_eval.tsv',  sep='\t')

    average_ratings = df.groupby('item_id')['rating'].mean()

    average_rating = average_ratings.get(item_id, None)

    return average_rating

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('item_id', type=int, help='Item ID for which to get the average rating')
    args = parser.parse_args()

    print(get_average_rating(args.item_id))

if __name__ == '__main__':
    main()