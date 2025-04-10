import os
import pandas as pd
from feature_extraction import *

def process_data(data_path):
    features_list = []
    genres = os.listdir(data_path)

    for genre in genres:
        genre_path = os.path.join(data_path, genre)

        print(f'Processing genre: {genre}')
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            features = feature_extraction(file_path)
            if features is not None:
                features['genre'] = genre
                features['file path'] = file_path
                features_list.append(features)

    features_df = pd.DataFrame(features_list)
    X = features_df.drop(['genre', 'file path'], axis = 1)
    y = features_df['genre']

    return features_df, X, y
