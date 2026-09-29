import os
import librosa
import pandas as pd
from feature_extraction import *

def process_data(data_path):
    features_list = []
    genres = os.listdir(data_path)

    for genre in genres:
        genre_path = os.path.join(data_path, genre)

        # skip .DS_Store or any non dir files
        if genre.startswith('.') or not os.path.isdir(genre_path):
            continue

        print(f'Processing genre: {genre}')

        for file_name in os.listdir(genre_path):
            # skip hidden files within genre dir
            if file_name.startswith('.'):
                continue

            try:
                file_path = os.path.join(genre_path, file_name)
                y, sr = librosa.load(file_path)
                features = feature_extraction(y, sr)
                if features is not None:
                    features['genre'] = genre
                    features['file path'] = file_path
                    features_list.append(features)
            except Exception as e:
                print(f"Skipping corrupt or unreadable file: {file_path}")
                continue

    features_df = pd.DataFrame(features_list)
    X = features_df.drop(['genre', 'file path'], axis = 1)
    y = features_df['genre']

    return features_df, X, y
