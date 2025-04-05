import os
import numpy as np
import librosa
import librosa.feature
import pandas as pd

def feature_extraction(file_path, n_mfcc = 13, n_chroma = 12):

    try:
        #load audio as a waveform 'y' and store sampling rate in 'sr'
        y, sr = librosa.load(file_path)

        #extract features

        #MFCCs
        mfccs = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = n_mfcc)
        mfcc_means = np.mean(mfccs, axis = 1)
        mfcc_vars = np.var(mfccs, axis = 1)

        #Chroma
        chroma = librosa.feature.chroma_stft(y = y, sr = sr, n_chroma = n_chroma)
        chroma_means = np.mean(chroma, axis = 1)
        chroma_vars = np.var(chroma, axis = 1)

        #zero crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y = y)
        zcr_mean = np.mean(zcr, axis = 0)
        zcr_var = np.var(zcr, axis = 0)

        #spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        spec_cent_var = np.var(spec_cent)

        #spectral rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_roll_mean = np.mean(spec_roll)
        spec_roll_var = np.var(spec_roll)

        #collecting all features in a dict { feature_name:feature }
        features = {
            'zcr_mean': zcr_mean,
            'zcr_var': zcr_var,
            'spec_cent_mean': spec_cent_mean,
            'spec_cent_var': spec_cent_var,
            'spec_roll_mean': spec_roll_mean,
            'spec_roll_var': spec_roll_var
        }

        for i, (mean, var) in enumerate(zip(mfcc_means, mfcc_vars)):
            features[f'mfcc_{i+1}_mean'] = mean
            features[f'mfcc_{i+1}_var'] = var
        for i, (mean, var) in enumerate(zip(chroma_means, chroma_vars)):
            features[f'chroma_{i+1}_mean'] = mean
            features[f'chroma_{i+1}_var'] = var

        return features
    except Exception as e:
        print(f'Error extracting features from {file_path}: {e}')
        return None

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
    X = features_df.drop(['genre', 'file path'], axis = 1).values
    y = features_df['genre'].values

    return features_df, X, y
