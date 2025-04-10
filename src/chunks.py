import pandas as pd
import numpy as np
import librosa
from feature_extraction import *
from predict import *

chunk_size = 30
overlap = 5

def get_chunk_intervals(file_path):
    try:
        y, sr = librosa.load(file_path)
        total_duration = librosa.get_duration(y =y, sr =sr)
        step = chunk_size - overlap

        chunk_times = []
        for i in np.arange(0, total_duration, step):
            chunk_times.append( (i, i + chunk_size) )

        return chunk_times

    except Exception as e:
        print(f'Error loading file: {e}')
        return None

def get_chunks(file_path, start_time):
    try:
        y, sr = librosa.load(file_path, offset = start_time, duration = 30)

        '''
        in feature_extraction.py, chroma.stft() expects chunk of at least 2048 samples.
        so we need to make sure chunk is of size >= 2048
        '''
        if y is None or len(y)<2048:
            return None, None

        return y, sr
    except Exception as e:
        print(f'Error loading file: {e}')
        return None, None

def predict_chunk(model_path, y, sr):
    try:
        features = feature_extraction(y, sr)
        if features is None:
            print("Feature extraction returned None")
            return None
        print(f"Extracted {len(features)} features")
        features = pd.DataFrame(features)
        model = load_model(model_path)
        result = model.predict(features)
        print(f"Chunk prediction: {result[0]}")
        return result[0]

    except Exception as e:
        print(f'Error predicting chunks: {e}')
        return 1

def get_most_common_genre(predictions):
    count_genre = {}
    most_common = None
    max_count = 0

    for genre in predictions:
        if genre in count_genre:
            count_genre[genre] +=1
        else:
            count_genre[genre] = 1

    for (genre, count) in count_genre.items():
        if count > max_count:
            max_count = count
            most_common = genre

    return most_common

def analyze_audio_file(model_path, file_path):
    y, sr = librosa.load(file_path)
    total_duration = librosa.get_duration(y=y, sr=sr)

    #if file is itself of duration chunk_size, no need to divide it into chunks
    if total_duration <= chunk_size:
        return predict(model_path, file_path)

    chunk_times = get_chunk_intervals(file_path)

    if chunk_times is None:
        print('Error getting chunk intervals')
        return None

    predictions = []
    for (i, start) in chunk_times:
        y, sr = get_chunks(file_path, start)
        if y is None:
            continue
        chunk_result = predict_chunk(model_path, y, sr)
        predictions.append(chunk_result)

    if not predictions:
        print("No valid predictions were made.")
        return None

    return get_most_common_genre(predictions)
