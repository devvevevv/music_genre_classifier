import pandas as pd
import numpy as np
import librosa
import pickle
from feature_extraction import feature_extraction

chunk_size = 30  # seconds
overlap = 5  # seconds


def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_chunk(model, y_chunk, sr):
    try:
        features = feature_extraction(y_chunk, sr)
        if features is None:
            return None

        # Wrap dictionary in a list so pandas creates a valid 1-row DataFrame
        df = pd.DataFrame([features])

        result = model.predict(df)
        return result[0]
    except Exception as e:
        print(f'Error predicting chunk: {e}')
        return None


def get_most_common_genre(predictions):
    if not predictions:
        return None

    count_genre = {}
    for genre in predictions:
        if genre is not None:
            count_genre[genre] = count_genre.get(genre, 0) + 1

    if not count_genre:
        return None

    return max(count_genre, key=count_genre.get)


def analyze_audio_file(model_path, file_path):
    try:
        model = load_model(model_path)
        # Load audio file ONCE into memory to prevent psf_fseek errors on MP3s
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f'Error loading audio file: {e}')
        return None

    total_samples = len(y)
    chunk_samples = chunk_size * sr
    step_samples = (chunk_size - overlap) * sr

    # Evaluate the full signal if the track is shorter than or equal to chunk_size
    if total_samples <= chunk_samples + sr:
        return predict_chunk(model, y, sr)

    predictions = []

    # Slice the loaded NumPy array directly in memory
    for start_sample in range(0, total_samples, step_samples):
        end_sample = start_sample + chunk_samples
        chunk = y[start_sample:end_sample]

        # Skip chunks that are too short for chroma feature extraction
        if len(chunk) < 2048:
            continue

        chunk_result = predict_chunk(model, chunk, sr)
        if chunk_result is not None:
            predictions.append(chunk_result)

    if not predictions:
        print("No valid predictions were made.")
        return None

    return get_most_common_genre(predictions)