import os
import pickle
import pandas as pd
import librosa
from feature_extraction import *

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        return model

def predict(model_path, file_path):
    if not os.path.exists(file_path):
        print('File does not exist')
        return 1

    y, sr = librosa.load(file_path)
    features = feature_extraction(y, sr)
    if features is None:
        print('Error extracting features')
        return 1

    model = load_model(model_path)
    features = pd.DataFrame([features])
    result = model.predict(features)
    print(f'Predicted genre: {result[0]}')

    return result