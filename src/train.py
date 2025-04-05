import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
from src.feature_extraction import *

def create_model(n_estimators = 100, random_state = 50):
    model = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)
    return model

#X = features, y = target (genre)
def train_model(X, y):
    model = create_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        'model': model,
        'accuracy': accuracy,
        'report': report
    }

def predict_genre(model, features):
    feature_values = []
    for key, value in features_dict.items():
        if key not in ['genre', 'file path']:
            feature_values.append(value)

            features_array = np.array(feature_values).reshape(1, -1)

            # Make prediction
            prediction = model.predict(features_array)[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(features_array)[0]
            confidence = max(probabilities)

            return prediction, confidence

def save_model(model, genres, file_path):

    model_data = {
        'model': model,
        'genres': genres
    }

    joblib.dump(model_data, file_path)

