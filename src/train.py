import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from feature_extraction import process_data

def train(dataset_path, model_save_path="model.pkl"):
    features_df, X, y = process_data(dataset_path)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_save_path}")

    return model, accuracy

