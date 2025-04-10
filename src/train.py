from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pickle
from process_data import *

#hyperparameters are tuned using GridSearchCV, and the best model is saved to a file in the specified location
def train(dataset_path, model_save_path="model.pkl"):
    print('Loading data...')
    features_df, X, y = process_data(dataset_path)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    print("Training Random Forest classifier...")
    param_grid = {
        'n_estimators': [200, 600],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=1),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.3f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    #saving model to model_save_path
    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_save_path}")

    return model, accuracy

if __name__ == "__main__":
    dataset_path = "C:/CS/Projects/music_genre_classifier/Data/genres_original"
    model, accuracy = train(dataset_path)
    print("Model saved to disk")