# Music Genre Classifier
A machine learning-based system that predicts the genre of music from audio files using feature extraction and Random Forest classification.

## Overview
This project implements an audio analysis system that extracts audio features from music files and uses a trained machine learning model to classify the genre. The system can handle both short audio clips and longer tracks by dividing them into overlapping chunks for more accurate genre prediction.

## Features

- Automated genre classification of audio files
- Support for processing both short clips and full-length tracks
- Audio chunking with overlap for better prediction accuracy
- Extensive audio feature extraction including:
  - MFCCs (Mel-Frequency Cepstral Coefficients)
  - Chromagrams (STFT and CQT-based)
  - Zero Crossing Rate
  - Spectral features (centroid, rolloff, contrast, flux, bandwidth)
  - Tempo and beat features
  - RMS energy

## Dataset
This project uses the [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset, which is widely used for music genre classification tasks. The dataset consists of 1000 audio tracks, each 30 seconds long, with 100 tracks per genre for 10 genres:
 - Blues
 - Classical
 - Country
 - Disco
 - Hip-hop
 - Jazz
 - Metal
 - Pop
 - Reggae
 - Rock

## Requirements
- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - librosa
  - scikit-learn
  - pickle
 
## Installation
`
git clone https://github.com/devvevevv/music_genre_classifier.git
cd music-genre-classifier
`
### Install required packages
`
pip install pandas numpy librosa scikit-learn
`
## Usage
### Training the model
Edit the dataset_path in train.py to point to your dataset.

`
python train.py
`
### Predicting Genre of an Audio File
To predict the genre of an audio file:

`
python main.py
`

When prompted, enter the path to your audio file. The program will analyze the file and output the predicted genre.

## How It Works

1. ### Feature Extraction:
   The system extracts features from audio files using the librosa library.
3. ### Audio Chunking:
   For longer tracks, the audio is divided into 30-second chunks with 5-second overlap to capture different sections of the song.
5. ### Classification:
   A trained Random Forest classifier predicts the genre based on the extracted features.
7. ### Consensus Prediction:
   For chunked audio, the final genre is determined by taking the most common prediction across all chunks.

## Project Structure

1. `main.py`: Entry point for the application
2. `feature_extraction.py`: Functions for extracting features from audio
3. `chunks.py`: Handles dividing longer audio files into overlapping chunks (if audio is of length>=30s, predicts directly)
4. `process_data.py`: Processes the dataset for training
5. `train.py`: Trains the Random Forest classifier with hyperparameter tuning

## Model Training
The classifier is a Random Forest model trained with GridSearchCV for hyperparameter optimization. The training process includes:
1. Data splitting (train/test):
   ![image](https://github.com/user-attachments/assets/1224dc73-0194-4e7f-b783-26f180dfbbe4)

2. Hyperparameter tuning (number of estimators, max depth, min samples split/leaf):
   ![image](https://github.com/user-attachments/assets/ce181aaf-8575-47a4-81a5-2441f052a512)

3. Class balancing:
   ![image](https://github.com/user-attachments/assets/8d90f115-67fd-4c21-ad4d-db08ac835ed8)
   ![image](https://github.com/user-attachments/assets/609d6bfc-4858-4497-9666-fa9a3f42ffc0)

4. Model evaluation with accuracy metrics and classification report:
   ![image](https://github.com/user-attachments/assets/0e7b4536-9264-424b-88d0-36b702e7b60c)
   ![image](https://github.com/user-attachments/assets/6f48e48e-3b98-4e88-bb56-b18d6ec69609)
   ![image](https://github.com/user-attachments/assets/d5f4619c-3b25-461c-a62b-276c9ff779d5)

## Feeding songs to model
![image](https://github.com/user-attachments/assets/811318d9-0d06-4918-9dd7-b6ace9bab5cf)
![image](https://github.com/user-attachments/assets/9256acad-6778-4b28-a079-a23daae6228b)
![image](https://github.com/user-attachments/assets/3420b977-eb44-4769-9362-3f588d43f277)
![image](https://github.com/user-attachments/assets/d7a07898-40d0-49e7-93ad-f4624655e0a1)

## Future Improvements 
1. Implement deep learning approaches (CNNs, RNNs)
2. Create a web interface for easier usage
3. Optimize feature extraction for better performance

## Industry Applications 
1. Create a plugin for digital audio workstations to suggest genre-specific production techniques
2. Build a similarity engine for copyright analysis and plagiarism detection
3. Incorporate lyric analysis using NLP techniques for genres with distinctive lyrical patterns
