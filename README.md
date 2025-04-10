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

   
