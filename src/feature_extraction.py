import numpy as np
import librosa

def feature_extraction(file_path, n_mfcc, n_chroma):

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

        #tempo in bpm
        tempo, _ = librosa.beat.beat_track(y = y, sr = sr)

        #zero crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y = y)
        zcr_mean = np.mean(zcr, axis = 0)
        zcr_var = np.var(zcr, axis = 0)

        #collecting all features in a dict { feature_name:feature }
        features = {
            'tempo': tempo,
            'zcr_mean': zcr_mean,
            'zcr_var': zcr_var,
        }

        for i, (mean, var) in enumerate(zip(mfcc_means, mfcc_vars)):
            features[f'mfcc_{i+1}_mean'] = mean
            features[f'mfcc_{i+1}_var'] = var
        for i, (mean, var) in enumerate(zip(chroma_means, chroma_vars)):
            features[f'chroma_{i+1}_mean'] = mean
            features[f'chroma_{i+1}_var'] = var

        return features
    except:
        print(f'Error extracting features from {file_path}')
        return None