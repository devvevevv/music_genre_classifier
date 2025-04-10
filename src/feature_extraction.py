import numpy as np
import librosa, librosa.feature

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

        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cqt_means = np.mean(chroma_cqt, axis = 1)
        chroma_cqt_vars = np.var(chroma_cqt, axis = 1)

        #zero crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y = y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)

        #spectral features

        #spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        spec_cent_var = np.var(spec_cent)

        #spectral rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_roll_mean = np.mean(spec_roll)
        spec_roll_var = np.var(spec_roll)

        #spectral contrast
        S = np.abs(librosa.stft(y))
        spec_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast)
        spec_contrast_var = np.var(spec_contrast)

        #spectral flux
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        spectral_flux_mean = np.mean(onset_env)
        spectral_flux_var = np.var(onset_env)

        #spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bw_mean = np.mean(spec_bw)
        spec_bw_var = np.var(spec_bw)

        #tempo features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        beat_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

        #rms features
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)

        #collecting all features in a dict { feature_name:feature }
        features = {
            'zcr_mean': zcr_mean,
            'zcr_var': zcr_var,
            'spec_cent_mean': spec_cent_mean,
            'spec_cent_var': spec_cent_var,
            'spec_roll_mean': spec_roll_mean,
            'spec_roll_var': spec_roll_var,
            'spec_contrast_mean': spec_contrast_mean,
            'spec_contrast_var': spec_contrast_var,
            'spec_bw_mean': spec_bw_mean,
            'spec_bw_var': spec_bw_var,
            'spectral_flux_mean': spectral_flux_mean,
            'spectral_flux_var': spectral_flux_var,
            'tempo': tempo,
            'beat_strength': beat_strength,
            'rms_mean': rms_mean,
            'rms_var': rms_var,
        }

        for i, (mean, var) in enumerate(zip(mfcc_means, mfcc_vars)):
            features[f'mfcc_{i+1}_mean'] = mean
            features[f'mfcc_{i+1}_var'] = var
        for i, (mean, var) in enumerate(zip(chroma_means, chroma_vars)):
            features[f'chroma_{i+1}_mean'] = mean
            features[f'chroma_{i+1}_var'] = var
        for i, (mean, var) in enumerate(zip(chroma_cqt_means, chroma_cqt_vars)):
            features[f'chroma_{i+1}_cqt_mean'] = mean
            features[f'chroma_{i+1}_cqt_var'] = var

        return features

    except Exception as e:
        print(f'Error extracting features from {file_path}: {e}')
        return None