import json
import os
import librosa
import numpy as np
from tqdm import tqdm
import collections
from joblib import Parallel, delayed

DATA_DIR = '/mnt/storage/Downloads/audio2midi-python/nsynth-train/'
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
JSON_PATH = os.path.join(DATA_DIR, 'examples.json')

def extract_features(filepath=None, y=None, sr=None):
    if filepath is not None:
        y, sr = librosa.load(filepath, sr=None)
    elif y is None or sr is None:
        raise ValueError("Must provide either filepath, or both y and sr")

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # # Yin bullshit
    # f0, voiced_flag, voiced_probs = librosa.pyin(
    #     y,
    #     fmin=librosa.note_to_hz('C1'),
    #     fmax=librosa.note_to_hz('C8'),
    #     sr=sr
    # )
    #
    # f0_confident = f0[voiced_probs > 0.5]
    #
    # if len(f0_confident) > 0:
    #     f0_mean = np.mean(f0_confident)
    #     f0_std = np.std(f0_confident)
    # else:
    #     f0_mean = 0.0
    #     f0_std = 0.0
    #
    # if np.isnan(f0_mean):
    #     f0_mean = 0.0
    # if np.isnan(f0_std):
    #     f0_std = 0.0

    # Combine everything into one flat feature vector
    feature_vector = np.concatenate([
        mfccs_mean, mfccs_std,
        chroma_mean, chroma_std,
        [centroid_mean, centroid_std],
        [zcr_mean, zcr_std],
        # [f0_mean, f0_std],
    ])

    return feature_vector

def main():
    # Loads labels as dictionary from json file
    with open(JSON_PATH, 'r') as f:
        labels_dict = json.load(f)

    # Creates list of tuples (filepath, label/pitch)
    file_label_pairs = []

    for filename in os.listdir(AUDIO_DIR):
        # Skips non wav files
        if not filename.endswith('.wav'):
            continue

        # Removes .wav file extension
        note_str = filename.split('.')[0]

        if note_str not in labels_dict:
            print(f'WARNING: Note {note_str} not found in labels.json, skipping...')
            continue

        pitch = labels_dict[note_str]['pitch']
        filepath = os.path.join(AUDIO_DIR, filename)

        file_label_pairs.append((filepath, pitch))

    X_list = []
    y_list = []

    # TQDM used to visualize the audio processing progress
    X_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(extract_features)(filepath=filepath) for filepath, pitch in file_label_pairs
    )
    y_list = [pitch for filepath, pitch in file_label_pairs]

    X = np.array(X_list)
    y = np.array(y_list)

    # Find which pitches have at least 2 samples
    counts = collections.Counter(y)
    valid_pitches = {pitch for pitch, count in counts.items() if count >= 2}

    # We use a mask to only preserve the valid pitches
    mask = np.array([label in valid_pitches for label in y])
    X_filtered = X[mask]
    y_filtered = y[mask]

    np.save('X.npy', X_filtered)
    np.save('y.npy', y_filtered)

if __name__ == '__main__':
    main()