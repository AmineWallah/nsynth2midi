from extract_features import extract_features
import joblib
import numpy as np
import librosa
import pretty_midi

FRAME_SIZE = 8192
HOP_SIZE = 2048
SAMPLE_RATE = 16000



def predict_melody(filepath, model, scaler, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)

    predictions = []
    start = 0
    while start + frame_size <= len(y):
        # Sliding window logic to predict melody
        window = y[start : start + frame_size]
        window_scaled_features = scaler.transform(extract_features(y=window, sr=sr).reshape(1, -1))
        prediction = model.predict(window_scaled_features)[0]

        # Fine tunes the prediction bassed on the fundamental frequencies extracted from the window
        f0, voiced_flag, voiced_probs = librosa.pyin(
            window,
            fmin=librosa.note_to_hz('C1'),
            fmax=librosa.note_to_hz('C8'),
            sr=sr
        )

        f0_confident = f0[voiced_probs > 0.5]
        if len(f0_confident) > 0:
            f0_mean = np.mean(f0_confident)
        else:
            f0_mean = 0.0

        ## Octave shifting based on the difference between the prediction and the fundamental frequency
        if f0_mean > 0.0:
            pyin_midi = librosa.hz_to_midi(f0_mean)
            octave_shift = int(np.round((prediction - pyin_midi) / 12))
            print(f"prediction={prediction}, f0_mean={f0_mean:.1f}, pyin_midi={pyin_midi:.1f}, shift={octave_shift}")
            prediction -= octave_shift * 12

        predictions.append(prediction)
        start += hop_size

    return predictions

def count_runs(predictions):
    runs = []
    current_pitch = predictions[0]
    current_length = 1
    current_start = 0

    for i, pitch in enumerate(predictions[1:], start=1):
        if pitch == current_pitch:
            current_length += 1
        else:
            runs.append({"pitch": current_pitch, "length": current_length, "start": current_start})
            current_pitch = pitch
            current_length = 1
            current_start = i

    runs.append({"pitch": current_pitch, "length": current_length, "start": current_start})
    return runs

def fixed_predictions(predictions, runs, min_length=1):
    for i in range(len(runs) - 1):
        pitch = runs[i]["pitch"]
        length = runs[i]["length"]
        start = runs[i]["start"]
        if length <= min_length:
            predictions[start:start + length] = [runs[i+1]['pitch']] * length

    if runs[-1]["length"] <= min_length:
        length = runs[-1]["length"]
        start = runs[-1]["start"]

        predictions[start:start + length] = [runs[-2]["pitch"]] * length

    return predictions

def create_note_events(runs):
    note_events = []
    for run in runs:
        start_time = (run["start"] * HOP_SIZE / SAMPLE_RATE)
        duration = (run["length"] * HOP_SIZE / SAMPLE_RATE)
        note_events.append(
            {
                'pitch': run["pitch"],
                'start_time': start_time,
                'duration': duration,
            }
        )

    return note_events

def build_midi(note_events):
    # MIDI object initialization
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for event in note_events:
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(event['pitch']),
            start=event['start_time'],
            end=event['start_time'] + event['duration']
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write('output.mid')

def main():
    scaler = joblib.load('scaler.pkl')
    clf = joblib.load('model.pkl')
    filepath = 'vocals.wav'

    # Pipeline would be:
    # 1. Predict melody
    # 2. Group into runs
    # 3. Fix predictions based on the runs
    # 4. Group the fixed predictions into runs again
    # 5. Use second grouping to know when the melody starts and ends
    # 6. Build MIDI notes
    # 7. Export MIDI file

    predictions = predict_melody(filepath, clf, scaler)
    runs = count_runs(predictions)
    predictions = fixed_predictions(predictions, runs, min_length=1)
    second_runs = count_runs(predictions)
    note_events = create_note_events(second_runs)
    build_midi(note_events)
    print("Done!")

if __name__ == '__main__':
    main()