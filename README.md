# ARCHIVED DUE TO MAESTRO2MIDI BEING A THING NOW
# NSynth2MIDI Monophonic Transcriber
Nsynth2MIDI is a small machine learning project (that i made after around 2 weeks of learning stuff idk) supposed to transcribe .wav files into MIDI format, it uses the NSynth dataset's (credit below) audio library of labeled audios files of single notes to predict a .wav file's succession of notes relying mainly on a random forest classifier and a sliding window algorithm.

## How It Works
The pipeline NSynth2MIDI relies on is the following:
1. Loading of the dataset's audio files
2. Feature extraction using `librosa`
3. Training a random forest classifier on the dataset's extracted features and labels using `scikit-learn`
4. Predicting a series of notes using the trained classifier and a sliding window algorithm
5. Octave correction on the predicted notes using the fundamental frequencies extracted by `librosa.pyin()`
6. MIDI export using `pretty_midi`

## Key Decisions and Challenges Faced
- The reason why a random forest classifier was used is because the initial goal of the model was to predict one single pitch from one .wav file, which would then be classified amongst ~112 different pitches according to the extracted features from the dataset's audio (MFCCs, chroma, ZCR, spectral centroid).
After evaluating the model's performance -check the model evaluation section- I found that it would often confuse the same note across different octaves (e.g. C3 vs C4 vs C5). This happens because chroma features, one of the main inputs to the model, intentionally collapse all octaves of the same pitch class into a single value, meaning the model has no reliable way to tell octaves apart from chroma alone. To fix this, I used `librosa.pyin()` to extract the fundamental frequency and correct the predicted octave after classification. Due to the high computational cost of extracting pyin features across the entire training set, I opted to use it only as a post-prediction correction step rather than retraining the model with it as an input feature.
- To clean up the awkward transitions between different pitches before exporting the processed MIDI, I added run-length filtering to clean up noisy predictions.

## Model Evaluation
![Confusion Matrix](confusion.png)

| Metric | Value |
|---|---|
| Accuracy | 0.9210 |
| F1 (macro avg) | 0.8513 |
| F1 (weighted avg) | 0.9204 |
| Recall (macro avg) | 0.8233 |
| Precision (macro avg) | 0.9220 |

- Weighted average: weighs each class's contribution by how many samples it had.

## Setup & Usage

1. Clone the repo and install dependencies:
```bash
   git clone https://github.com/your-username/nsynth2midi.git
   cd nsynth2midi
   uv sync
```

2. Download the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) (validation split recommended for faster iteration) and place it in the project root as `nsynth-valid/`.

3. Extract features and train the model:
```bash
   python extract_features.py
   python train_model.py
```

4. Transcribe an audio file:
```bash
   python test_model.py
```
   (Edit the `filepath` variable in `test_model.py` to point to your `.wav` file.)


## Limitations & Future Work
- For now it only works on single notes (monophonic)
- The model cannot process silence in between notes for now
- Timings are not super accurate
- This model's performance is not great compared to most audio to midi converters I've came across, and this is largely due to the fact that the NSynth dataset wasn't the most optimal for this purpose. A similar project with an appropriate dataset will probably be made in due time


## Credits
- [NSynth Dataset](https://github.com/google/nsynth)
