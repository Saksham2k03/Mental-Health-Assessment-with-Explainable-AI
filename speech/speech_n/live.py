import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import time
from scipy.io.wavfile import write
from sklearn.preprocessing import LabelEncoder

# Load Pre-trained Model
model = tf.keras.models.load_model("new_emotion_model.h5")

# Load Label Encoder
emotion_labels = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)


def record_audio(duration=3, sr=22050):
    print("\n🎤 Recording... Speak now!")
    audio_data = sd.rec(int(duration * sr), samplerate=sr,
                        channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete.")

    # Save audio (optional)
    write("live_audio.wav", sr, (audio_data * 32767).astype(np.int16))

    return audio_data.flatten(), sr


def extract_features_from_audio(audio_data, sr, max_pad_len=174):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)

    # Padding
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


def predict_real_time_emotion():
    # Step 1: Record Voice
    audio_data, sr = record_audio()

    # Step 2: Extract Features
    features = extract_features_from_audio(audio_data, sr)
    features = features.reshape(1, 40, 174, 1)  # Reshape for CNN

    # Step 3: Predict Emotion
    prediction = model.predict(features)
    emotion = label_encoder.inverse_transform([np.argmax(prediction)])

    print(f"🗣️ Detected Emotion: {emotion[0].upper()}")
    return emotion[0]


# Run Real-time Emotion Detection
predict_real_time_emotion()
