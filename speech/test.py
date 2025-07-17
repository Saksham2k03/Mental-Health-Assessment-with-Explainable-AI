import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Define the emotion labels (adjust based on your dataset)
emotion_labels = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)


def extract_features(audio_path, max_pad_len=174):
    y, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Padding
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


def predict_emotion(audio_path):
    # Extract features
    feature = extract_features(audio_path)

    # Reshape for model
    feature = feature.reshape(1, 40, 174, 1)

    # Predict
    prediction = model.predict(feature)

    # Get the emotion label
    emotion = label_encoder.inverse_transform([np.argmax(prediction)])

    print(f"🎭 Predicted Emotion: {emotion[0].upper()}")
    return emotion[0]


# Test with a new audio file
predict_emotion("test_audio.wav")
