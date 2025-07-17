import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def extract_features(audio_path, max_pad_len=174):
    y, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Padding to ensure uniform shape
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


# Assuming dataset structure: "Dataset/Emotion_Label/audio.wav"
dataset_path = r"C:\Users\saksh\Downloads\crema\AudioWAV"
labels = []
features = []

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                features.append(extract_features(file_path))
                labels.append(folder)  # Folder name as emotion label

# Convert to NumPy array
X = np.array(features)
y = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape for CNN+LSTM
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Load the trained model
model = tf.keras.models.load_model("emotion_model.h5")


def predict_emotion(audio_path):
    feature = extract_features(audio_path)
    feature = feature.reshape(1, 40, 174, 1)  # Reshape for CNN
    prediction = model.predict(feature)
    emotion = label_encoder.inverse_transform([np.argmax(prediction)])
    return emotion[0]


# Example Prediction
print("Predicted Emotion:", predict_emotion("test_audio.wav"))
