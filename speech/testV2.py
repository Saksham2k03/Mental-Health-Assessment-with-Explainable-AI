import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Define the RAVDESS dataset directory
Ravdess = r'C:\Users\saksh\Downloads\nf\audio_speech_actors_01-24'
ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []

# Loop through each actor directory
for dir in ravdess_directory_list:
    actor_path = os.path.join(Ravdess, dir)  # Properly join paths
    if os.path.isdir(actor_path):  # Ensure it's a directory
        actor_files = os.listdir(actor_path)
        for file in actor_files:
            part = file.split('.')[0].split('-')  # Split filename
            try:
                emotion = int(part[2])  # Extract emotion label
                file_emotion.append(emotion)
                # Correct path handling
                file_path.append(os.path.join(actor_path, file))
            except (IndexError, ValueError):
                print(f"Skipping file: {file}")  # Handle unexpected filenames

# Create DataFrame for emotions and file paths
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# Map emotion numbers to labels
emotion_mapping = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
}
Ravdess_df['Emotions'] = Ravdess_df['Emotions'].map(emotion_mapping)

# Display DataFrame
print(Ravdess_df.head())
