import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import time
from scipy.io.wavfile import write
from sklearn.preprocessing import LabelEncoder


def record_audio(duration=3, sr=22050):
    print("\n🎤 Recording... Speak now!")
    audio_data = sd.rec(int(duration * sr), samplerate=sr,
                        channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete.")

    # Save audio (optional)
    write("live_audio.wav", sr, (audio_data * 32767).astype(np.int16))

    return audio_data.flatten(), sr


record_audio()
