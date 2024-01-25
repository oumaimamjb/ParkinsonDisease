from flask import Flask, render_template, request, redirect, url_for , jsonify
import os
import librosa
import numpy as np
import soundfile as sf
import joblib
from scipy.signal import medfilt
import sounddevice as sd
import io



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record')
def record_audio():
    return render_template('record.html')

@app.route('/upload')
def upload_audio():
    return render_template('upload.html')

@app.route('/result', methods=['POST'])
def show_result():
    if request.method == 'POST':
        if 'audio_data' in request.files:
            audio_file = request.files['audio_data']
            filename = 'temp_audio.wav'
            audio_file.save(os.path.join('static', filename))
            features_audio = extract_audio_features_from_file(os.path.join('static', filename))
            prediction = predict_parkinson(features_audio)
            os.remove(os.path.join('static', filename))
            return render_template('prediction_result.html', prediction=prediction)
        elif 'file_data' in request.files:
            file = request.files['file_data']
            filename = 'temp_file.wav'
            file.save(os.path.join('static', filename))
            features_audio = extract_audio_features_from_file(os.path.join('static', filename))
            prediction = predict_parkinson(features_audio)
            os.remove(os.path.join('static', filename))
            return render_template('prediction_result.html', prediction=prediction)
    return redirect(url_for('index'))

@app.route('/predict_microphone', methods=['POST'])
def predict_microphone():
    if request.method == 'POST':
        CHUNK = 1024
        FORMAT = 'int16'
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = 5

        print("Enregistrement audio...")

        # Enregistrement audio avec sounddevice
        audio_data = sd.rec(int(RATE * RECORD_SECONDS), samplerate=RATE, channels=CHANNELS, dtype=FORMAT)
        sd.wait()

        print("Enregistrement terminé.")

        # Conversion des données audio en fichier wav
        bytes_io = io.BytesIO()
        with wave.open(bytes_io, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # Ajout de cette ligne pour spécifier la largeur d'échantillon
            wf.setframerate(RATE)
            wf.writeframes(audio_data.tobytes())

        # Utilisation du fichier enregistré pour la prédiction
        features_audio = extract_audio_features_from_bytesio(bytes_io)
        prediction = predict_parkinson(features_audio)

        return jsonify({'prediction': prediction})

    return jsonify({'error': 'No audio data received'})

import wave

def get_audio_properties(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        # Récupérer les propriétés du fichier audio
        channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
        frame_rate = audio_file.getframerate()
        num_frames = audio_file.getnframes()
        compression_type = audio_file.getcomptype()
        comp_name = audio_file.getcompname()

        print(f"Channels: {channels}")
        print(f"Sample Width: {sample_width}")
        print(f"Frame Rate: {frame_rate}")
        print(f"Number of Frames: {num_frames}")
        print(f"Compression Type: {compression_type}")
        print(f"Compression Name: {comp_name}")

# Chemin vers le fichier audio
file_path = r'C:\Users\info\Downloads\newnew\static\temp_audio_microphone.wav'

# Obtention des propriétés du fichier audio
get_audio_properties(file_path)



def extract_audio_features_from_file(file_path):
    audio, sr = librosa.load(file_path)
    audio_normalized = librosa.util.normalize(audio)
    audio_filtre = medfilt(audio_normalized, kernel_size=3)

    mfcc = librosa.feature.mfcc(y=audio_filtre, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_filtre, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=audio_filtre, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio_filtre)

    mfcc_mean = np.mean(mfcc, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    centroid_mean = np.mean(centroid)
    zcr_mean = np.mean(zcr)

    contrast = librosa.feature.spectral_contrast(y=audio_filtre, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    rolloff = librosa.feature.spectral_rolloff(y=audio_filtre, sr=sr)
    rolloff_mean = np.mean(rolloff)
    
    tonnetz = librosa.feature.tonnetz(y=audio_filtre, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitch)
    
    jitter = librosa.effects.harmonic(audio)
    jitter_mean = np.mean(jitter)

    features = list(mfcc_mean) + list(chroma_mean) + [centroid_mean, zcr_mean, rolloff_mean] + list(tonnetz_mean) + list(contrast_mean) + [pitch_mean, jitter_mean]
    
    return np.array(features).reshape(1, -1)

def extract_audio_features_from_bytesio(bytes_io):
    bytes_io.seek(0)
    audio, sr = sf.read(bytes_io)
    audio_normalized = librosa.util.normalize(audio)
    audio_filtre = medfilt(audio_normalized, kernel_size=3)

    mfcc = librosa.feature.mfcc(y=audio_filtre, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_filtre, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=audio_filtre, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio_filtre)

    mfcc_mean = np.mean(mfcc, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    centroid_mean = np.mean(centroid)
    zcr_mean = np.mean(zcr)

    contrast = librosa.feature.spectral_contrast(y=audio_filtre, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    rolloff = librosa.feature.spectral_rolloff(y=audio_filtre, sr=sr)
    rolloff_mean = np.mean(rolloff)
    
    tonnetz = librosa.feature.tonnetz(y=audio_filtre, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitch)
    
    jitter = librosa.effects.harmonic(audio)
    jitter_mean = np.mean(jitter)

    features = list(mfcc_mean) + list(chroma_mean) + [centroid_mean, zcr_mean, rolloff_mean] + list(tonnetz_mean) + list(contrast_mean) + [pitch_mean, jitter_mean]

    return np.array(features).reshape(1, -1)



def predict_parkinson(features):
    model = joblib.load('models/lrrr0_model.pkl')
    prediction = model.predict(features)
    # Convertir le résultat de la prédiction en un entier simple
    prediction = int(prediction[0])
    return prediction


if __name__ == '__main__':
    app.run(debug=True)