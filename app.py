from fastapi import FastAPI, WebSocket
import numpy as np
import joblib
import librosa
import noisereduce as nr
import tensorflow as tf

app = FastAPI()

# Load TensorFlow Lite model, scaler, and label encoder
interpreter = tf.lite.Interpreter(model_path="CNN_best_model_last1.2.keras")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load('scaler1.joblib')
label_encoder = joblib.load('label_encoder1.joblib')

def extract_features(audio_chunk, sample_rate=44100, n_mfcc=40):
    audio_chunk = nr.reduce_noise(y=audio_chunk, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return scaler.transform([mfccs_mean])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            features = extract_features(audio_chunk).reshape(1, -1, 1)
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            await websocket.send_json({"class": predicted_label, "confidence": confidence})
        except Exception as e:
            await websocket.send_json({"error": str(e)})

