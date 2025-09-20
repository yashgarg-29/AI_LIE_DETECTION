from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, os, numpy as np, librosa, parselmouth
import random

app = Flask(__name__)
CORS(app)

# Load pretrained voice model
model = joblib.load("lie_detector_voice.pkl")

LABELS = {0: "Truth ✅", 1: "Lie ❌", 2: "Uncertain ⚠️"}

# ---------- Feature Extraction ----------
def extract_jitter_shimmer(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return jitter, shimmer
    except:
        return 0.0, 0.0

def extract_features(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch_vals) if pitch_vals.size > 0 else 0.0
    pitch_std = np.std(pitch_vals) if pitch_vals.size > 0 else 0.0

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    jitter, shimmer = extract_jitter_shimmer(file_path)

    feats = list(mfcc_mean) + list(mfcc_std) + [
        pitch_mean, pitch_std,
        np.mean(zcr), np.std(zcr),
        np.mean(rms), np.std(rms),
        jitter, shimmer
    ]
    return np.array(feats).reshape(1, -1)

# ---------- API Routes ----------

@app.route("/")
def dashboard():
    return render_template("dashboard_html.html")

@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    if "file" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    file = request.files["file"]
    tmp_path = "temp.wav"
    file.save(tmp_path)

    try:
        feats = extract_features(tmp_path)
        proba = model.predict_proba(feats)[0]  # probability distribution
        pred = np.argmax(proba)
        label = LABELS.get(int(pred), str(pred))
        confidence = float(proba[pred])

        return jsonify({
            "type": "voice",
            "prediction": label,
            "confidence": confidence
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route("/predict_face", methods=["POST"])
def predict_face():
    # Simulated results for now
    pred = random.choice([0, 1, 2])  # Truth, Lie, Uncertain
    label = LABELS[pred]
    confidence = round(random.uniform(0.6, 0.95), 2)

    return jsonify({
        "type": "face",
        "prediction": label,
        "confidence": confidence
    })



if __name__ == "__main__":
    app.run(debug=True)
