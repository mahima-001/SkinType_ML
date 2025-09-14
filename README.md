# SkiScan - Skin Type Prediction Web Page

SkiScan is a full-stack machine learning web application that predicts your skin type and provides personalized skincare recommendations. It features a modern frontend, a Python Flask backend, and a trained machine learning model.

---

## 🚀 Features
- **Interactive Web UI**: Modern, responsive design with multi-page navigation
- **Skin Analysis Quiz**: Collects user data for prediction
- **ML Model**: RandomForestClassifier trained on real skin data
- **Personalized Recommendations**: Tailored advice for each skin type
- **End-to-End Integration**: Frontend, backend, and ML pipeline all connected

---

## 🗂️ Project Structure

```
SkinType/
│
├── app.py                  # Flask backend (real ML model)
├── app_frontend_test.py    # Flask backend (mock/test mode)
├── skin.py                 # All-in-one ML pipeline (train/test/save)
├── Data.csv                # Skin type dataset
├── requirements.txt        # Python dependencies
├── run_skin_app.bat        # Batch file to start the app
│
├── statics/
│   ├── style.css           # Frontend CSS
│   └── script.js           # Frontend JS
│
├── templates/
│   └── index.html          # Main HTML template
│
├── skin_type_model.pkl     # Trained ML model
├── scaler.pkl              # Feature scaler
├── skin_type_encoder.pkl   # Label encoders
├── categorical_encoders.pkl# Categorical encoders
└── ...
```

---

## ⚡ Quick Start

### 1. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Train the Model (Optional)**
If you want to retrain the model:
```bash
python skin_ml.py
```

### 3. **Run the App**
- **Easiest:** Double-click `run_skin_app.bat`
- **Or:**
```bash
python app.py
```

### 4. **Open in Browser**
Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 How It Works
- **Frontend**: User fills out the skin analysis quiz
- **Backend**: Flask receives data, preprocesses, and predicts using the ML model
- **ML Model**: RandomForestClassifier trained on `Data.csv`
- **Results**: Skin type, confidence, and recommendations are shown to the user

---

## 📝 File Descriptions
- `app.py`: Main Flask backend, connects frontend to ML model
- `skin_ml.py`: All-in-one script for training/testing/saving the ML model
- `Data.csv`: Dataset for training/testing
- `statics/`: Contains all static frontend files (CSS, JS)
- `templates/`: Contains HTML templates
- `skin_type_model.pkl`, `scaler.pkl`, `skin_type_encoder.pkl`, `categorical_encoders.pkl`: Saved model and preprocessors

---

## 🛠️ Troubleshooting
- If you see errors about missing `.pkl` files, run `python skin_ml.py` to retrain and save the model
- If the app doesn't start, check that all dependencies are installed and the correct Python environment is activated
- For Windows PowerShell script issues, use the batch file or run Python commands directly

---

## 📄 License
This project is for educational and demonstration purposes.

---

- Built by Mahima Tripathi

