# winequality
# Wine Quality Predictor

Predict the quality of red wines (Good/Bad) based on physicochemical properties using a machine learning model trained on wine datasets.

##  Project Overview

This application allows users to:
- Select a **popular wine** or
- Enter **custom chemical properties** of a red wine

The backend predicts wine quality and provides a reasoning report based on the input.

---

##  Project Structure

```
.
├── app.py                  # Flask backend API
├── index.html              # Frontend user interface
├── decision_tree.py        # Optional: rule-based model (for explainability)
├── feature_engineering.py  # Feature transformations
├── model_training.py       # Model training logic
├── hyperparameter_tuning.py# Tuning logic for best performance
├── model_performance.py    # Evaluation metrics and visualization
├── README.md               # Project overview and instructions
```

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/wine-quality-predictor.git
cd wine-quality-predictor
```

### 2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model
```bash
python model_training.py
```

### 4. Run the Flask server
```bash
python app.py
```

### 5. Open the frontend
Open `index.html` in a browser. The form connects to `http://127.0.0.1:5001/predict`.

---

## 🔬 Machine Learning Details

- **Model Type**: e.g., Random Forest, Logistic Regression (based on `model_training.py`)
- **Features Used**: Fixed acidity, Volatile acidity, Citric acid, Residual sugar, etc.
- **Target**: Binary classification of wine quality ("Good" or "Bad")

---

##  Reasoning System

The app uses a hybrid rule-based explanation engine to describe **why** a wine was rated good or bad, based on thresholds for acidity, alcohol, sulphates, etc.

---

##  Evaluation

Run `model_performance.py` to see model accuracy, confusion matrix, and other evaluation metrics.

---

##  Example Wines Supported

The frontend offers quick selections like:
- Merlot
- Cabernet Sauvignon
- Pinot Noir
- Faulty House Red

---

##  Custom Input Validation

The frontend ensures all custom inputs stay within realistic min-max ranges to prevent poor-quality predictions.

---

##  API Endpoint

- **POST** `/predict`  
  **Input**: JSON with wine features  
  **Output**: Predicted label (`Good`/`Bad`) and numeric score  

---

## Credits

Built by [Vatsal Malkari]  
Based on UCI Red Wine Quality dataset.
