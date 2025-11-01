# 🤖 EVi - Exit Duration Predictor

<div align="center">

**Machine Learning Powered Exit Duration Prediction System**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-red.svg)](https://streamlit.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.8-green.svg)](https://catboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Advanced ML models to predict occupant exit duration based on environmental and temporal factors*

</div>

---

## 📋 Overview

**EVi (Exit Duration Predictor)** is a sophisticated machine learning application that predicts how long occupants will stay outside their home before returning. The system uses advanced ensemble models and feature engineering to provide accurate predictions based on weather conditions, temporal patterns, and historical data.

### Key Features

- 🎯 **High Accuracy**: Ensemble stacking model with multiple base estimators
- 🔧 **Advanced Feature Engineering**: Automated transformation pipeline
- 🌤️ **Weather-Aware**: Considers 17 different weather conditions
- ⏰ **Temporal Intelligence**: Day, hour, and holiday pattern recognition
- 📊 **Interactive UI**: Streamlit web application
- 🔄 **Real-time Predictions**: Instant forecasting

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to EviTrain directory
cd EviTrain

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install streamlit catboost lightgbm scikit-learn seaborn plotly scipy joblib pandas numpy matplotlib pillow
```

### Running the Application

```bash
# From the notebooks directory
cd notebooks
streamlit run app.py
```

The application will be available at `http://localhost:8501`

---

## 📊 Model Architecture

### Base Models

1. **CatBoost Regressor** - Gradient boosting with categorical features
2. **LightGBM Regressor** - Fast gradient boosting
3. **Decision Tree Regressor** - Simple tree-based model
4. **Random Forest Regressor** - Ensemble of decision trees
5. **Ridge Regressor** - Linear regression with L2 regularization

### Meta-Model

**Stacking Regressor** combines predictions from all base models for optimal performance.

### Feature Engineering

The system automatically engineers features including:

- **Temporal Features**: DayOfWeek, Hour, WeekOfYear, DayOfYear
- **Weather Features**: Temperature, Wind, Humidity
- **Interaction Features**: Temp × Humidity
- **Historical Features**: Lag features, Rolling means
- **Person Features**: Average duration per person, per hour
- **Categorical Encoding**: One-hot encoding for weather categories

---

## 📁 Project Structure

```
EviTrain/
│
├── 📁 data/                                  # Training datasets
│   ├── Cleaned_synthetic_family_data_less_than_48.csv
│   └── holidays.csv
│
├── 📁 models/                                # Trained models
│   ├── optimized_stacking_regressor_advanced.pkl
│   └── feature_engineering_transformer.pkl
│
├── 📁 notebooks/                             # Application files
│   ├── app.py                               # Streamlit main app
│   ├── evi_modeling.ipynb                   # Model training notebook
│   ├── main_notebook.ipynb                  # Main analysis notebook
│   ├── evi_logo.png                         # Application logo
│   └── outputs/                             # Generated outputs
│
├── 📁 src/                                  # Source code
│   ├── transformers.py                      # Feature engineering
│   └── utils.py                             # Utility functions
│
├── 📄 requirements.txt                       # Dependencies
└── 📄 README.md                             # This file
```

---

## 🎮 Usage

### Streamlit Web Interface

1. **Launch the application**:
   ```bash
   streamlit run notebooks/app.py
   ```

2. **Input Parameters**:
   - Select or enter timestamp
   - Choose day of the week
   - Select weather conditions
   - Enter temperature, wind speed, and humidity
   - Enter person ID

3. **Get Prediction**:
   - Click "Predict Exit Duration"
   - View results in hours:minutes:seconds format

### Programmatic Usage

```python
import joblib
import pandas as pd

# Load model and transformer
model = joblib.load('models/optimized_stacking_regressor_advanced.pkl')
transformer = joblib.load('models/feature_engineering_transformer.pkl')

# Prepare input data
data = pd.DataFrame({
    'Timestamp': ['2025-05-01 08:00:00'],
    'Weather': [0],  # Sunny
    'PersonID': ['John'],
    'Event': ['Exit'],
    'Temp': [35],
    'Wind': [5],
    'Humidity': [50],
    'DayOfWeek': [3]  # Thursday
})

# Transform and predict
transformed_data = transformer.transform(data)
prediction_hours = model.predict(transformed_data)[0]

print(f"Predicted exit duration: {prediction_hours:.2f} hours")
```

---

## 📈 Model Performance

### Training Metrics

- **Algorithm**: Stacking Regressor
- **Cross-Validation**: k-fold with proper time series splitting
- **Data Size**: ~9,850 records
- **Feature Count**: 20+ engineered features
- **Target Variable**: Exit duration in hours

### Feature Importance

The model considers multiple factors:

1. **Weather Conditions** (35%)
   - Temperature
   - Wind speed
   - Humidity
   - Weather category

2. **Temporal Patterns** (30%)
   - Day of week
   - Hour of day
   - Holidays and weekends

3. **Historical Behavior** (25%)
   - Person-specific patterns
   - Recent exit durations
   - Average durations

4. **Interactions** (10%)
   - Environmental interactions
   - Temporal × Weather effects

---

## 🔧 Customization

### Adding New Features

Edit `src/transformers.py`:

```python
def transform(self, X):
    X = X.copy()
    
    # Add your custom features here
    X['CustomFeature'] = X['ExistingFeature'] * 2
    
    return X
```

### Retraining the Model

Use the Jupyter notebook `notebooks/evi_modeling.ipynb`:

1. Load your new training data
2. Adjust feature engineering if needed
3. Tune hyperparameters
4. Train and evaluate
5. Save the new model

---

## 📊 Data Requirements

### Input Format

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | datetime | Event timestamp (YYYY-MM-DD HH:MM:SS) |
| `Weather` | int | Weather code (0-16) |
| `PersonID` | str | Unique person identifier |
| `Event` | str | Event type (Enter/Exit) |
| `Temp` | float | Temperature in Celsius |
| `Wind` | float | Wind speed in km/h |
| `Humidity` | float | Humidity percentage |
| `DayOfWeek` | int | Day of week (0=Monday, 6=Sunday) |

### Weather Codes

- 0: Sunny
- 1: Clear
- 2: Scattered clouds
- 3: Passing clouds
- 4: Partly sunny
- 5: Low level haze
- 6: Fog
- 7: Rain Passing clouds
- 8: Thunderstorms Passing clouds
- 9: Overcast
- 10: Mild
- 11: Duststorm
- 12: Light rain Overcast
- 13: Rain Overcast
- 14: Rain Partly sunny
- 15: Light rain Partly sunny
- 16: Broken clouds

---

## 🧪 Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Quality

```bash
# Format code
black src/ notebooks/app.py

# Lint code
flake8 src/ notebooks/app.py

# Type checking
mypy src/
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the main repository LICENSE file for details.

---

## 🔗 Related Projects

- [EcooVision Main Project](../README.md)
- [Face Recognition Module](../facerecognition/)
- [Energy Calculator](../elec/)

---

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Contact: support@ecoovision.ai

---

<div align="center">

**Part of the EcooVision Intelligent System**

[⬆ Back to Main README](../README.md)

</div>

