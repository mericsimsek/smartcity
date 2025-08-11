
# utils/predictor.py
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Modeli yükle (ilk seferde)
with open("models/model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

scaler = joblib.load("scaler.pkl") 

df2=pd.read_csv("emlaktahminenson.csv")
# 2. Tahmin fonksiyonu
def predict_house_price(data):
    """
    data: JSON dict (Flask'tan gelen input)
    örnek: {
        'm2_location_interact': [36.224941],
    'price_per_m2': [83739.130435],
    'room_density': [13.999995],
    'location_tier': [4]
    }
    """

    # 3. Data'yı modelin beklediği formata getir
    df = pd.DataFrame([data])  # tek satırlık DataFrame
    
    # 4. Tahmin yap
    X_scaled = scaler.transform(df[data]) if scaler else df[data]
    prediction = model.predict(df)[0]

    return float(prediction)
