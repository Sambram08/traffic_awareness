import os
import pickle
import xgboost as xgb
import pandas as pd

# ✅ Define the model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgboost_model.pkl")

# ✅ Load the model once
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    print("✅ XGBoost model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  

# ✅ Define expected features (same as during training)
EXPECTED_FEATURES = [
    "Year", "Start_Lat", "Start_Lng", "Distance(mi)", "Street", "City", "County", "State",
    "Airport_Code", "Temperature(F)", "Wind_Chill(F)", "Visibility(mi)", "Wind_Direction",
    "Weather_Condition", "Traffic_Signal", "Sunrise_Sunset", "TimeDiff"
]

# ✅ Define severity labels
SEVERITY_LABELS = ["Low", "Moderate", "High", "Critical"]

# ✅ Function to make predictions
def predict_model(input_data):
    try:
        if model is None:
            return {"error": "Model not loaded. Check model path and file integrity."}

        # ✅ Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # ✅ Ensure all expected features are present
        for feat in EXPECTED_FEATURES:
            if feat not in input_df.columns:
                input_df[feat] = None  

        # ✅ Convert categorical columns to 'category' type
        cat_cols = ['Street', 'City', 'County', 'State', 'Airport_Code', 'Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset']
        for col in cat_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype('category')

        # ✅ Convert DataFrame to DMatrix
        dinput = xgb.DMatrix(input_df, enable_categorical=True)

        # ✅ Make predictions (probabilities)
        predictions = model.predict(dinput)

        # ✅ Convert probabilities to percentages
        severity_percentages = [round(prob * 100, 2) for prob in predictions[0]]

        # ✅ Get the severity with the highest probability
        max_index = severity_percentages.index(max(severity_percentages))
        final_prediction = SEVERITY_LABELS[max_index]
        final_percentage = severity_percentages[max_index]

        return {
            "severity": final_prediction,
            "confidence": f"{final_percentage}%"  # Show confidence as percentage
        }

    except Exception as e:
        return {"error": str(e)}
