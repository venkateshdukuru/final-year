import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
model_path = "best_anxiety_model.pkl"
model = joblib.load(model_path)

# Load the label encoders and scaler used during training
encoders_path = "label_encoders.pkl"
scaler_path = "scaler.pkl"

label_encoders = joblib.load(encoders_path)  # Load saved label encoders
scaler = joblib.load(scaler_path)  # Load saved scaler

# Function to take user input from terminal
def get_user_input():
    user_data = {
        "GADE(Generalized Anxiety disorder)": input("GADE (e.g., 'Somewhat difficult'): "),
        "SWL(Satisfaction of work life)": int(input("SWL (1-5): ")),
        "Game": input("Favorite Game (e.g., 'Skyrim'): "),
        "Platform": input("Gaming Platform (e.g., 'PC'): "),
        "Hours": float(input("Hours Spent on Gaming: ")),
        "earnings": input("Earnings (e.g., 'I play for fun'): "),
        "whyplay": input("Reason for Playing (e.g., 'having fun'): "),
        "Gender": input("Gender (e.g., 'Male'): "),
        "Age": int(input("Age: ")),
        "Work": input("Employment Status (e.g., 'Employed'): "),
        "Degree": input("Degree (e.g., 'Bachelor'): "),
        "Birthplace": input("Birthplace (e.g., 'USA'): "),
        "Residence": input("Residence (e.g., 'USA'): ")
    }
    return pd.DataFrame([user_data])

# Get user input
user_df = get_user_input()

# Encode categorical variables using the saved encoders
for col in label_encoders:
    if col in user_df.columns:
        user_df[col] = user_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

# Scale numerical variables using the saved scaler
user_df["Hours"] = scaler.transform(user_df[["Hours"]])

# Make prediction
prediction = model.predict(user_df)

# Output the result
print(f"Predicted Anxiety Result: {prediction[0]}")
