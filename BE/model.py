import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "Axiety_ds_modified.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Selecting input features and output
features = [
    "GADE(Generalized Anxiety disorder)", "SWL(Satisfaction of work life)", "Game",
    "Platform", "Hours", "earnings", "whyplay", "Gender", "Age", "Work",
    "Degree", "Birthplace", "Residence"
]
target = "Percentage(RESULT)"

# Handling missing values
df = df.dropna(subset=[target])
df = df.fillna(df.mode().iloc[0])

# Encoding categorical variables
label_encoders = {}
for col in features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Save label encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Standardizing numerical features
scaler = StandardScaler()
df["Hours"] = scaler.fit_transform(df[["Hours"]])

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Encoding target variable
df[target] = LabelEncoder().fit_transform(df[target])

# Splitting data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
}

# Train and evaluate models
best_model = None
best_model_name = ""
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_model = model
        best_model_name = name
        best_accuracy = accuracy

# Save the best model
joblib.dump(best_model, "best_anxiety_model.pkl")
print(f"Best model '{best_model_name}' saved successfully!")
