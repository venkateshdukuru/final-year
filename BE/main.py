from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import os
import hashlib
import jwt
import datetime
import uuid
import joblib
import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# JWT Secret Key
SECRET_KEY = "123454321"
ALGORITHM = "HS256"

# Function to establish database connection
def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# Function to hash passwords
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Create `users_list` table if not exists
def create_users_table():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users_list (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            """)
            conn.commit()

# Create `student_anxiety_prediction` table if not exists
def create_anxiety_table():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_anxiety_prediction (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id),
                    GADE TEXT,
                    SWL INTEGER,
                    Game TEXT,
                    Platform TEXT,
                    Hours FLOAT,
                    earnings TEXT,
                    whyplay TEXT,
                    Gender TEXT,
                    Age INTEGER,
                    Work TEXT,
                    Degree TEXT,
                    Birthplace TEXT,
                    Residence TEXT,
                    Predicted_Anxiety_Level INTEGER,
                    Anxiety_Category TEXT,
                    Recommendation TEXT,
                    Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

# Run table creation on startup
create_users_table()
create_anxiety_table()

# User Signup Model
class UserSignUp(BaseModel):
    name: str
    email: str
    password: str

# User Login Model
class UserLogin(BaseModel):
    email: str
    password: str

# Signup Route
@app.post("/signup")
def signup(user: UserSignUp):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT email FROM users_list WHERE email = %s", (user.email,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Email already exists!")

            hashed_password = hash_password(user.password)
            cursor.execute("INSERT INTO users_list (name, email, password) VALUES (%s, %s, %s)",
                           (user.name, user.email, hashed_password))
            conn.commit()
    return {"message": "User registered successfully!"}

# Login Route
@app.post("/login")
def login(user: UserLogin):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, name, email, password FROM users_list WHERE email = %s", (user.email,))
            existing_user = cursor.fetchone()
            if not existing_user or existing_user["password"] != hash_password(user.password):
                raise HTTPException(status_code=400, detail="Invalid email or password!")

            token_payload = {
                "sub": existing_user["email"],
                "user_id": str(existing_user["id"]),
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
            }
            token = jwt.encode(token_payload, SECRET_KEY, algorithm=ALGORITHM)
    return {"message": "Login successful!", "token": token}
# Load the trained model, encoders, and scaler
try:
    model = joblib.load("best_anxiety_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("Error loading model files:", str(e))
    model, label_encoders, scaler = None, None, None

# Define anxiety categories
ANXIETY_LEVELS = {
    0: "Low Anxiety(0%)",
    1: "Moderate Anxiety(1-25%)",
    2: "High Anxiety(26-50%)",
    3: "Severe Anxiety(51-75%)",
    4: "Extreme Anxiety(75-100%)"
}

# Define recommendations based on anxiety level
RECOMMENDATIONS = {
    0: "You're doing great! Maintain a balanced lifestyle.",
    1: "Try relaxation techniques and ensure work-life balance.",
    2: "Consider speaking with a counselor or making lifestyle changes.",
    3: "Seek professional help and support from loved ones.",
    4: "Immediate help is recommended. Contact a mental health professional."
}

# Define input model for predictions
class InputData(BaseModel):
    GADE_Generalized_Anxiety_disorder: str
    SWL_Satisfaction_of_work_life: int
    Game: str
    Platform: str
    Hours: float
    earnings: str
    whyplay: str
    Gender: str
    Age: int
    Work: str
    Degree: str
    Birthplace: str
    Residence: str

# Root Endpoint
@app.get("/")
def home():
    return {"message": "Anxiety Level Prediction API is running!"}

# Prediction Endpoint
@app.post("/predict")
def predict(data: InputData):
    if not model or not label_encoders or not scaler:
        raise HTTPException(status_code=500, detail="Model files are missing!")

    try:
        user_data = {
            "GADE(Generalized Anxiety disorder)": data.GADE_Generalized_Anxiety_disorder,
            "SWL(Satisfaction of work life)": data.SWL_Satisfaction_of_work_life,
            "Game": data.Game,
            "Platform": data.Platform,
            "Hours": data.Hours,
            "earnings": data.earnings,
            "whyplay": data.whyplay,
            "Gender": data.Gender,
            "Age": data.Age,
            "Work": data.Work,
            "Degree": data.Degree,
            "Birthplace": data.Birthplace,
            "Residence": data.Residence
        }
        user_df = pd.DataFrame([user_data])

        for col in label_encoders:
            if col in user_df.columns:
                user_df[col] = user_df[col].map(
                    lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
                )

        user_df["Hours"] = scaler.transform(user_df[["Hours"]])
        prediction = model.predict(user_df)[0]
        anxiety_label = ANXIETY_LEVELS.get(int(prediction), "Unexpected Level")
        recommendation = RECOMMENDATIONS.get(int(prediction), "No specific recommendation available.")

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO student_anxiety_prediction (GADE, SWL, Game, Platform, Hours, earnings, whyplay, Gender, Age, Work, Degree, Birthplace, Residence, Predicted_Anxiety_Level, Anxiety_Category, Recommendation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (data.GADE_Generalized_Anxiety_disorder, data.SWL_Satisfaction_of_work_life, data.Game, data.Platform, data.Hours, data.earnings, data.whyplay, data.Gender, data.Age, data.Work, data.Degree, data.Birthplace, data.Residence, int(prediction), anxiety_label, recommendation))
        conn.commit()
        cursor.close()
        conn.close()

        return {"Predicted Anxiety Level": int(prediction), "Anxiety Category": anxiety_label, "Recommendation": recommendation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# @app.get("/user-anxiety-stats")
# def get_user_anxiety_stats():
#     conn = get_db_connection()
#     cursor = conn.cursor(cursor_factory=RealDictCursor)  # Ensure dictionary output
#     cursor.execute("""
#         SELECT Predicted_Anxiety_Level AS predicted_anxiety_level, COUNT(*) AS count
#         FROM student_anxiety_prediction
#         GROUP BY Predicted_Anxiety_Level
#         ORDER BY Predicted_Anxiety_Level ASC
#     """)
#     stats = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     if not stats:
#         return {"message": "No anxiety data available."}

#     print(stats)  # Debugging output

#     anxiety_distribution = {str(row["predicted_anxiety_level"]): row["count"] for row in stats}
#     return {"anxiety_distribution": anxiety_distribution}
# @app.get("/user-anxiety-stats")
# def get_user_anxiety_stats(email: str = None):
#     conn = get_db_connection()
#     cursor = conn.cursor(cursor_factory=RealDictCursor)

#     if email:
#         cursor.execute("""
#             SELECT Predicted_Anxiety_Level AS predicted_anxiety_level, COUNT(*) AS count
#             FROM student_anxiety_prediction
#             WHERE email = %s
#             GROUP BY Predicted_Anxiety_Level
#             ORDER BY Predicted_Anxiety_Level ASC
#         """, (email,))
#     else:
#         cursor.execute("""
#             SELECT Predicted_Anxiety_Level AS predicted_anxiety_level, COUNT(*) AS count
#             FROM student_anxiety_prediction
#             GROUP BY Predicted_Anxiety_Level
#             ORDER BY Predicted_Anxiety_Level ASC
#         """)

#     stats = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     if not stats:
#         return {"message": "No anxiety data available."}

#     anxiety_distribution = {str(row["predicted_anxiety_level"]): row["count"] for row in stats}
#     return {"anxiety_distribution": anxiety_distribution}
# Fetch list of unique users (for dropdown)

@app.get("/user-anxiety-stats")
def get_anxiety_stats():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        cursor.execute("""
            SELECT Predicted_Anxiety_Level AS predicted_anxiety_level, COUNT(*) AS count
            FROM student_anxiety_prediction
            GROUP BY Predicted_Anxiety_Level
            ORDER BY Predicted_Anxiety_Level ASC
        """)
        
        stats = cursor.fetchall()
        if not stats:
            return {"message": "No anxiety data available."}

        # Format the data
        anxiety_distribution = {str(row["predicted_anxiety_level"]): row["count"] for row in stats}

        return {"anxiety_distribution": anxiety_distribution}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        cursor.close()
        conn.close()













# from fastapi import FastAPI, HTTPException
# import joblib
# import pandas as pd
# import psycopg2
# import os
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from fastapi.middleware.cors import CORSMiddleware
# from psycopg2.extras import RealDictCursor
# import hashlib

# # Load environment variables
# load_dotenv()

# # Initialize FastAPI app
# app = FastAPI()

# # Enable CORS for frontend connection
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # Database Configuration
# DB_CONFIG = {
#     "dbname": os.getenv("DB_NAME"),
#     "user": os.getenv("DB_USER"),
#     "password": os.getenv("DB_PASSWORD"),
#     "host": os.getenv("DB_HOST"),
#     "port": os.getenv("DB_PORT")
# }

# # Function to establish database connection
# def get_db_connection():
#     try:
#         conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
#         return conn
#     except Exception as e:
#         print("Database connection failed:", str(e))
#         raise HTTPException(status_code=500, detail="Database connection failed")

# # Create `users_list` table if not exists
# def create_users_table():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS users_list (
#             id SERIAL PRIMARY KEY,
#             name TEXT NOT NULL,
#             email TEXT UNIQUE NOT NULL,
#             password TEXT NOT NULL
#         )
#     """)
#     conn.commit()
#     cursor.close()
#     conn.close()

# # Create `student_anxiety_prediction` table if not exists
# def create_anxiety_table():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS student_anxiety_prediction (
#             id SERIAL PRIMARY KEY,
#             GADE TEXT,
#             SWL INTEGER,
#             Game TEXT,
#             Platform TEXT,
#             Hours FLOAT,
#             earnings TEXT,
#             whyplay TEXT,
#             Gender TEXT,
#             Age INTEGER,
#             Work TEXT,
#             Degree TEXT,
#             Birthplace TEXT,
#             Residence TEXT,
#             Predicted_Anxiety_Level INTEGER,
#             Anxiety_Category TEXT,
#             Recommendation TEXT,
#             Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     """)
#     conn.commit()
#     cursor.close()
#     conn.close()

# # Run table creation on startup
# create_users_table()
# create_anxiety_table()

# # User Signup Model
# class UserSignUp(BaseModel):
#     name: str
#     email: str
#     password: str

# # Function to hash passwords
# def hash_password(password: str) -> str:
#     return hashlib.sha256(password.encode()).hexdigest()

# # Signup Route
# @app.post("/signup")
# def signup(user: UserSignUp):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT email FROM users_list WHERE email = %s", (user.email,))
#     existing_user = cursor.fetchone()
#     if existing_user:
#         raise HTTPException(status_code=400, detail="Email already exists!")

#     hashed_password = hash_password(user.password)
#     cursor.execute("INSERT INTO users_list (name, email, password) VALUES (%s, %s, %s)",
#                    (user.name, user.email, hashed_password))
#     conn.commit()
#     cursor.close()
#     conn.close()
#     return {"message": "User registered successfully!"}

# # Load the trained model, encoders, and scaler
# try:
#     model = joblib.load("best_anxiety_model.pkl")
#     label_encoders = joblib.load("label_encoders.pkl")
#     scaler = joblib.load("scaler.pkl")
# except Exception as e:
#     print("Error loading model files:", str(e))
#     model, label_encoders, scaler = None, None, None

# # Define anxiety categories
# ANXIETY_LEVELS = {
#     0: "Low Anxiety",
#     1: "Moderate Anxiety",
#     2: "High Anxiety",
#     3: "Severe Anxiety",
#     4: "Extreme Anxiety"
# }

# # Define recommendations based on anxiety level
# RECOMMENDATIONS = {
#     0: "You're doing great! Maintain a balanced lifestyle.",
#     1: "Try relaxation techniques and ensure work-life balance.",
#     2: "Consider speaking with a counselor or making lifestyle changes.",
#     3: "Seek professional help and support from loved ones.",
#     4: "Immediate help is recommended. Contact a mental health professional."
# }

# # Define input model for predictions
# class InputData(BaseModel):
#     GADE_Generalized_Anxiety_disorder: str
#     SWL_Satisfaction_of_work_life: int
#     Game: str
#     Platform: str
#     Hours: float
#     earnings: str
#     whyplay: str
#     Gender: str
#     Age: int
#     Work: str
#     Degree: str
#     Birthplace: str
#     Residence: str

# # Root Endpoint
# @app.get("/")
# def home():
#     return {"message": "Anxiety Level Prediction API is running!"}

# # Prediction Endpoint
# @app.post("/predict")
# def predict(data: InputData):
#     if not model or not label_encoders or not scaler:
#         raise HTTPException(status_code=500, detail="Model files are missing!")

#     try:
#         user_data = {
#             "GADE(Generalized Anxiety disorder)": data.GADE_Generalized_Anxiety_disorder,
#             "SWL(Satisfaction of work life)": data.SWL_Satisfaction_of_work_life,
#             "Game": data.Game,
#             "Platform": data.Platform,
#             "Hours": data.Hours,
#             "earnings": data.earnings,
#             "whyplay": data.whyplay,
#             "Gender": data.Gender,
#             "Age": data.Age,
#             "Work": data.Work,
#             "Degree": data.Degree,
#             "Birthplace": data.Birthplace,
#             "Residence": data.Residence
#         }
#         user_df = pd.DataFrame([user_data])

#         for col in label_encoders:
#             if col in user_df.columns:
#                 user_df[col] = user_df[col].map(
#                     lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
#                 )

#         user_df["Hours"] = scaler.transform(user_df[["Hours"]])
#         prediction = model.predict(user_df)[0]
#         anxiety_label = ANXIETY_LEVELS.get(int(prediction), "Unexpected Level")
#         recommendation = RECOMMENDATIONS.get(int(prediction), "No specific recommendation available.")

#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO student_anxiety_prediction (
#                 GADE, SWL, Game, Platform, Hours, earnings, whyplay, Gender, Age, Work, Degree, 
#                 Birthplace, Residence, Predicted_Anxiety_Level, Anxiety_Category, Recommendation
#             ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """, (
#             data.GADE_Generalized_Anxiety_disorder, data.SWL_Satisfaction_of_work_life, data.Game,
#             data.Platform, data.Hours, data.earnings, data.whyplay, data.Gender, data.Age,
#             data.Work, data.Degree, data.Birthplace, data.Residence, int(prediction),
#             anxiety_label, recommendation
#         ))
#         conn.commit()
#         cursor.close()
#         conn.close()

#         return {"Predicted Anxiety Level": int(prediction), "Anxiety Category": anxiety_label, "Recommendation": recommendation}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))









# from fastapi import FastAPI
# import joblib
# import pandas as pd
# import psycopg2
# import os
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from fastapi.middleware.cors import CORSMiddleware

# # Load environment variables from .env file
# load_dotenv()
# app = FastAPI()

# app.add_middleware(
#        CORSMiddleware,
#        allow_origins=["http://localhost:3000"],  # Adjust this to your frontend URL
#        allow_credentials=True,
#        allow_methods=["*"],
#        allow_headers=["*"],
#    )

# # Load the trained model, encoders, and scaler
# model = joblib.load("best_anxiety_model.pkl")
# label_encoders = joblib.load("label_encoders.pkl")
# scaler = joblib.load("scaler.pkl")

# # Define anxiety categories
# ANXIETY_LEVELS = {
#     0: "Low Anxiety",
#     1: "Moderate Anxiety",
#     2: "High Anxiety",
#     3: "Severe Anxiety",
#     4: "Extreme Anxiety"
# }

# # Define recommendations based on anxiety level
# RECOMMENDATIONS = {
#     0: "You're doing great! Maintain a balanced lifestyle.",
#     1: "Try relaxation techniques and ensure work-life balance.",
#     2: "Consider speaking with a counselor or making lifestyle changes.",
#     3: "Seek professional help and support from loved ones.",
#     4: "Immediate help is recommended. Contact a mental health professional."
# }

# # Load database credentials from environment variables
# DB_CONFIG = {
#     "dbname": os.getenv("DB_NAME"),
#     "user": os.getenv("DB_USER"),
#     "password": os.getenv("DB_PASSWORD"),
#     "host": os.getenv("DB_HOST"),
#     "port": os.getenv("DB_PORT")
# }

# # Function to connect to the database
# def get_db_connection():
#     return psycopg2.connect(**DB_CONFIG)

# # Create table if it doesn't exist
# def create_table():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS student_anxiety_prediction (
#             id SERIAL PRIMARY KEY,
#             GADE TEXT,
#             SWL INTEGER,
#             Game TEXT,
#             Platform TEXT,
#             Hours FLOAT,
#             earnings TEXT,
#             whyplay TEXT,
#             Gender TEXT,
#             Age INTEGER,
#             Work TEXT,
#             Degree TEXT,
#             Birthplace TEXT,
#             Residence TEXT,
#             Predicted_Anxiety_Level INTEGER,
#             Anxiety_Category TEXT,
#             Recommendation TEXT,
#             Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     """)
#     conn.commit()
#     cursor.close()
#     conn.close()

# # Run table creation on startup
# create_table()

# # Define input model
# class InputData(BaseModel):
#     GADE_Generalized_Anxiety_disorder: str
#     SWL_Satisfaction_of_work_life: int
#     Game: str
#     Platform: str
#     Hours: float
#     earnings: str
#     whyplay: str
#     Gender: str
#     Age: int
#     Work: str
#     Degree: str
#     Birthplace: str
#     Residence: str

# @app.get("/")
# def home():
#     return {"message": "Anxiety Level Prediction API is running!"}

# @app.post("/predict")
# def predict(data: InputData):
#     try:
#         # Convert input data to DataFrame
#         user_data = {
#             "GADE(Generalized Anxiety disorder)": data.GADE_Generalized_Anxiety_disorder,
#             "SWL(Satisfaction of work life)": data.SWL_Satisfaction_of_work_life,
#             "Game": data.Game,
#             "Platform": data.Platform,
#             "Hours": data.Hours,
#             "earnings": data.earnings,
#             "whyplay": data.whyplay,
#             "Gender": data.Gender,
#             "Age": data.Age,
#             "Work": data.Work,
#             "Degree": data.Degree,
#             "Birthplace": data.Birthplace,
#             "Residence": data.Residence
#         }
#         user_df = pd.DataFrame([user_data])

#         # Encode categorical variables
#         for col in label_encoders:
#             if col in user_df.columns:
#                 user_df[col] = user_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

#         # Scale numerical variable
#         user_df["Hours"] = scaler.transform(user_df[["Hours"]])

#         # Make prediction
#         prediction = model.predict(user_df)[0]
#         anxiety_label = ANXIETY_LEVELS.get(int(prediction), "Unexpected Level")
#         recommendation = RECOMMENDATIONS.get(int(prediction), "No specific recommendation available.")

#         # Store the prediction result in the database
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO student_anxiety_prediction (
#                 GADE, SWL, Game, Platform, Hours, earnings, whyplay, Gender, Age, Work, Degree, 
#                 Birthplace, Residence, Predicted_Anxiety_Level, Anxiety_Category, Recommendation
#             ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """, (
#             data.GADE_Generalized_Anxiety_disorder, data.SWL_Satisfaction_of_work_life, data.Game, 
#             data.Platform, data.Hours, data.earnings, data.whyplay, data.Gender, data.Age, 
#             data.Work, data.Degree, data.Birthplace, data.Residence, int(prediction), 
#             anxiety_label, recommendation
#         ))
#         conn.commit()
#         cursor.close()
#         conn.close()

#         return {
#             "Predicted Anxiety Level": int(prediction),
#             "Anxiety Category": str(anxiety_label),
#             "Recommendation": str(recommendation),
#             "message": "Prediction stored in the database successfully!"
#         }

#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

















# from fastapi import FastAPI
# import joblib
# import pandas as pd
# from pydantic import BaseModel
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# app = FastAPI()

# # Load the trained model, encoders, and scaler
# model = joblib.load("best_anxiety_model.pkl")
# label_encoders = joblib.load("label_encoders.pkl")
# scaler = joblib.load("scaler.pkl")

# # Define anxiety categories
# ANXIETY_LEVELS = {
#     0: "Low Anxiety",
#     1: "Moderate Anxiety",
#     2: "High Anxiety",
#     3: "Severe Anxiety",
#     4: "Extreme Anxiety"  # Added missing level
# }

# # Define recommendations based on anxiety level
# RECOMMENDATIONS = {
#     0: "You're doing great! Maintain a balanced lifestyle.",
#     1: "Try relaxation techniques and ensure work-life balance.",
#     2: "Consider speaking with a counselor or making lifestyle changes.",
#     3: "Seek professional help and support from loved ones.",
#     4: "Immediate help is recommended. Contact a mental health professional."
# }

# # Define input model
# class InputData(BaseModel):
#     GADE_Generalized_Anxiety_disorder: str
#     SWL_Satisfaction_of_work_life: int
#     Game: str
#     Platform: str
#     Hours: float
#     earnings: str
#     whyplay: str
#     Gender: str
#     Age: int
#     Work: str
#     Degree: str
#     Birthplace: str
#     Residence: str

# @app.get("/")
# def home():
#     return {"message": "Anxiety Level Prediction API is running!"}

# @app.post("/predict")
# def predict(data: InputData):
#     try:
#         # Convert input data to DataFrame with expected column names
#         user_data = {
#             "GADE(Generalized Anxiety disorder)": data.GADE_Generalized_Anxiety_disorder,
#             "SWL(Satisfaction of work life)": data.SWL_Satisfaction_of_work_life,
#             "Game": data.Game,
#             "Platform": data.Platform,
#             "Hours": data.Hours,
#             "earnings": data.earnings,
#             "whyplay": data.whyplay,
#             "Gender": data.Gender,
#             "Age": data.Age,
#             "Work": data.Work,
#             "Degree": data.Degree,
#             "Birthplace": data.Birthplace,
#             "Residence": data.Residence
#         }
#         user_df = pd.DataFrame([user_data])

#         # Encode categorical variables
#         for col in label_encoders:
#             if col in user_df.columns:
#                 user_df[col] = user_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

#         # Scale numerical variable
#         user_df["Hours"] = scaler.transform(user_df[["Hours"]])

#         # Make prediction
#         prediction = model.predict(user_df)[0]

#         # Ensure the prediction is within expected levels
#         anxiety_label = ANXIETY_LEVELS.get(int(prediction), "Unexpected Level")
#         recommendation = RECOMMENDATIONS.get(int(prediction), "No specific recommendation available.")

#         return {
#             "Predicted Anxiety Level": int(prediction),
#             "Anxiety Category": str(anxiety_label),
#             "Recommendation": str(recommendation)
#         }

#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)











# from fastapi import FastAPI
# import joblib
# import pandas as pd
# from pydantic import BaseModel
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# app = FastAPI()

# # Load the trained model, encoders, and scaler
# model = joblib.load("best_anxiety_model.pkl")
# label_encoders = joblib.load("label_encoders.pkl")
# scaler = joblib.load("scaler.pkl")

# # Define anxiety categories
# ANXIETY_LEVELS = {
#     0: "Low Anxiety",
#     1: "Moderate Anxiety",
#     2: "High Anxiety",
#     3: "Severe Anxiety"
# }

# # Define recommendations based on anxiety level
# RECOMMENDATIONS = {
#     0: "You're doing great! Maintain a balanced lifestyle.",
#     1: "Try relaxation techniques and ensure work-life balance.",
#     2: "Consider speaking with a counselor or making lifestyle changes.",
#     3: "Seek professional help and support from loved ones."
# }

# # Define input model
# class InputData(BaseModel):
#     GADE_Generalized_Anxiety_disorder: str
#     SWL_Satisfaction_of_work_life: int
#     Game: str
#     Platform: str
#     Hours: float
#     earnings: str
#     whyplay: str
#     Gender: str
#     Age: int
#     Work: str
#     Degree: str
#     Birthplace: str
#     Residence: str

# @app.get("/")
# def home():
#     return {"message": "Anxiety Level Prediction API is running!"}

# @app.post("/predict")
# def predict(data: InputData):
#     try:
#         # Convert input data to DataFrame with expected column names
#         user_data = {
#             "GADE(Generalized Anxiety disorder)": data.GADE_Generalized_Anxiety_disorder,
#             "SWL(Satisfaction of work life)": data.SWL_Satisfaction_of_work_life,
#             "Game": data.Game,
#             "Platform": data.Platform,
#             "Hours": data.Hours,
#             "earnings": data.earnings,
#             "whyplay": data.whyplay,
#             "Gender": data.Gender,
#             "Age": data.Age,
#             "Work": data.Work,
#             "Degree": data.Degree,
#             "Birthplace": data.Birthplace,
#             "Residence": data.Residence
#         }
#         user_df = pd.DataFrame([user_data])

#         # Encode categorical variables
#         for col in label_encoders:
#             if col in user_df.columns:
#                 user_df[col] = user_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

#         # Scale numerical variable
#         user_df["Hours"] = scaler.transform(user_df[["Hours"]])

#         # Make prediction
#         prediction = model.predict(user_df)[0]  # Convert NumPy int64 to int
#         anxiety_label = ANXIETY_LEVELS.get(int(prediction), "Unknown")
#         recommendation = RECOMMENDATIONS.get(int(prediction), "No recommendation available.")

#         return {
#             "Predicted Anxiety Level": int(prediction),  # Fix serialization issue
#             "Anxiety Category": str(anxiety_label),
#             "Recommendation": str(recommendation)
#         }

#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)











# from fastapi import FastAPI
# import joblib
# import pandas as pd
# from pydantic import BaseModel
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# app = FastAPI()

# # Load the trained model, encoders, and scaler
# model = joblib.load("best_anxiety_model.pkl")
# label_encoders = joblib.load("label_encoders.pkl")
# scaler = joblib.load("scaler.pkl")

# # Define Pydantic model with correct field names
# class InputData(BaseModel):
#     GADE_Generalized_Anxiety_disorder: str
#     SWL_Satisfaction_of_work_life: int
#     Game: str
#     Platform: str
#     Hours: float
#     earnings: str
#     whyplay: str
#     Gender: str
#     Age: int
#     Work: str
#     Degree: str
#     Birthplace: str
#     Residence: str

# @app.get("/")
# def home():
#     return {"message": "Anxiety Level Prediction API is running!"}

# @app.post("/predict")
# def predict(data: InputData):
#     try:
#         # Convert input data to DataFrame with expected column names
#         user_data = {
#             "GADE(Generalized Anxiety disorder)": data.GADE_Generalized_Anxiety_disorder,
#             "SWL(Satisfaction of work life)": data.SWL_Satisfaction_of_work_life,
#             "Game": data.Game,
#             "Platform": data.Platform,
#             "Hours": data.Hours,
#             "earnings": data.earnings,
#             "whyplay": data.whyplay,
#             "Gender": data.Gender,
#             "Age": data.Age,
#             "Work": data.Work,
#             "Degree": data.Degree,
#             "Birthplace": data.Birthplace,
#             "Residence": data.Residence
#         }
#         user_df = pd.DataFrame([user_data])

#         # Encode categorical variables
#         for col in label_encoders:
#             if col in user_df.columns:
#                 user_df[col] = user_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

#         # Scale numerical variable
#         user_df["Hours"] = scaler.transform(user_df[["Hours"]])

#         # Make prediction
#         prediction = model.predict(user_df)
#         return {"Predicted Anxiety Level": int(prediction[0])}

#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
