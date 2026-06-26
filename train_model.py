import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Load dataset
df = pd.read_csv("data/heart.csv")

# Encode categorical columns
label_encoders = {}

categorical_columns = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

# Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "models/heart_failure_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("✅ Model saved successfully!")
print("✅ Label encoders saved successfully!")