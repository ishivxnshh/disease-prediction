import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ---- LOAD DATA ----
data_path = "data/dataset.csv"
df = pd.read_csv(data_path)

# ---- DROP UNUSED COLUMNS ----
df = df.drop(columns=["Patient ID", "Confidence Score (%)"])

# ---- FEATURES & TARGETS ----
X = df[["Age", "Gender", "Symptoms"]]
y = df[["Predicted Disease", "Severity"]]

# ---- ENCODE TARGETS ----
disease_encoder = LabelEncoder()
severity_encoder = LabelEncoder()

y_disease = disease_encoder.fit_transform(y["Predicted Disease"])
y_severity = severity_encoder.fit_transform(y["Severity"])

y_encoded = pd.DataFrame({
    "Disease": y_disease,
    "Severity": y_severity
})

# Save encoders
os.makedirs("model", exist_ok=True)
joblib.dump(disease_encoder, "model/disease_encoder.pkl")
joblib.dump(severity_encoder, "model/severity_encoder.pkl")

# ---- SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ---- PREPROCESSING ----
preprocessor = ColumnTransformer(
    transformers=[
        ('symptoms', TfidfVectorizer(), 'Symptoms'),
        ('gender', OneHotEncoder(handle_unknown='ignore'), ['Gender']),
        ('age', 'passthrough', ['Age'])
    ]
)

# ---- MODEL ----
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
multi_model = MultiOutputClassifier(xgb)

# ---- PIPELINE ----
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', multi_model)
])

# ---- TRAIN ----
pipeline.fit(X_train, y_train)

# ---- EVALUATE ----
y_pred = pipeline.predict(X_test)
disease_acc = accuracy_score(y_test["Disease"], y_pred[:, 0])
severity_acc = accuracy_score(y_test["Severity"], y_pred[:, 1])

print(f"Disease Prediction Accuracy: {disease_acc:.2f}")
print(f"Severity Prediction Accuracy: {severity_acc:.2f}")

# ---- SAVE MODEL ----
joblib.dump(pipeline, "model/disease_severity_model.pkl")
print("Model saved at model/disease_severity_model.pkl")