import pickle
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Marks",
    "StudyHours",
    "Attendance",
    "AssignmentsCompleted",
]


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df["Result"].astype(str).str.strip()

    if X.isna().all().any():
        raise ValueError("A feature column contains no numeric data.")

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two records.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Test accuracy:", round(
        accuracy_score(y_test, predictions), 3
    ))

    # Train final model on all available records.
    model.fit(X, y)

    model_path = MODEL_DIR / "student_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print("Model saved:", model_path)


if __name__ == "__main__":
    main()