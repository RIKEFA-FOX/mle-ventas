"""
src/train_model.py
Script 2: Entrenamiento del modelo de clasificación.
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

PROCESSED_PATH_DEFAULT = os.path.join("data", "processed", "data_train.csv")
MODEL_PATH_DEFAULT = os.path.join("models", "best_model.pkl")


def load_processed(path: str = PROCESSED_PATH_DEFAULT) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(
    data_path: str = PROCESSED_PATH_DEFAULT,
    model_path: str = MODEL_PATH_DEFAULT,
    target_col: str = "Phone_sale",
):
    df = load_processed(data_path)

    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset procesado.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    print("Accuracy validación:", acc)
    print("\nReporte de clasificación:")
    print(report)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Modelo guardado en: {model_path}")


def main():
    train_model()


if __name__ == "__main__":
    main()
