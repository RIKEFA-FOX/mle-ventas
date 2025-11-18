"""
src/make_dataset.py
Script 1: Preparación de datos para el entrenamiento.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_PATH_DEFAULT = os.path.join("data/raw", "data_original.csv")
PROCESSED_DIR = os.path.join("data", "processed")
TRAIN_PATH_DEFAULT = os.path.join(PROCESSED_DIR, "data_train.csv")
TEST_PATH_DEFAULT = os.path.join(PROCESSED_DIR, "data_test.csv")


def load_raw_data(path: str = RAW_PATH_DEFAULT) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo RAW: {path}")
    df = pd.read_csv(path)
    print(f"Dimensión del dataset: {df.shape}")
    return df


def identify_types(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print("\nVariables numéricas:")
    for c in num_cols:
        print(" -", c)

    print("\nVariables categóricas:")
    for c in cat_cols:
        print(" -", c)

    return num_cols, cat_cols


def initial_clean(df: pd.DataFrame, target_col: str = "Phone_sale"):
    df = df.copy()

    # Eliminar columnas tipo ID
    cols_drop = [c for c in df.columns if c.strip().lower() in ["id", "id#", "id_cliente", "customer_id"]]
    if cols_drop:
        print("\nEliminando columnas ID:", cols_drop)
        df = df.drop(columns=cols_drop, errors="ignore")

    # Convertir variable objetivo a 0/1
    if df[target_col].dtype == "O":
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "y": 1, "si": 1, "sí": 1, "no": 0, "n": 0})
        )

    # Validación del objetivo
    valores = set(df[target_col].dropna().unique())
    if valores - {0, 1}:
        raise ValueError(f"{target_col} tiene valores no convertibles a 0/1: {valores}")

    print(f"\nDistribución objetivo {target_col}: {df[target_col].value_counts().to_dict()}")

    return df


def quality_checks(df: pd.DataFrame):
    print("\nNulos por columna:")
    print(df.isna().sum().sort_values(ascending=False))

    negativos = {c: int((df[c] < 0).sum()) for c in df.select_dtypes(include=[np.number]).columns}
    print("\nConteo de valores negativos:")
    print(negativos)

    return negativos


def apply_one_hot_encoding(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print("\nAplicando One-Hot Encoding en columnas:")
    print(cat_cols)

    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    print(f"Shape antes del encoding: {X.shape}")
    print(f"Shape después del encoding: {X_encoded.shape}")

    return X_encoded, y, num_cols, cat_cols


def prepare_splits(X, y, test_size=0.30, random_state=42):
    print("\nRealizando split 70/30 estratificado...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Tamaños:")
    print(" - X_train:", X_train.shape)
    print(" - X_test:", X_test.shape)

    train_df = X_train.copy()
    train_df["Phone_sale"] = y_train

    test_df = X_test.copy()
    test_df["Phone_sale"] = y_test

    return train_df, test_df


def save_splits(train_df, test_df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_PATH_DEFAULT, index=False)
    test_df.to_csv(TEST_PATH_DEFAULT, index=False)

    print(f"\nTrain guardado en: {TRAIN_PATH_DEFAULT}")
    print(f"Test guardado en: {TEST_PATH_DEFAULT}")
    
def save_metadata(X_encoded, target_col="Phone_sale"):
    meta_path = os.path.join(PROCESSED_DIR, "metadata_columns.txt")
    features = list(X_encoded.columns)

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"target_col: {target_col}\n\n")
        f.write("features:\n")
        for c in features:
            f.write(f" - {c}\n")

    print(f"Metadata guardada en: {meta_path}")

def main():
    df = load_raw_data()

    identify_types(df)

    df = initial_clean(df)

    quality_checks(df)

    X_encoded, y, num_cols, cat_cols = apply_one_hot_encoding(df, target_col="Phone_sale")

    train_df, test_df = prepare_splits(X_encoded, y)

    save_splits(train_df, test_df)
    
    save_metadata(X_encoded)

    print("\n>>> Proceso de make_dataset completado.")

if __name__ == "__main__":
    main()
