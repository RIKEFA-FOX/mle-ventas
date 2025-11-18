"""
src/predict_model.py
Predicción profesional: reproduce exactamente el pipeline del entrenamiento.
"""

import os
import pickle
import pandas as pd


MODEL_PATH_DEFAULT = os.path.join("models", "best_model.pkl")
METADATA_PATH_DEFAULT = os.path.join("data", "processed", "metadata_columns.txt")
NEW_DATA_DIR = os.path.join("data", "new")
SCORED_DIR = os.path.join("data", "scored")


def load_model(path=MODEL_PATH_DEFAULT):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Modelo cargado desde: {path}")
    return model


def load_metadata(path=METADATA_PATH_DEFAULT):
    """Carga las columnas usadas por el modelo desde metadata_columns.txt"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No existe metadata_columns.txt en: {path}")

    features = []
    target_col = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("target_col"):
                target_col = line.split(":")[1].strip()

            if line.startswith("-"):
                col = line.replace("-", "").strip()
                features.append(col)

    print("\n--- Metadata cargada ---")
    print("Target:", target_col)
    print("Features usadas por el modelo:")
    for f in features:
        print("  -", f)

    return features, target_col


def prepare_new_data(df, features):
    """Transforma el archivo crudo para que coincida con las features del modelo."""

    df_proc = df.copy()

    # 1. One-Hot Encoding para columnas categóricas originales
    df_proc = pd.get_dummies(df_proc, drop_first=True)

    # 2. Crear columnas faltantes (las one-hot que sí están en metadata pero no en el archivo nuevo)
    for col in features:
        if col not in df_proc.columns:
            df_proc[col] = 0  # rellenar con 0

    # 3. Eliminar columnas que no sirven para el modelo
    df_proc = df_proc[features]

    return df_proc


def predict_on_file(input_path, output_path, features):
    """Procesa un archivo nuevo y genera predicciones."""
    print(f"\n=== Procesando archivo: {input_path} ===")

    df_raw = pd.read_csv(input_path)
    print("Shape original:", df_raw.shape)

    df_ready = prepare_new_data(df_raw, features)

    print("Shape después de preparar:", df_ready.shape)

    model = load_model()

    preds = model.predict(df_ready)

    df_raw["prediction"] = preds

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_raw.to_csv(output_path, index=False)

    print(f"✔ Archivo scored guardado en: {output_path}")


def main():
    print("=== PREDICCIÓN MASIVA ===")

    features, target = load_metadata(METADATA_PATH_DEFAULT)

    os.makedirs(SCORED_DIR, exist_ok=True)

    files = [f for f in os.listdir(NEW_DATA_DIR) if f.endswith(".csv")]

    if not files:
        print(f"❌ No hay archivos CSV en {NEW_DATA_DIR}")
        return

    for file in files:
        input_path = os.path.join(NEW_DATA_DIR, file)
        output_path = os.path.join(SCORED_DIR, f"scored_{file}")

        predict_on_file(input_path, output_path, features)


if __name__ == "__main__":
    main()
