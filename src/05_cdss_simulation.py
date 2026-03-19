# src/05_cdss_simulation.py

from pathlib import Path
import json
import warnings

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "heart_disease.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "training_results"
CDSS_DIR = OUTPUTS_DIR / "cdss_simulation"

CDSS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "TenYearCHD"
RANDOM_STATE = 42

# Umbral de decisión del CDSS.
# Se puede ajustar según el objetivo clínico.
# Un valor menor que 0.50 favorece la detección de positivos.
THRESHOLD = 0.35

BEST_PARAMS_PATH = RESULTS_DIR / "mejores_hiperparametros.json"


# =========================================================
# UTILIDADES
# =========================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def save_text(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(CDSS_DIR / filename, index=False)


# =========================================================
# CARGA DE DATOS Y PARÁMETROS
# =========================================================

def load_dataset():
    """
    Carga el dataset original del proyecto.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo '{TARGET_COL}' en el dataset.")

    return df


def load_best_params():
    """
    Recupera los mejores hiperparámetros guardados en la fase 04.
    Si el archivo no existe, usa una configuración por defecto razonable.
    """
    default_params = {
        "C": 0.01,
        "solver": "liblinear"
    }

    if not BEST_PARAMS_PATH.exists():
        print(
            "No se encontró el archivo de mejores hiperparámetros. "
            "Se utilizarán valores por defecto."
        )
        return default_params

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_params = data.get("best_params", {})
    cleaned_params = {}

    for key, value in raw_params.items():
        if key.startswith("model__"):
            cleaned_params[key.replace("model__", "")] = value
        else:
            cleaned_params[key] = value

    if "C" not in cleaned_params:
        cleaned_params["C"] = default_params["C"]
    if "solver" not in cleaned_params:
        cleaned_params["solver"] = default_params["solver"]

    return cleaned_params


# =========================================================
# PREPARACIÓN DEL MODELO
# =========================================================

def prepare_training_data(df: pd.DataFrame):
    """
    Separa variables predictoras y objetivo.
    """
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    return X, y


def fit_preprocessing_objects(X: pd.DataFrame):
    """
    Ajusta el imputador y el escalador sobre el conjunto de variables predictoras.
    """
    feature_names = X.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=feature_names
    )

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=feature_names
    )

    return imputer, scaler, X_scaled, feature_names


def train_final_model(X_scaled: pd.DataFrame, y: pd.Series, best_params: dict):
    """
    Entrena el modelo final: Regresión Logística con ajuste de pesos de clase.
    """
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        C=best_params["C"],
        solver=best_params["solver"]
    )

    model.fit(X_scaled, y)
    return model


# =========================================================
# ENTRADA INTERACTIVA DEL PACIENTE
# =========================================================

def ask_float(prompt: str, min_value=None, max_value=None):
    while True:
        value = input(prompt).strip().replace(",", ".")
        try:
            value = float(value)

            if min_value is not None and value < min_value:
                print(f"El valor debe ser mayor o igual que {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"El valor debe ser menor o igual que {max_value}.")
                continue

            return value

        except ValueError:
            print("Introduce un número válido.")


def ask_optional_float(prompt: str, min_value=None, max_value=None):
    while True:
        value = input(prompt).strip().replace(",", ".")

        if value == "":
            return None

        try:
            value = float(value)

            if min_value is not None and value < min_value:
                print(f"El valor debe ser mayor o igual que {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"El valor debe ser menor o igual que {max_value}.")
                continue

            return value

        except ValueError:
            print("Introduce un número válido o deja vacío si no se conoce.")


def ask_binary(prompt: str):
    while True:
        value = input(prompt).strip().lower()

        if value in {"1", "si", "sí", "s", "yes", "y"}:
            return 1
        if value in {"0", "no", "n"}:
            return 0

        print("Introduce 1/0 o sí/no.")


def collect_patient_data():
    """
    Recoge por consola los datos del paciente.
    Los nombres coinciden con las variables del dataset Framingham.
    """
    print_section("INTRODUCCIÓN DE DATOS DEL PACIENTE")
    print("Introduce los datos del paciente. Si alguna variable no se conoce, pulsa Enter cuando se permita.\n")

    current_smoker = ask_binary("¿Es fumador actual? (1 = sí, 0 = no): ")

    if current_smoker == 1:
        cigs_per_day = ask_optional_float(
            "Número de cigarrillos al día [Enter si no se conoce]: ",
            min_value=0,
            max_value=100
        )
    else:
        cigs_per_day = 0.0

    patient = {
        "male": ask_binary("Sexo biológico (1 = hombre, 0 = mujer): "),
        "age": ask_float("Edad (años): ", min_value=18, max_value=110),
        "education": ask_optional_float("Nivel educativo (1-4) [Enter si no se conoce]: ", min_value=1, max_value=4),
        "currentSmoker": current_smoker,
        "cigsPerDay": cigs_per_day,
        "BPMeds": ask_optional_float("¿Toma medicación para la presión arterial? (1/0) [Enter si no se conoce]: ", min_value=0, max_value=1),
        "prevalentStroke": ask_binary("¿Antecedentes de ictus? (1 = sí, 0 = no): "),
        "prevalentHyp": ask_binary("¿Hipertensión previa? (1 = sí, 0 = no): "),
        "diabetes": ask_binary("¿Diabetes? (1 = sí, 0 = no): "),
        "totChol": ask_optional_float("Colesterol total (mg/dL) [Enter si no se conoce]: ", min_value=50, max_value=700),
        "sysBP": ask_float("Presión arterial sistólica (mmHg): ", min_value=60, max_value=300),
        "diaBP": ask_float("Presión arterial diastólica (mmHg): ", min_value=30, max_value=200),
        "BMI": ask_optional_float("Índice de masa corporal - BMI [Enter si no se conoce]: ", min_value=10, max_value=80),
        "heartRate": ask_optional_float("Frecuencia cardíaca (lpm) [Enter si no se conoce]: ", min_value=20, max_value=220),
        "glucose": ask_optional_float("Glucosa (mg/dL) [Enter si no se conoce]: ", min_value=30, max_value=600),
    }

    return pd.DataFrame([patient])


# =========================================================
# PREPROCESAMIENTO DEL PACIENTE
# =========================================================

def preprocess_patient(patient_df: pd.DataFrame, imputer, scaler, feature_names):
    """
    Aplica al paciente exactamente el mismo preprocesamiento que al conjunto de entrenamiento.
    """
    patient_df = patient_df[feature_names].copy()

    patient_imputed = pd.DataFrame(
        imputer.transform(patient_df),
        columns=feature_names
    )

    patient_scaled = pd.DataFrame(
        scaler.transform(patient_imputed),
        columns=feature_names
    )

    return patient_imputed, patient_scaled


# =========================================================
# PREDICCIÓN E INTERPRETACIÓN
# =========================================================

def predict_risk(model, patient_scaled: pd.DataFrame, threshold: float):
    """
    Realiza la predicción usando una probabilidad y un umbral configurable.
    """
    prob = float(model.predict_proba(patient_scaled)[0, 1])
    pred = 1 if prob >= threshold else 0
    label = "Alto riesgo" if pred == 1 else "Bajo riesgo"

    return prob, pred, label


def generate_clinical_message(prob: float, label: str):
    """
    Devuelve una interpretación orientativa del resultado.
    """
    if label == "Alto riesgo":
        return (
            "El sistema clasifica al paciente como de alto riesgo cardiovascular. "
            "Este resultado sugiere la conveniencia de una valoración clínica prioritaria "
            "y de una vigilancia más estrecha. Debe interpretarse siempre como apoyo al "
            "juicio clínico, no como diagnóstico definitivo."
        )

    return (
        "El sistema clasifica al paciente como de bajo riesgo cardiovascular. "
        "Este resultado no excluye la valoración clínica ni la necesidad de pruebas "
        "complementarias si la situación asistencial lo requiere."
    )


def get_top_coefficients(model, feature_names, top_n=5):
    """
    Ordena los coeficientes del modelo para mostrar las variables con mayor influencia.
    """
    coef_df = pd.DataFrame({
        "Variable": feature_names,
        "Coeficiente": model.coef_[0]
    }).sort_values(by="Coeficiente", ascending=False)

    return coef_df, coef_df.head(top_n)


def save_simulation_outputs(
    patient_raw: pd.DataFrame,
    patient_imputed: pd.DataFrame,
    patient_scaled: pd.DataFrame,
    prob: float,
    pred: int,
    label: str,
    message: str,
    coef_df: pd.DataFrame,
    top_coef_df: pd.DataFrame
):
    """
    Guarda todos los resultados generados durante la simulación.
    """
    save_dataframe(patient_raw, "paciente_introducido_raw.csv")
    save_dataframe(patient_imputed, "paciente_imputado.csv")
    save_dataframe(patient_scaled, "paciente_escalado.csv")
    save_dataframe(coef_df, "coeficientes_modelo_cdss.csv")
    save_dataframe(top_coef_df, "top_coeficientes_cdss.csv")

    summary_df = pd.DataFrame([{
        "Probabilidad_riesgo": round(prob, 4),
        "Prediccion_binaria": pred,
        "Clasificacion": label,
        "Umbral_utilizado": THRESHOLD
    }])
    save_dataframe(summary_df, "resultado_cdss.csv")

    report_text = (
        "SIMULACIÓN INTERACTIVA DEL CDSS\n"
        + "=" * 60 + "\n\n"
        + f"Probabilidad estimada de riesgo: {prob:.4f}\n"
        + f"Predicción binaria: {pred}\n"
        + f"Clasificación final: {label}\n"
        + f"Umbral utilizado: {THRESHOLD}\n\n"
        + "Interpretación orientativa:\n"
        + message + "\n\n"
        + "Variables con mayor influencia positiva en la predicción:\n"
        + top_coef_df.to_string(index=False)
        + "\n\n"
        + "Aviso: esta simulación representa una posible implementación práctica "
          "del modelo como sistema de apoyo a la decisión clínica. No sustituye el "
          "juicio clínico profesional ni constituye una herramienta diagnóstica validada "
          "para uso real.\n"
    )

    save_text(CDSS_DIR / "interpretacion_cdss.txt", report_text)


# =========================================================
# MAIN
# =========================================================

def main():
    print_section("CARGA DE DATOS PARA LA SIMULACIÓN DEL CDSS")
    df = load_dataset()
    X, y = prepare_training_data(df)

    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    print_section("CARGA DE HIPERPARÁMETROS")
    best_params = load_best_params()
    print("Mejores hiperparámetros recuperados:")
    print(best_params)

    print_section("AJUSTE DEL PREPROCESAMIENTO")
    imputer, scaler, X_scaled, feature_names = fit_preprocessing_objects(X)
    print("Imputador y escalador ajustados correctamente.")

    print_section("REENTRENAMIENTO DEL MODELO FINAL")
    model = train_final_model(X_scaled, y, best_params)
    print("Modelo final entrenado correctamente.")

    print_section("SIMULACIÓN INTERACTIVA DE UN CASO CLÍNICO")
    patient_raw = collect_patient_data()

    print("\nPaciente introducido:")
    print(patient_raw)

    patient_imputed, patient_scaled = preprocess_patient(
        patient_df=patient_raw,
        imputer=imputer,
        scaler=scaler,
        feature_names=feature_names
    )

    print_section("PREDICCIÓN DEL CDSS")
    prob, pred, label = predict_risk(model, patient_scaled, threshold=THRESHOLD)
    message = generate_clinical_message(prob, label)

    print(f"Probabilidad estimada de riesgo: {prob:.4f}")
    print(f"Predicción binaria: {pred}")
    print(f"Clasificación final: {label}")
    print(f"Umbral utilizado: {THRESHOLD:.2f}")

    print("\nInterpretación orientativa:")
    print(message)

    print_section("INTERPRETABILIDAD DEL MODELO")
    coef_df, top_coef_df = get_top_coefficients(model, feature_names, top_n=5)

    print("Variables con mayor influencia positiva:")
    print(top_coef_df)

    print_section("GUARDADO DE RESULTADOS")
    save_simulation_outputs(
        patient_raw=patient_raw,
        patient_imputed=patient_imputed,
        patient_scaled=patient_scaled,
        prob=prob,
        pred=pred,
        label=label,
        message=message,
        coef_df=coef_df,
        top_coef_df=top_coef_df
    )

    print("Archivos generados en:")
    print(CDSS_DIR / "paciente_introducido_raw.csv")
    print(CDSS_DIR / "paciente_imputado.csv")
    print(CDSS_DIR / "paciente_escalado.csv")
    print(CDSS_DIR / "resultado_cdss.csv")
    print(CDSS_DIR / "coeficientes_modelo_cdss.csv")
    print(CDSS_DIR / "top_coeficientes_cdss.csv")
    print(CDSS_DIR / "interpretacion_cdss.txt")

    print_section("NOTA FINAL")
    print(
        "Esta simulación representa una posible implementación práctica del modelo "
        "como sistema de apoyo a la decisión clínica (CDSS)."
    )


if __name__ == "__main__":
    main()