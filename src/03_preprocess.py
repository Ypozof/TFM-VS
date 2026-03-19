import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "TenYearCHD"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "heart_disease.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def print_section(title: str) -> None:
    """Muestra un encabezado para organizar mejor la salida."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    """Guarda un DataFrame en la carpeta de salida."""
    output_path = OUTPUTS_DIR / filename
    df.to_csv(output_path, index=False)


# =========================================================
# SCRIPT PRINCIPAL
# =========================================================

def main() -> None:
    # -----------------------------------------------------
    # 1) CARGA DEL CONJUNTO DE DATOS
    # -----------------------------------------------------
    print_section("1) CARGA DEL CONJUNTO DE DATOS")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"Ruta del dataset: {DATA_PATH}")
    print(f"Dimensiones del dataset: {df.shape}")
    print("\nPrimeras 5 filas:")
    print(df.head())

    print("\nInformación general del dataset:")
    print(df.info())

    # -----------------------------------------------------
    # 2) VALIDACIÓN DE LA VARIABLE OBJETIVO
    # -----------------------------------------------------
    print_section("2) VALIDACIÓN DE LA VARIABLE OBJETIVO")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"La columna objetivo '{TARGET_COL}' no está en el dataset.\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    print(f"Variable objetivo detectada correctamente: {TARGET_COL}")
    print("\nDistribución de la variable objetivo:")
    print(df[TARGET_COL].value_counts(dropna=False))

    # -----------------------------------------------------
    # 3) REVISIÓN DE VALORES FALTANTES
    # -----------------------------------------------------
    print_section("3) REVISIÓN DE VALORES FALTANTES")

    missing_abs = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    missing_summary = pd.DataFrame({
        "missing_count": missing_abs,
        "missing_percentage": missing_pct.round(2)
    })

    print("Resumen de valores faltantes:")
    print(missing_summary[missing_summary["missing_count"] > 0])

    save_dataframe(
        missing_summary.reset_index().rename(columns={"index": "variable"}),
        "missing_values_summary.csv"
    )

    # -----------------------------------------------------
    # 4) SEPARACIÓN ENTRE VARIABLES PREDICTORAS Y OBJETIVO
    # -----------------------------------------------------
    print_section("4) SEPARACIÓN ENTRE VARIABLES PREDICTORAS Y OBJETIVO")

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")

    # -----------------------------------------------------
    # 5) IDENTIFICACIÓN DE TIPOS DE VARIABLES
    # -----------------------------------------------------
    print_section("5) IDENTIFICACIÓN DE TIPOS DE VARIABLES")

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]

    print("Variables numéricas detectadas:")
    print(numeric_cols)

    if non_numeric_cols:
        print("\nVariables no numéricas detectadas:")
        print(non_numeric_cols)
    else:
        print("\nNo se detectaron variables no numéricas.")

    # En este dataset las variables ya están en formato numérico,
    # incluidas las binarias codificadas como 0 y 1.
    # Por tanto, no es necesario aplicar una codificación adicional.

    # -----------------------------------------------------
    # 6) DIVISIÓN DEL CONJUNTO DE DATOS
    # -----------------------------------------------------
    print_section("6) DIVISIÓN TRAIN / TEST")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    print("\nDistribución de clases en y_train:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nDistribución de clases en y_test:")
    print(y_test.value_counts(normalize=True).round(4))

    # -----------------------------------------------------
    # 7) IMPUTACIÓN DE VALORES FALTANTES
    # -----------------------------------------------------
    print_section("7) IMPUTACIÓN DE VALORES FALTANTES")

    # Se utiliza la mediana por ser una medida robusta en presencia
    # de distribuciones asimétricas y posibles valores extremos.
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train[numeric_cols]),
        columns=numeric_cols,
        index=X_train.index
    )

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test[numeric_cols]),
        columns=numeric_cols,
        index=X_test.index
    )

    print("Imputación realizada con estrategia: median")
    print("Comprobación de nulos en X_train tras imputación:")
    print(X_train_imputed.isnull().sum().sum())

    print("Comprobación de nulos en X_test tras imputación:")
    print(X_test_imputed.isnull().sum().sum())

    # -----------------------------------------------------
    # 8) ESCALADO DE VARIABLES
    # -----------------------------------------------------
    print_section("8) ESCALADO DE VARIABLES")

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=numeric_cols,
        index=X_train_imputed.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=numeric_cols,
        index=X_test_imputed.index
    )

    print("Escalado realizado con StandardScaler")
    print("\nResumen estadístico de X_train escalado:")
    print(X_train_scaled.describe().round(3))

    # -----------------------------------------------------
    # 9) REVISIÓN DEL DESBALANCE DE CLASES
    # -----------------------------------------------------
    print_section("9) REVISIÓN DEL DESBALANCE DE CLASES")

    train_counts = y_train.value_counts().sort_index()
    train_pct = y_train.value_counts(normalize=True).sort_index() * 100

    imbalance_summary = pd.DataFrame({
        "class": train_counts.index,
        "count": train_counts.values,
        "percentage": train_pct.round(2).values
    })

    print("Distribución de clases en el conjunto de entrenamiento:")
    print(imbalance_summary)

    save_dataframe(imbalance_summary, "class_imbalance_summary_train.csv")

    print("\nSe observa un desequilibrio entre clases en la variable objetivo.")
    print(
        "Este aspecto se tendrá en cuenta en la fase de modelado, "
        "donde se compararán distintas estrategias para su tratamiento."
    )

    # -----------------------------------------------------
    # 10) GUARDADO DE ARCHIVOS PREPROCESADOS
    # -----------------------------------------------------
    print_section("10) GUARDADO DE ARCHIVOS PREPROCESADOS")

    save_dataframe(X_train, "X_train_raw.csv")
    save_dataframe(X_test, "X_test_raw.csv")
    save_dataframe(y_train.to_frame(name=TARGET_COL), "y_train.csv")
    save_dataframe(y_test.to_frame(name=TARGET_COL), "y_test.csv")

    save_dataframe(X_train_imputed, "X_train_imputed.csv")
    save_dataframe(X_test_imputed, "X_test_imputed.csv")

    save_dataframe(X_train_scaled, "X_train_scaled.csv")
    save_dataframe(X_test_scaled, "X_test_scaled.csv")

    print(f"Archivos guardados correctamente en: {OUTPUTS_DIR}")

    # -----------------------------------------------------
    # 11) RESUMEN FINAL DEL PREPROCESAMIENTO
    # -----------------------------------------------------
    print_section("11) RESUMEN FINAL DEL PREPROCESAMIENTO")

    print("Resumen del pipeline aplicado:")
    print("- Carga y revisión inicial del dataset")
    print("- Validación de la variable objetivo")
    print("- Identificación de valores faltantes")
    print("- Separación entre variables predictoras y variable objetivo")
    print("- División en entrenamiento y prueba (80/20, estratificada)")
    print("- Imputación de valores faltantes mediante mediana")
    print("- Escalado de variables numéricas con StandardScaler")
    print("- Revisión del desbalance de clases en entrenamiento")
    print("- Exportación de archivos procesados para etapas posteriores")

    print("\nPreprocesamiento completado correctamente.")


if __name__ == "__main__":
    main()