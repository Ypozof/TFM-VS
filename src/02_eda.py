import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "heart_disease.csv"

FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sns.set(style="whitegrid")

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def save_fig(filepath: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filepath, bbox_inches="tight", dpi=400)
    plt.close()


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def make_output_dirs() -> None:
    subdirs = [
        "01_overview",
        "02_missing_values",
        "03_univariate_numeric",
        "04_univariate_categorical",
        "05_boxplots_by_target",
        "06_distributions_by_target",
        "07_correlations",
        "08_scatterplots",
        "09_optional_pairplots",
    ]
    for subdir in subdirs:
        (FIGURES_DIR / subdir).mkdir(exist_ok=True)


# =========================================================
# SCRIPT PRINCIPAL
# =========================================================

def main():
    make_output_dirs()

    # -----------------------------------------------------
    # 1) CARGA DEL CONJUNTO DE DATOS
    # -----------------------------------------------------
    print_section("1) CARGA DEL CONJUNTO DE DATOS")

    df = pd.read_csv(DATA_PATH)

    print("Ruta del archivo:", DATA_PATH)
    print("Dimensiones del conjunto de datos:", df.shape)
    print("\nColumnas:")
    print(df.columns.tolist())

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nPrimeras filas:")
    print(df.head())

    print("\nResumen estadístico:")
    print(df.describe(include="all"))

    # -----------------------------------------------------
    # 2) INFORMACIÓN GENERAL Y CALIDAD DEL DATO
    # -----------------------------------------------------
    print_section("2) INFORMACIÓN GENERAL Y CALIDAD DEL DATO")

    missing_abs = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    print("\nValores nulos (absolutos):")
    print(missing_abs)

    print("\nValores nulos (%):")
    print(missing_pct.round(2))

    # =====================================================
    # FIGURA 3 DEL TFM
    # Distribución de la variable objetivo
    # =====================================================
    plt.figure(figsize=(7, 5))

    ax = sns.countplot(
        x="TenYearCHD",
        data=df,
        color=sns.color_palette("deep")[0]
    )

    plt.title(
        "Distribución de la variable objetivo (TenYearCHD)",
        fontsize=18,
        fontweight="bold",
        pad=20
    )
    plt.xlabel("TenYearCHD", fontsize=13, fontweight="bold")
    plt.ylabel("Número de pacientes", fontsize=13, fontweight="bold")

    counts = df["TenYearCHD"].value_counts().sort_index()
    total = len(df)

    ymax = counts.max() * 1.18
    ax.set_ylim(0, ymax)

    for i, v in enumerate(counts.values):
        porcentaje = v / total * 100
        ax.text(
            i,
            v + counts.max() * 0.02,
            f"{v}\n({porcentaje:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    save_fig("01_overview/figura01_distribucion_objetivo.png")

    # Figura 4: valores nulos absolutos
    missing_nonzero = missing_abs[missing_abs > 0]
    if not missing_nonzero.empty:
        plt.figure(figsize=(10, 5))
        missing_nonzero.plot(kind="bar")
        plt.title("Valores ausentes por variable", fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Variable", fontsize=13, fontweight="bold")
        plt.ylabel("Número de valores ausentes", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        save_fig("02_missing_values/figura02_valores_nulos_absolutos.png")

    # Figura 5: valores nulos porcentuales
    missing_pct_nonzero = missing_pct[missing_pct > 0]
    if not missing_pct_nonzero.empty:
        plt.figure(figsize=(10, 5))
        missing_pct_nonzero.plot(kind="bar")
        plt.title("Porcentaje de valores ausentes por variable", fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Variable", fontsize=13, fontweight="bold")
        plt.ylabel("Porcentaje de valores ausentes (%)", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        save_fig("02_missing_values/figura03_valores_nulos_porcentaje.png")

    # -----------------------------------------------------
    # 3) IDENTIFICACIÓN DE VARIABLES
    # -----------------------------------------------------
    print_section("3) IDENTIFICACIÓN DE VARIABLES")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_like_cols = [
        "male",
        "education",
        "currentSmoker",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
        "TenYearCHD",
    ]

    continuous_cols = [col for col in numeric_cols if col not in categorical_like_cols]

    print("Variables numéricas detectadas:")
    print(numeric_cols)

    print("\nVariables continuas seleccionadas:")
    print(continuous_cols)

    print("\nVariables binarias/categóricas seleccionadas:")
    print(categorical_like_cols)

    continuous_labels = {
        "age": "Edad (años)",
        "cigsPerDay": "Cigarrillos por día",
        "totChol": "Colesterol total (mg/dl)",
        "sysBP": "Presión arterial sistólica (mmHg)",
        "diaBP": "Presión arterial diastólica (mmHg)",
        "BMI": "Índice de masa corporal (kg/m²)",
        "heartRate": "Frecuencia cardiaca (latidos/min)",
        "glucose": "Glucosa (mg/dl)",
    }

    categorical_labels = {
        "male": "Sexo (0 = mujer, 1 = hombre)",
        "education": "Nivel educativo",
        "currentSmoker": "Fumador actual (0 = no, 1 = sí)",
        "BPMeds": "Medicación antihipertensiva (0 = no, 1 = sí)",
        "prevalentStroke": "Antecedente de ictus (0 = no, 1 = sí)",
        "prevalentHyp": "Hipertensión prevalente (0 = no, 1 = sí)",
        "diabetes": "Diabetes (0 = no, 1 = sí)",
        "TenYearCHD": "Evento coronario a 10 años (0 = no, 1 = sí)",
    }

    # -----------------------------------------------------
    # 4) HISTOGRAMAS CONJUNTOS DE VARIABLES CONTINUAS
    # -----------------------------------------------------
    print_section("4) HISTOGRAMAS CONJUNTOS DE VARIABLES CONTINUAS")

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        sns.histplot(df[col].dropna(), bins=25, kde=True, edgecolor="black", ax=axes[i])
        axes[i].set_title(col, fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Valor", fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Frecuencia", fontsize=12, fontweight="bold")
        axes[i].tick_params(axis="both", labelsize=11)

    for j in range(len(continuous_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Distribución conjunta de las principales variables continuas",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )

    plt.subplots_adjust(hspace=0.45, wspace=0.30, top=0.90)
    plt.savefig(
        FIGURES_DIR / "03_univariate_numeric/figura04_histogramas_conjuntos_variables_continuas.png",
        dpi=400,
        bbox_inches="tight"
    )
    plt.close()

    # -----------------------------------------------------
    # 5) BOXPLOTS CONJUNTOS DE VARIABLES CONTINUAS
    # -----------------------------------------------------
    print_section("5) BOXPLOTS CONJUNTOS DE VARIABLES CONTINUAS")

    plt.figure(figsize=(16, 8))

    sns.boxplot(
        data=df[continuous_cols],
        orient="h"
    )

    plt.title(
        "Boxplots conjuntos de las principales variables continuas",
        fontsize=20,
        fontweight="bold",
        pad=18
    )
    plt.xlabel("Valor", fontsize=13, fontweight="bold")
    plt.ylabel("Variable", fontsize=13, fontweight="bold")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=12, fontweight="bold")

    save_fig("03_univariate_numeric/figura05_boxplots_conjuntos_variables_continuas.png")

    # -----------------------------------------------------
    # 6) ANÁLISIS UNIVARIANTE DETALLADO DE VARIABLES CONTINUAS
    # -----------------------------------------------------
    print_section("6) ANÁLISIS UNIVARIANTE DETALLADO DE VARIABLES CONTINUAS")

    for col in continuous_cols:
        label = continuous_labels.get(col, col)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), bins=30, kde=True, edgecolor="black")
        plt.title(f"Histograma de {col}", fontsize=15, fontweight="bold")
        plt.xlabel(label, fontsize=12, fontweight="bold")
        plt.ylabel("Número de pacientes", fontsize=12, fontweight="bold")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot de {col}", fontsize=15, fontweight="bold")
        plt.xlabel(label, fontsize=12, fontweight="bold")

        save_fig(f"03_univariate_numeric/{col}_histograma_boxplot.png")

    # -----------------------------------------------------
    # 7) ANÁLISIS UNIVARIANTE DE VARIABLES BINARIAS/CATEGÓRICAS
    # -----------------------------------------------------
    print_section("7) ANÁLISIS UNIVARIANTE DE VARIABLES BINARIAS/CATEGÓRICAS")

    for col in categorical_like_cols:
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=col, data=df)
        plt.title(f"Distribución de {col}", fontsize=15, fontweight="bold")
        plt.xlabel(categorical_labels.get(col, col), fontsize=12, fontweight="bold")
        plt.ylabel("Número de pacientes", fontsize=12, fontweight="bold")

        counts = df[col].value_counts(dropna=False).sort_index()
        for i, v in enumerate(counts.values):
            ax.text(
                i,
                v + max(counts.values) * 0.01,
                str(v),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        save_fig(f"04_univariate_categorical/{col}_countplot.png")

    # -----------------------------------------------------
    # 8) BOXPLOTS POR CLASE SEGÚN TenYearCHD
    # -----------------------------------------------------
    print_section("8) BOXPLOTS POR CLASE SEGÚN TenYearCHD")

    for col in continuous_cols:
        label = continuous_labels.get(col, col)

        plt.figure(figsize=(7, 5))
        sns.boxplot(x="TenYearCHD", y=col, data=df)
        plt.title(f"{col} según TenYearCHD", fontsize=15, fontweight="bold")
        plt.xlabel("TenYearCHD", fontsize=12, fontweight="bold")
        plt.ylabel(label, fontsize=12, fontweight="bold")

        save_fig(f"05_boxplots_by_target/{col}_por_TenYearCHD_boxplot.png")

    # -----------------------------------------------------
    # 9) DISTRIBUCIONES POR CLASE
    # -----------------------------------------------------
    print_section("9) DISTRIBUCIONES POR CLASE")

    compare_dist_cols = [
        "age",
        "sysBP",
        "diaBP",
        "BMI",
        "totChol",
        "glucose",
        "heartRate",
        "cigsPerDay",
    ]

    for col in compare_dist_cols:
        label = continuous_labels.get(col, col)

        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=df,
            x=col,
            hue="TenYearCHD",
            bins=25,
            kde=True,
            element="step",
            stat="count",
            common_norm=False
        )
        plt.title(f"Distribución de {col} según TenYearCHD", fontsize=15, fontweight="bold")
        plt.xlabel(label, fontsize=12, fontweight="bold")
        plt.ylabel("Frecuencia", fontsize=12, fontweight="bold")

        save_fig(f"06_distributions_by_target/{col}_distribucion_por_TenYearCHD.png")

    # -----------------------------------------------------
    # 10) MATRICES DE CORRELACIÓN
    # -----------------------------------------------------
    print_section("10) MATRICES DE CORRELACIÓN")

    corr_all = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr_all, dtype=bool))

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_all,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matriz de correlación entre variables clínicas", fontsize=18, fontweight="bold", pad=18)
    save_fig("07_correlations/matriz_correlacion_completa_triangular.png")

    key_corr_cols = ["age", "BMI", "totChol", "sysBP", "diaBP", "glucose", "heartRate", "cigsPerDay"]
    corr_key = df[key_corr_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_key,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )
    plt.title("Correlación entre variables clínicas seleccionadas", fontsize=18, fontweight="bold", pad=18)
    save_fig("07_correlations/matriz_correlacion_reducida.png")

    # -----------------------------------------------------
    # 11) SCATTERPLOTS ENTRE VARIABLES RELEVANTES
    # -----------------------------------------------------
    print_section("11) SCATTERPLOTS ENTRE VARIABLES RELEVANTES")

    scatter_pairs = [
        ("age", "totChol"),
        ("BMI", "sysBP"),
        ("age", "sysBP"),
        ("glucose", "BMI"),
        ("age", "glucose"),
        ("totChol", "sysBP"),
    ]

    for x_col, y_col in scatter_pairs:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue="TenYearCHD",
            alpha=0.7
        )
        plt.title(f"{x_col} vs {y_col} (coloreado por TenYearCHD)", fontsize=15, fontweight="bold")
        plt.xlabel(continuous_labels.get(x_col, x_col), fontsize=12, fontweight="bold")
        plt.ylabel(continuous_labels.get(y_col, y_col), fontsize=12, fontweight="bold")

        save_fig(f"08_scatterplots/{x_col}_vs_{y_col}_por_TenYearCHD.png")

    # -----------------------------------------------------
    # 12) VARIABLES BINARIAS SEGÚN TenYearCHD
    # -----------------------------------------------------
    print_section("12) VARIABLES BINARIAS SEGÚN TenYearCHD")

    binary_compare_cols = [
        "male",
        "currentSmoker",
        "diabetes",
        "prevalentHyp",
        "BPMeds",
        "prevalentStroke",
    ]

    for col in binary_compare_cols:
        plt.figure(figsize=(7, 5))
        sns.countplot(x=col, hue="TenYearCHD", data=df)
        plt.title(f"{col} según TenYearCHD", fontsize=15, fontweight="bold")
        plt.xlabel(categorical_labels.get(col, col), fontsize=12, fontweight="bold")
        plt.ylabel("Número de pacientes", fontsize=12, fontweight="bold")

        save_fig(f"04_univariate_categorical/{col}_segun_TenYearCHD.png")

    # -----------------------------------------------------
    # 13) PAIRPLOT REDUCIDO
    # -----------------------------------------------------
    print_section("13) PAIRPLOT REDUCIDO")

    pairplot_cols = ["age", "sysBP", "BMI", "glucose", "TenYearCHD"]
    pairplot_df = df[pairplot_cols].dropna().copy()

    g = sns.pairplot(pairplot_df, hue="TenYearCHD", corner=True)
    g.fig.suptitle("Pairplot reducido de variables clínicas seleccionadas", y=1.02, fontsize=18, fontweight="bold")
    g.savefig(
        FIGURES_DIR / "09_optional_pairplots/pairplot_variables_clinicas.png",
        bbox_inches="tight",
        dpi=400
    )
    plt.close()

    # -----------------------------------------------------
    # 14) RESUMEN FINAL
    # -----------------------------------------------------
    print_section("14) RESUMEN FINAL")

    print("EDA ampliado completado correctamente.")
    print("Las figuras se han guardado en:")
    print(FIGURES_DIR)

    total_figs = sum(1 for _ in FIGURES_DIR.rglob("*.png"))
    print(f"\nNúmero total de figuras generadas: {total_figs}")


if __name__ == "__main__":
    main()
