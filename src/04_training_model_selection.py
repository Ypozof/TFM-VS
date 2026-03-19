from pathlib import Path
import json
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "training_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "TenYearCHD"
RANDOM_STATE = 42
N_SPLITS = 5


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


def save_json(path: Path, content: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


def load_data():
    """
    Carga los datos generados en la fase de preprocesamiento.
    """
    x_train_path = OUTPUTS_DIR / "X_train_scaled.csv"
    x_test_path = OUTPUTS_DIR / "X_test_scaled.csv"
    y_train_path = OUTPUTS_DIR / "y_train.csv"
    y_test_path = OUTPUTS_DIR / "y_test.csv"

    required_files = [x_train_path, x_test_path, y_train_path, y_test_path]
    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "No se encontraron los siguientes archivos preprocesados:\n"
            + "\n".join(missing)
        )

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    y_train_df = pd.read_csv(y_train_path)
    y_test_df = pd.read_csv(y_test_path)

    y_train = y_train_df[TARGET_COL] if TARGET_COL in y_train_df.columns else y_train_df.iloc[:, 0]
    y_test = y_test_df[TARGET_COL] if TARGET_COL in y_test_df.columns else y_test_df.iloc[:, 0]

    return X_train, X_test, y_train, y_test


def get_scoring():
    """
    Métricas empleadas en la comparación de modelos.
    """
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
    }


def evaluate_final_model(model, X_test, y_test):
    """
    Evalúa el modelo final en el conjunto de prueba.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        roc_auc = None

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (1)": precision_score(y_test, y_pred, zero_division=0),
        "Recall (1)": recall_score(y_test, y_pred, zero_division=0),
        "F1 (1)": f1_score(y_test, y_pred, zero_division=0),
        "F1-macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc,
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3, zero_division=0)

    return results, cm, report, y_pred, y_prob


# =========================================================
# MODELOS
# =========================================================

def get_baseline_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def get_class_weight_models():
    return {
        "Logistic Regression (class_weight)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Support Vector Machine (class_weight)": SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Decision Tree (class_weight)": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest (class_weight)": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
    }


def get_smote_models():
    return {
        "Logistic Regression (SMOTE)": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "K-Nearest Neighbors (SMOTE)": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Support Vector Machine (SMOTE)": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", SVC(probability=True, random_state=RANDOM_STATE))
        ]),
        "Decision Tree (SMOTE)": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]),
        "Random Forest (SMOTE)": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
        ]),
        "Gradient Boosting (SMOTE)": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
    }


# =========================================================
# FASE 1: COMPARACIÓN INICIAL
# =========================================================

def model_comparison(X_train, y_train):
    print_section("FASE 1 - COMPARACIÓN INICIAL DE MODELOS")

    scoring = get_scoring()
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    modelos = {}
    modelos.update(get_baseline_models())
    modelos.update(get_smote_models())
    modelos.update(get_class_weight_models())

    resultados = []

    for nombre, modelo in modelos.items():
        print(f"\n>>> Evaluando modelo: {nombre}")

        scores = cross_validate(
            estimator=modelo,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        fila = {
            "Modelo": nombre,
            "Accuracy CV": round(scores["test_accuracy"].mean() * 100, 2),
            "Precisión CV": round(scores["test_precision"].mean() * 100, 2),
            "Recall CV": round(scores["test_recall"].mean() * 100, 2),
            "F1 CV": round(scores["test_f1"].mean() * 100, 2),
            "F1-macro CV": round(scores["test_f1_macro"].mean() * 100, 2),
            "ROC-AUC CV": round(scores["test_roc_auc"].mean() * 100, 2),
        }

        resultados.append(fila)

        print(f"Accuracy CV medio:   {fila['Accuracy CV']}%")
        print(f"Precisión CV media:  {fila['Precisión CV']}%")
        print(f"Recall CV medio:     {fila['Recall CV']}%")
        print(f"F1 CV medio:         {fila['F1 CV']}%")
        print(f"F1-macro CV medio:   {fila['F1-macro CV']}%")
        print(f"ROC-AUC CV medio:    {fila['ROC-AUC CV']}%")

    tabla = (
        pd.DataFrame(resultados)
        .sort_values(by=["Recall CV", "F1-macro CV"], ascending=False)
        .reset_index(drop=True)
    )

    print_section("TABLA COMPARATIVA INICIAL")
    print(tabla)

    tabla.to_csv(RESULTS_DIR / "tabla_comparativa_modelos_cv.csv", index=False)

    return tabla


# =========================================================
# SELECCIÓN CLÍNICA DEL MODELO
# =========================================================

def select_clinical_model():
    """
    La selección final se fija de forma explícita en función del criterio clínico
    priorizado en el trabajo: maximizar la detección de pacientes de riesgo
    manteniendo interpretabilidad.
    """
    model_name = "Logistic Regression (class_weight)"
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    return model_name, model


# =========================================================
# FASE 2: AJUSTE DE HIPERPARÁMETROS
# =========================================================

def tune_selected_model(model_name: str, X_train, y_train):
    print_section("FASE 2 - AJUSTE DE HIPERPARÁMETROS DEL MODELO SELECCIONADO")

    if model_name != "Logistic Regression (class_weight)":
        raise ValueError("Este script está configurado para ajustar Logistic Regression (class_weight).")

    estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print(f"\nModelo seleccionado para ajuste: {model_name}")
    print(f"Mejores hiperparámetros: {grid.best_params_}")
    print(f"Mejor recall CV: {grid.best_score_:.4f}")

    save_json(
        RESULTS_DIR / "mejores_hiperparametros.json",
        {
            "modelo": model_name,
            "best_params": grid.best_params_,
            "best_recall_cv": round(float(grid.best_score_), 6),
        }
    )

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# =========================================================
# FASE 3: EVALUACIÓN FINAL EN TEST
# =========================================================

def final_test_evaluation(best_model_name, best_model, X_test, y_test):
    print_section("FASE 3 - EVALUACIÓN FINAL DEL MODELO EN TEST")

    results, cm, report, _, _ = evaluate_final_model(best_model, X_test, y_test)

    print(f"Modelo final evaluado: {best_model_name}\n")
    print("Métricas en test:")
    for k, v in results.items():
        if v is None:
            print(f"- {k}: No disponible")
        else:
            print(f"- {k}: {v:.4f}")

    print("\nMatriz de confusión:")
    print(cm)

    print("\nInforme de clasificación:")
    print(report)

    pd.DataFrame([{
        "Modelo final": best_model_name,
        **{
            k: (round(v * 100, 2) if v is not None else None)
            for k, v in results.items()
        }
    }]).to_csv(RESULTS_DIR / "evaluacion_final_test.csv", index=False)

    save_json(
        RESULTS_DIR / "matriz_confusion_final.json",
        {"confusion_matrix": cm.tolist()}
    )

    save_text(
        RESULTS_DIR / "classification_report_final.txt",
        report
    )

    return results, cm, report


# =========================================================
# FASE 4: CURVAS ROC
# =========================================================

def roc_variants_comparison(X_train, X_test, y_train, y_test):
    """
    Compara variantes clave para apoyar visualmente la elección final del modelo.
    """
    print_section("FASE 4 - CURVAS ROC DE VARIANTES CLAVE")

    modelos = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Logistic Regression (SMOTE)": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Logistic Regression (class_weight)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting (SMOTE)": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
    }

    roc_results = []

    plt.figure(figsize=(10, 8))

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)

        roc_results.append({
            "Modelo": nombre,
            "ROC-AUC Test": round(auc_score, 4)
        })

        plt.plot(fpr, tpr, linewidth=2, label=f"{nombre} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curvas ROC comparativas de variantes seleccionadas")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves_selected_variants.png", dpi=300, bbox_inches="tight")
    plt.close()

    roc_df = pd.DataFrame(roc_results).sort_values(by="ROC-AUC Test", ascending=False)
    roc_df.to_csv(RESULTS_DIR / "roc_auc_selected_variants.csv", index=False)

    print("\nComparativa ROC-AUC en test:")
    print(roc_df)

    return roc_df


def final_model_roc_curve(best_model_name, best_model, X_test, y_test):
    """
    Genera la curva ROC del modelo final.
    """
    print_section("FASE 4B - CURVA ROC DEL MODELO FINAL")

    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"{best_model_name} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC del modelo final")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve_final_model.png", dpi=300, bbox_inches="tight")
    plt.close()

    save_json(
        RESULTS_DIR / "roc_final_model.json",
        {"modelo": best_model_name, "roc_auc_test": round(float(auc_score), 6)}
    )

    print(f"ROC-AUC del modelo final: {auc_score:.4f}")

    return auc_score


# =========================================================
# FASE 5: INTERPRETABILIDAD
# =========================================================

def logistic_coefficients_analysis(best_model_name, best_model, X_train):
    """
    Para regresión logística se analizan los coeficientes del modelo.
    """
    print_section("FASE 5 - INTERPRETABILIDAD DEL MODELO")

    if not hasattr(best_model, "coef_"):
        print("El modelo seleccionado no dispone de coeficientes interpretables.")
        return None

    coefs = pd.Series(best_model.coef_[0], index=X_train.columns).sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )

    print("\nCoeficientes del modelo:")
    print(coefs)

    coef_df = coefs.reset_index()
    coef_df.columns = ["Variable", "Coeficiente"]
    coef_df.to_csv(RESULTS_DIR / "logistic_coefficients.csv", index=False)

    plt.figure(figsize=(10, 6))
    coefs.sort_values().plot(kind="barh")
    plt.title(f"Coeficientes del modelo - {best_model_name}")
    plt.xlabel("Coeficiente")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "logistic_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close()

    return coefs


# =========================================================
# MAIN
# =========================================================

def main():
    print_section("CARGA DE DATOS")
    X_train, X_test, y_train, y_test = load_data()

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    # Fase 1: comparación inicial completa
    tabla_cv = model_comparison(X_train, y_train)

    # Selección clínica explícita del modelo final
    best_model_name, _ = select_clinical_model()
    print_section("SELECCIÓN DEL MODELO FINAL")
    print(f"Modelo seleccionado por criterio clínico: {best_model_name}")

    # Fase 2: ajuste de hiperparámetros del modelo elegido
    best_model, best_params, best_score = tune_selected_model(best_model_name, X_train, y_train)

    # Fase 3: evaluación final en test
    final_results, cm, report = final_test_evaluation(
        best_model_name,
        best_model,
        X_test,
        y_test
    )

    # Fase 4: curvas ROC
    roc_variants = roc_variants_comparison(X_train, X_test, y_train, y_test)
    final_auc = final_model_roc_curve(best_model_name, best_model, X_test, y_test)

    # Fase 5: interpretabilidad
    coefs = logistic_coefficients_analysis(best_model_name, best_model, X_train)

    print_section("RESUMEN FINAL")
    print("Tabla comparativa guardada en:")
    print(RESULTS_DIR / "tabla_comparativa_modelos_cv.csv")

    print("\nMejores hiperparámetros guardados en:")
    print(RESULTS_DIR / "mejores_hiperparametros.json")

    print("\nEvaluación final guardada en:")
    print(RESULTS_DIR / "evaluacion_final_test.csv")

    print("\nMatriz de confusión final guardada en:")
    print(RESULTS_DIR / "matriz_confusion_final.json")

    print("\nClassification report final guardado en:")
    print(RESULTS_DIR / "classification_report_final.txt")

    print("\nCurvas ROC guardadas en:")
    print(RESULTS_DIR / "roc_curves_selected_variants.png")
    print(RESULTS_DIR / "roc_auc_selected_variants.csv")
    print(RESULTS_DIR / "roc_curve_final_model.png")

    if coefs is not None:
        print("\nCoeficientes del modelo guardados en:")
        print(RESULTS_DIR / "logistic_coefficients.csv")
        print(RESULTS_DIR / "logistic_coefficients.png")

    print("\nProceso completado correctamente.")


if __name__ == "__main__":
    main()