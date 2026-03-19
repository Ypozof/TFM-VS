# 🩺 Sistema de Apoyo a la Decisión Clínica (CDSS) para la Predicción de Riesgo Cardiovascular

## 📌 Resumen del proyecto

Este proyecto presenta el desarrollo de un **Sistema de Apoyo a la Decisión Clínica (Clinical Decision Support System, CDSS)** basado en técnicas de aprendizaje automático, orientado a la **estimación del riesgo cardiovascular** a partir de variables clínicas fácilmente disponibles.

El sistema está diseñado como una herramienta de apoyo en entornos asistenciales, especialmente en fases tempranas de atención clínica (**triaje**), donde la rapidez y la capacidad de priorización son críticas.

El modelo final implementado es una **Regresión Logística con ajuste de pesos de clase (class_weight)**, optimizada para mejorar la detección de pacientes en riesgo, priorizando la métrica **recall**.

---

## 🎯 Objetivos

### Objetivo principal

Desarrollar un modelo predictivo capaz de estimar el riesgo cardiovascular y apoyar la toma de decisiones clínicas en contextos asistenciales.

### Objetivos específicos

* Analizar y comprender la distribución de los datos clínicos
* Tratar problemas de calidad de datos
* Abordar el desbalance de clases
* Comparar algoritmos de clasificación
* Seleccionar un modelo clínicamente adecuado
* Optimizar hiperparámetros
* Evaluar rendimiento en test
* Implementar un CDSS

---

## 🧠 Dataset

**Framingham Heart Study**

* Dataset clínico validado
* Predicción de enfermedad coronaria a 10 años
* Variables clínicas y demográficas

---

## ⚙️ Metodología

Basado en CRISP-DM:

1. EDA
2. Preprocesamiento (imputación, escalado, balanceo)
3. Modelado
4. Selección
5. Optimización (GridSearchCV)
6. Evaluación
7. Interpretabilidad

---

## 🤖 Modelo final

**Regresión Logística (class_weight)**

✔ Mejora recall
✔ Interpretable
✔ Estable
✔ Bajo sobreajuste

---

## 🖥️ Implementación del CDSS

### Características:

* Entrada interactiva
* Preprocesamiento automático
* Predicción de riesgo
* Clasificación (alto/bajo)
* Interpretación clínica
* Exportación de resultados

### Archivo principal:

```
src/05_cdss_simulation.py
```

---

## 📂 Estructura del proyecto

```
TFM/
├── src/
│   ├── 03_preprocessing.py
│   ├── 04_training_model_selection.py
│   ├── 05_cdss_simulation.py
├── data/
│   └── heart_disease.csv
├── outputs/
├── figures/
└── README.md
```

---

## ▶️ Ejecución

```
python src/05_cdss_simulation.py
```

---

## 🧪 Ejemplo

Paciente:

* Edad: 62
* Colesterol: 240
* Fumador: Sí

Resultado:

* Probabilidad: 0.78
* Clasificación: Alto riesgo

---

## 🏥 Enfoque clínico

* Triaje
* Priorización
* Apoyo a decisión

⚠️ No sustituye al médico

---

## ⚠️ Limitaciones

* Dataset a largo plazo
* Sin ECG ni troponinas
* Desbalance
* Sin validación externa

---

## 🔮 Futuro

* Validación clínica
* Integración hospitalaria
* Mejora recall
* XAI

---

## 👩‍💻 Autor

Yaiza Pozo Fernández
TFM – Máster en IA aplicada a Salud

---

## 📌 Nota

Uso académico. No válido para uso clínico real.
