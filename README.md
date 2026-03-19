</> Markdown 
src/05_cdss_simulation.py


---

## 📂 Estructura del proyecto


TFM/
│
├── src/
│ ├── 03_preprocessing.py
│ ├── 04_training_model_selection.py
│ ├── 05_cdss_simulation.py
│
├── data/
│ └── heart_disease.csv
│
├── outputs/
│ ├── training_results/
│ └── cdss_simulation/
│
├── figures/
│
└── README.md


---

## ▶️ Ejecución

Para ejecutar el sistema:

```bash
python src/05_cdss_simulation.py

El sistema solicitará los datos del paciente y generará:

Probabilidad de riesgo

Clasificación clínica

Interpretación del resultado

Archivos exportados con resultados

🧪 Ejemplo de uso

Paciente simulado:

Edad: 62

Presión arterial: 150 mmHg

Colesterol: 240 mg/dL

Fumador: Sí

Resultado:

Probabilidad: 0.78

Clasificación: Alto riesgo

🏥 Enfoque clínico

El sistema está orientado a:

Apoyo en triaje

Evaluación inicial del paciente

Priorización de atención

⚠️ No sustituye el juicio clínico

⚠️ Limitaciones

Dataset orientado a riesgo a largo plazo (10 años)

No incluye variables críticas de diagnóstico agudo:

ECG

Troponinas

Desbalance de clases

Falta de validación externa

Modelo lineal

🔮 Líneas futuras

Validación en entornos clínicos reales

Integración en sistemas de historia clínica electrónica

Mejora del recall

Incorporación de nuevas variables clínicas

Técnicas de explicabilidad (XAI)

Monitorización del modelo

👩‍💻 Autor

Trabajo de Fin de Máster (TFM)
Máster en Inteligencia Artificial aplicada a Salud
