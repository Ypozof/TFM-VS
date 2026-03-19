import pandas as pd
from pathlib import Path

# Construimos una ruta robusta al dataset
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart_disease.csv"

df = pd.read_csv(DATA_PATH)

print("Dimensiones del dataset:", df.shape)
print("\nColumnas:")
print(df.columns)
print("\nPrimeras filas:")
print(df.head())