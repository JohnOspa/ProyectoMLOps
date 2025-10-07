# 🧠 Proyecto MLOps 

## 📘 Descripción General

Este proyecto implementa un flujo completo de **Machine Learning y MLOps**, desde la exploración inicial de los datos hasta el entrenamiento, registro de modelos y pruebas de predicción.  
El objetivo es aplicar buenas prácticas de **orquestación, versionamiento y automatización** dentro de un entorno reproducible.

El proyecto incluye:
- Un **análisis exploratorio (EDA)** en Jupyter Notebook.  
- Un módulo de **entrenamiento y registro de modelos** gestionado con **MLflow**.  
- Scripts de **ingesta y carga de datos** automatizados.  
- Una sección de **pruebas y API de predicción**.

---

## 📂 Estructura del Repositorio

ProyectoMLOps-main/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── notebooks/
│   └── 01_EDA.ipynb
│
└── src/
    └── app/
        ├── test/
        │   ├── predict_api.py
        │   ├── test_api.py
        │   ├── pyproject.toml
        │   └── uv.lock
        │
        └── train/
            ├── ingesta.ipynb
            ├── load_data.py
            ├── backend.db
            ├── mlflow.db
            ├── prefect_run_id.txt
            ├── mlartifacts/
            └── mlruns/

---

## ⚙️ Requisitos y Dependencias

El entorno está gestionado mediante **pyproject.toml** y **uv.lock**, lo que garantiza la reproducibilidad.  
Asegúrate de tener instaladas las siguientes herramientas:

- **Python 3.10+**
- **MLflow**
- **Prefect**
- **Pandas**, **NumPy**, **scikit-learn**
- **uv** (para manejar el entorno si se usa `uv.lock`)

Instalación recomendada:

```bash
# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
uv sync



