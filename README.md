# ProyectoMLOps - Sistema Completo de Machine Learning con Enfoque MLOps

## Propósito del Proyecto

Este proyecto implementa un **sistema completo de MLOps (Machine Learning Operations)** para la predicción de ingresos utilizando el dataset "Adult" de UCI ML Repository. El sistema está diseñado para demostrar las mejores prácticas en el ciclo de vida completo de un proyecto de machine learning, desde la exploración de datos hasta el deployment y monitoreo en producción.

## Problema de Negocio

**Objetivo**: Predecir si una persona gana más de $50K USD al año basándose en características demográficas y laborales.

**Aplicaciones**:

- Segmentación de clientes para productos financieros
- Análisis de mercado para estrategias de marketing
- Estudios socioeconómicos y políticas públicas
- Sistemas de recomendación personalizados

## Arquitectura del Sistema

### **Componentes Principales**

```
ProyectoMLOps/
│
├── notebooks/              # Análisis Exploratorio
│   └── 01_EDA.ipynb          # Exploración y visualización de datos
│
├── src/app/
│   ├── train/             # Pipeline de Entrenamiento
│   │   ├── load_data.py       # Pipeline principal con Prefect + MLflow
│   │   ├── ingesta.ipynb      # Experimentación y desarrollo
│   │   ├── models/            # Modelos entrenados y artefactos
│   │   ├── mlartifacts/       # Artefactos MLflow
│   │   └── mlruns/            # Experimentos MLflow
│   │
│   └── test/               # Servicio de Predicción
│       ├── predict_api.py     # API REST Flask
│       ├── test_api.py        # Suite de pruebas
│       └── pyproject.toml     # Dependencias del servicio
│
├── pyproject.toml          # Configuración principal del proyecto
├── uv.lock                 # Lock file para reproducibilidad
└── README.md               # Documentación principal
```

## Stack Tecnológico

### **Machine Learning**

- **XGBoost**: Algoritmo principal de clasificación con hiperparámetros optimizados
- **Scikit-learn**: Preprocesamiento, métricas y validación
- **Pandas & NumPy**: Manipulación y análisis de datos
- **Matplotlib & Seaborn**: Visualización de datos y resultados

### **MLOps & Orquestación**

- **MLflow**: Seguimiento de experimentos, registro de modelos y gestión del ciclo de vida
- **Prefect**: Orquestación de workflows y gestión de tareas
- **UCI ML Repository**: Fuente de datos automatizada y reproducible

### **Deployment & Testing**

- **Flask**: Framework web para API REST
- **Requests**: Cliente HTTP para testing automatizado
- **Pytest**: Framework de testing (configurado)

### **Gestión de Dependencias**

- **UV**: Gestor moderno de dependencias Python
- **pyproject.toml**: Configuración estándar del proyecto

## Flujo de Trabajo MLOps

### **1. Exploración y Análisis (`notebooks/`)**

- **EDA Inicial**: Análisis exploratorio completo del dataset Adult
- **Visualizaciones**: Distribuciones, correlaciones y patrones
- **Feature Engineering**: Identificación de variables importantes

### **2. Pipeline de Entrenamiento (`src/app/train/`)**

#### **Características del Pipeline**:

- **Automatización Completa**: Desde descarga hasta modelo listo para producción
- **Orquestación con Prefect**: Gestión de tareas, reintentos y manejo de errores
- **Tracking con MLflow**: Registro de experimentos, métricas y artefactos
- **Reproducibilidad**: Semillas fijas y versionado completo

#### **Proceso Step-by-Step**:

1. **Descarga Automática**: Dataset Adult desde UCI ML Repository
2. **Preprocesamiento**:
   - Codificación de variables categóricas
   - Normalización de datos
   - Feature engineering automatizado
3. **División de Datos**: Train/Validation estratificado (60/40)
4. **Entrenamiento**: XGBoost con hiperparámetros optimizados via Optuna
5. **Evaluación**: Métricas completas (Accuracy, Precision, Recall, F1, ROC-AUC)
6. **Optimización**: Threshold óptimo para maximizar F1-score
7. **Persistencia**: Modelo completo + preprocesador + métricas

#### **Artefactos Generados**:

- `model_complete.bin`: Paquete completo del modelo
- `preprocessor.b`: Preprocesador para nuevos datos
- Visualizaciones: ROC, Precision-Recall, Confusion Matrix
- Registros MLflow: Experimentos completos con trazabilidad

### **3. Servicio de Predicción (`src/app/test/`)**

#### **API REST con Flask**:

- **Endpoints Disponibles**:
  - `GET /health`: Verificación del estado del sistema
  - `POST /predict`: Predicción individual con probabilidades
  - `GET /model_info`: Información detallada del modelo
  - `GET /example_request`: Formato de ejemplo para requests

#### **Características del Servicio**:

- **Carga Automática**: Modelo y preprocesador al inicio
- **Validación de Datos**: Verificación de features requeridas
- **Threshold Óptimo**: Aplicación automática del threshold calculado
- **Respuestas Estructuradas**: JSON con predicción, probabilidad y confianza
- **Manejo de Errores**: Respuestas HTTP apropiadas

#### **Testing Automatizado**:

- **Suite Completa**: Validación de todos los endpoints
- **Casos de Prueba**: Escenarios de ingresos altos y bajos
- **Logging Detallado**: Información completa de requests/responses
- **Validación de Formato**: Verificación de estructura de datos

## Métricas y Rendimiento

### **Modelo Actual (XGBoost Optimizado)**:

- **Accuracy**: ~83.5%
- **F1-Score**: ~71.2%
- **Precision**: ~61.2%
- **Recall**: ~85.1%
- **ROC-AUC**: ~92.9%

### **Características del Modelo**:

- **Balanceo de Clases**: Ajuste automático de pesos
- **Early Stopping**: Prevención de overfitting
- **Hiperparámetros Optimizados**: Via Optuna para máximo rendimiento
- **Threshold Dinámico**: Optimizado para F1-score

## Resumen

**ProyectoMLOps** es una implementación completa y profesional de un sistema de machine learning que abarca desde la exploración inicial de datos hasta el deployment de un servicio de predicción en producción.

El proyecto utiliza el dataset Adult para demostrar cómo construir, entrenar, evaluar y desplegar un modelo de clasificación de ingresos utilizando las mejores prácticas de MLOps, incluyendo orquestación con Prefect, tracking con MLflow, y deployment con Flask.

## Links Adicionales

[🔗 Presentación](https://gamma.app/docs/mzcwv5lcwdqcozh)

