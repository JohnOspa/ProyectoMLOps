# Documentación - Pipeline de Entrenamiento (Adult Income)

Este directorio contiene el pipeline completo de entrenamiento para el modelo de clasificación de ingresos basado en el dataset Adult de UCI ML Repository. El pipeline está implementado usando **Prefect** para orquestación y **MLflow** para el seguimiento de experimentos.

## Descripción General

El sistema implementa un pipeline completo de machine learning que descarga, procesa y entrena un modelo XGBoost para predecir si una persona gana más de $50K al año basándose en características demográficas y laborales.

## Archivo Principal

### `load_data.py`

**Propósito**: Pipeline completo de entrenamiento que incluye descarga de datos, preprocesamiento, entrenamiento del modelo y registro de experimentos.

**Arquitectura del Pipeline**:

- **Orquestación**: Prefect para gestión de flujos y tareas
- **Seguimiento**: MLflow para logging de experimentos y modelos
- **Modelo**: XGBoost con hiperparámetros optimizados via Optuna
- **Métricas**: Accuracy, Precision, Recall, F1-score, ROC AUC

## Componentes del Pipeline

### **1. Configuración de MLflow**

```python
def setup_mlflow()
```

- **Funcionalidad**: Configura la conexión a MLflow con fallback automático
- **URI por defecto**: `sqlite:///mlflow.db` (base de datos local)
- **Experimento**: `adult-income-experiment-prefect`
- **Manejo de errores**: Fallback automático si no puede conectar al servidor remoto

### **2. Carga y Preparación de Datos**

```python
@task(name="load_and_prepare_data")
def load_and_prepare_data(dataset_id: int = 2) -> pd.DataFrame
```

- **Fuente**: UCI ML Repository (Adult dataset, ID: 2)
- **Funcionalidad**:
  - Descarga automática del dataset
  - Combinación de features y target
  - Renombrado de columnas para consistencia
- **Salida**: DataFrame con 14 features + target (`income`)

### **3. Preprocesamiento de Datos**

```python
@task(name="preprocess_data")
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame
```

- **Transformaciones aplicadas**:

  - **Codificación de género**: `{'Female': 1, 'Male': 0}`
  - **Label Encoding**: Para variables categóricas (`workclass`, `marital-status`, `occupation`, `relationship`, `native-country`)
  - **Limpieza del target**: Normalización de etiquetas de ingresos
  - **Binarización**: `'>50K' → 1`, `'<=50K' → 0`
- **Variables categóricas procesadas**:

  - `workclass`: Tipo de empleo
  - `marital-status`: Estado civil
  - `occupation`: Ocupación
  - `relationship`: Relación familiar
  - `native-country`: País de origen

### **4. Creación de Features**

```python
@task(name="create_features")
def create_features(df: pd.DataFrame, dv: Optional[DictVectorizer] = None)
```

- **Método**: DictVectorizer de scikit-learn
- **Funcionalidad**:
  - Conversión a matriz de features sparse
  - Manejo automático de variables categóricas
  - Reutilización del vectorizador para datos nuevos
- **Salida**: Matriz de features + DictVectorizer entrenado

### **5. Entrenamiento del Modelo**

```python
@task(name="train_model")
def train_model(X_train, y_train, X_val, y_val, dv: DictVectorizer)
```

#### **Configuración del Modelo**:

- **Algoritmo**: XGBoost (binary:logistic)
- **Hiperparámetros optimizados** (via Optuna):
  - `max_depth`: 9
  - `learning_rate`: 0.069
  - `subsample`: 0.738
  - `colsample_bytree`: 0.746
  - `gamma`: 3.131
  - `num_boost_round`: 333
  - `early_stopping_rounds`: 50

Esta configuración puede ser actualizada automáticamente, para que el modelo se entrene con nuevos datos, permitiendo una adaptación continua y mejorando su rendimiento.

#### **Balanceo de Clases**:

- `scale_pos_weight`: Calculado automáticamente basado en la distribución de clases
- Formula: `n_neg / n_pos`

#### **Métricas Calculadas**:

- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión para clase positiva (>50K)
- **Recall**: Sensibilidad para clase positiva
- **F1-score**: Media armónica de precision y recall
- **ROC AUC**: Área bajo la curva ROC

#### **Optimización de Threshold**:

- **Método**: Grid search en rango 0.1 a 0.9
- **Criterio**: Maximización del F1-score
- **Uso**: Aplicado en predicciones para mejorar rendimiento

#### **Visualizaciones Generadas**:

1. **Curva ROC** (`roc.png`)
2. **Curva Precision-Recall** (`pr.png`)
3. **Matriz de Confusión** (`confusion_matrix.png`)

#### **Artefactos Guardados**:

1. **`preprocessor.b`**: DictVectorizer serializado
2. **`model_complete.bin`**: Paquete completo del modelo incluyendo:
   - Modelo XGBoost entrenado
   - Preprocesador
   - Threshold óptimo
   - Métricas de rendimiento
   - Parámetros del modelo
   - Nombres de features

### **6. Flow Principal**

```python
@flow(name="Adult Income Classification Pipeline")
def income_classification_flow() -> str
```

- **Orquestación**: Ejecuta todas las tareas en secuencia
- **División de datos**: 60% entrenamiento, 40% validación
- **Estratificación**: Mantiene distribución de clases
- **Artefactos de Prefect**: Resúmenes y métricas visuales

## Salidas del Pipeline

### **Archivos Locales Generados**:

```
models/
├── model_complete.bin          # Modelo completo para API
├── preprocessor.b              # Preprocesador serializado
├── roc.png                     # Curva ROC
├── pr.png                      # Curva Precision-Recall
└── confusion_matrix.png        # Matriz de confusión

prefect_run_id.txt             # ID de ejecución para referencia
mlflow.db                      # Base de datos MLflow local
```

### **Registros en MLflow**:

- **Parámetros**: Hiperparámetros del modelo
- **Métricas**: Accuracy, Precision, Recall, F1, ROC AUC, Threshold
- **Artefactos**: Modelo, preprocesador, gráficos
- **Modelo registrado**: En formato MLflow para deployment

### **Artefactos de Prefect**:

- **Resumen de datos**: Estadísticas del dataset
- **Información de features**: Dimensiones de la matriz
- **Rendimiento del modelo**: Métricas detalladas
- **Resumen del pipeline**: Información completa de ejecución

## Monitoring y Logging

### **Logs del Sistema**:

- Información de conexión a MLflow
- Progreso de descarga de datos
- Estadísticas de preprocesamiento
- Métricas de entrenamiento
- Estados de guardado de artefactos

### **Prefect UI**:

- Visualización del flujo de tareas
- Estado de ejecución
- Artefactos generados
- Tiempo de ejecución

## Configuración del Entorno

### **Dependencias Requeridas**:

```python
# Principales
ucimlrepo          # Descarga de datasets UCI
pandas, numpy      # Manipulación de datos
scikit-learn       # Preprocesamiento y métricas
xgboost           # Modelo de machine learning
matplotlib        # Visualizaciones

# Orquestación y Tracking
prefect           # Gestión de workflows
mlflow            # Seguimiento de experimentos
```

### **Estructura de Directorios**:

```
train/
├── load_data.py              # Pipeline principal
├── models/                   # Modelos y artefactos
├── mlflow.db                # Base de datos MLflow
├── mlartifacts/             # Artefactos MLflow
├── mlruns/                  # Experimentos MLflow
└── prefect_run_id.txt       # ID de ejecución
```

## Características Avanzadas

### **Tolerancia a Fallos**:

- **Reintentos**: 3 intentos automáticos para descarga de datos
- **Delay**: 2 segundos entre reintentos
- **Fallback MLflow**: Cambio automático a SQLite local

### **Reproducibilidad**:

- **Semilla fija**: `seed=42` para resultados consistentes
- **Versionado**: Registro completo en MLflow
- **Trazabilidad**: IDs únicos para cada ejecución

### **Escalabilidad**:

- **Sparse matrices**: Eficiencia de memoria con DictVectorizer
- **Early stopping**: Prevención de overfitting
- **Modularidad**: Tareas independientes y reutilizables

## Próximos Pasos

1. **Monitoreo**: Implementar alertas de drift de datos
2. **Automatización**: Configurar ejecución programada
3. **Optimización**: Experimentar con otros algoritmos
4. **Deployment**: Integrar con sistemas de producción
5. **A/B Testing**: Comparar versiones de modelos en producción
