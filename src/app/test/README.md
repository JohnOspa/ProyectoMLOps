# Documentación - Servicio de Predicción de Ingresos (Adult Dataset)

Este directorio contiene los archivos necesarios para desplegar y probar un servicio de predicción de ingresos basado en el dataset Adult utilizando un modelo XGBoost entrenado previamente.

## Descripción General

El servicio implementa una API REST que predice si una persona gana más de $50K al año basándose en características demográficas y laborales. El modelo utiliza un threshold óptimo calculado durante el entrenamiento para maximizar el F1-score.

## Archivos del Proyecto

### `pyproject.toml`

**Propósito**: Archivo de configuración del proyecto que define las dependencias y metadatos necesarios para el servicio de predicción.

**Contenido**:

- **Dependencias principales**:
  - `flask`: Framework web para crear la API REST
  - `xgboost`: Biblioteca de machine learning para cargar y ejecutar el modelo
  - `numpy` y `pandas`: Procesamiento de datos
  - `scikit-learn`: Preprocesamiento de características
  - `requests`: Cliente HTTP para testing

### `predict_api.py`

**Propósito**: Servidor Flask que expone una API REST para realizar predicciones de ingresos utilizando el modelo XGBoost entrenado.

**Características principales**:

- **Carga del modelo**: Carga automática del modelo completo (.bin) al iniciar el servidor
- **Preprocesamiento**: Aplica las mismas transformaciones utilizadas durante el entrenamiento
- **Threshold óptimo**: Utiliza el threshold calculado durante el entrenamiento para maximizar F1-score
- **Validación de datos**: Verifica que todas las características requeridas estén presentes

**Endpoints disponibles**:

1. **`GET /health`** - Verificación del estado del servicio

   - Retorna métricas del modelo y estado del sistema
2. **`POST /predict`** - Realizar predicción

   - Requiere JSON con las 14 características del dataset Adult
   - Retorna predicción, probabilidad, categoría de ingresos y confianza
3. **`GET /model_info`** - Información del modelo

   - Retorna métricas, parámetros y threshold del modelo
4. **`GET /example_request`** - Formato de ejemplo para predicciones

   - Retorna un ejemplo de JSON válido para hacer predicciones

**Características requeridas para predicción**:

```json
{
  "age": 35,
  "workclass": "Private",
  "fnlwgt": 280464,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 5178,
  "capital-loss": 0,
  "hours-per-week": 45,
  "native-country": "United-States"
}
```

El servidor estará disponible en: http://localhost:9696

### `test_api.py`

**Propósito**: Archivo de pruebas completa para validar el funcionamiento correcto de todos los endpoints de la API.

**Funcionalidades de testing**:

1. **`test_health()`** - Prueba el endpoint de salud

   - Verifica que el modelo esté cargado correctamente
   - Valida que las métricas del modelo estén disponibles
2. **`test_model_info()`** - Prueba el endpoint de información del modelo

   - Verifica acceso a métricas, parámetros y threshold
3. **`test_example_request()`** - Obtiene formato de ejemplo

   - Valida que el endpoint retorne un ejemplo válido
   - Utiliza el ejemplo para pruebas posteriores
4. **`test_prediction(data)`** - Prueba predicciones con datos específicos

   - Valida el formato de respuesta
   - Verifica que las predicciones sean coherentes
   - Prueba con casos de ingresos altos y bajos

**Casos de prueba incluidos**:

- **Ejemplo automático**: Utiliza el ejemplo proporcionado por la API
- **Ingreso alto**: Persona con características asociadas a ingresos >$50K
- **Ingreso bajo**: Persona con características asociadas a ingresos ≤$50K

## Manejo de Errores

La API maneja los siguientes tipos de errores:

- **500**: Modelo no cargado
- **400**: Datos faltantes o formato incorrecto
- **400**: Características requeridas ausentes

## Logs y Monitoreo

El sistema incluye logging detallado que registra:

- Carga del modelo y métricas
- Errores de predicción
- Información de requests
- Estado del sistema

## Consideraciones de Seguridad

- El servicio está configurado para desarrollo (debug=True)
- Para producción, desactivar el modo debug
- Implementar autenticación si es necesario
- Validar y sanitizar todas las entradas

## Próximos Pasos

- Implementar caché de modelo para mejor rendimiento
- Añadir más validaciones de entrada
- Integrar con sistemas de monitoreo
- Implementar versionado de modelos
