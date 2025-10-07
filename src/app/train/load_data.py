#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional
from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)

import mlflow
from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration with fallback
def setup_mlflow():
    """Setup MLflow with proper error handling and fallback options."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # Test connection
        mlflow.search_experiments()
        logger.info(f"Connected to MLflow at: {mlflow_uri}")
    except Exception as e:
        logger.warning(f"Failed to connect to {mlflow_uri}: {e}")
        logger.info("Falling back to local SQLite database")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    try:
        mlflow.set_experiment("adult-income-experiment-prefect")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise

# Initialize MLflow
setup_mlflow()


@task(name="load_and_prepare_data", description="Load and prepare Adult dataset from UCI ML Repository", retries=3, retry_delay_seconds=2)
def load_and_prepare_data(dataset_id: int = 2) -> pd.DataFrame:
    """Descarga y prepara el dataset Adult desde UCI ML Repository."""
    logger_task = get_run_logger()
    
    logger_task.info(f"Descargando dataset Adult (ID: {dataset_id})")
    try:
        adult = fetch_ucirepo(id=dataset_id)
        X = adult.data.features
        y = adult.data.targets
        logger_task.info(f"Dataset cargado exitosamente. Shape: {X.shape}")
        
        # Combinar features y target en un solo DataFrame  (target -> 'income')
        if isinstance(y, pd.DataFrame):
            y_renamed = y.copy()
            if len(y.columns) == 1:
                y_renamed.columns = ["income"]
            else:
                y_renamed = y_renamed(columns={y.columns[0]:"income"})
        else:
            y_renamed = pd.DataFrame(y, columns=["income"])

        df = pd.concat([X, y_renamed], axis=1)
        
        return df
    except Exception as e:
        logger_task.error(f"Error al cargar el dataset: {e}")
        raise


@task(name="preprocess_data", description="Preprocess and encode categorical variables")
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Procesa y codifica las variables categóricas del dataset."""
    logger_task = get_run_logger()
    
    # Feature engineering
    mapeo_sex = {'Female': 1, 'Male': 0}
    df['sex'] = df['sex'].map(mapeo_sex)
    logger_task.info(f"Mapeo correcto de sexo")

    # Columnas categóricas a codificar
    columnas_categoricas = ['workclass','marital-status','occupation','relationship','native-country']

    # Inicializar codificadores
    encoders = {}
    for col in columnas_categoricas:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le  # Guardamos el encoder para posibles decodificaciones posteriores
    logger_task.info(f"Mapeo correcto de columnas categóricas")

    # Normaliza la etiqueta: quita espacios y punto final
    df['income_clean'] = (
        df['income']
        .astype(str)
        .str.strip()                 # quita espacios al inicio/fin
        .str.replace(r'\.$', '', regex=True)   # elimina punto final si existe
    )
    df['income_bin'] = df['income_clean'].map({'>50K': 1, '<=50K': 0})
    df = df.drop(columns=["income", "income_clean"])
    df = df.rename(columns={"income_bin": "income"})

    # Create artifact with data summary
    summary_data = [
        ["Total Records", len(df)],
        ["Average Age", f"{df['age'].mean():.2f} years"],
        ["Min Age", f"{df['age'].min():.2f} years"],
        ["Max Age", f"{df['age'].max():.2f} years"],
        ["Unique income combinations", df['income'].nunique()]
    ]

    create_table_artifact(
        key=f"data-summary-age",
        table=summary_data,
        description=f"Data summary for processed dataset"
    )

    return df

@task(name="create_features", description="Create feature matrix using DictVectorizer")
def create_features(df: pd.DataFrame, dv: Optional[DictVectorizer] = None) -> Tuple[any, DictVectorizer]:
    """
    Create feature matrix from DataFrame.

    Args:
        df: Input DataFrame
        dv: Pre-fitted DictVectorizer (optional)

    Returns:
        Tuple of (feature matrix, DictVectorizer)
    """
    logger_task = get_run_logger()
    
    # Usar todas las columnas excepto la variable objetivo como features
    feature_columns = [col for col in df.columns if col != 'income']
    
    # Ensure all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    dicts = df[feature_columns].to_dict(orient='records')
    logger_task.info(f"Created {len(dicts)} feature dictionaries with {len(feature_columns)} features")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)

        # Create artifact with feature info
        feature_info = [
            ["Total Features", X.shape[1]],
            ["Original Features", len(feature_columns)],
            ["Samples", X.shape[0]]
        ]

        create_table_artifact(
            key="feature-info",
            table=feature_info,
            description="Feature matrix information"
        )
    else:
        X = dv.transform(dicts)

    return X, dv


@task(name="train_model", description="Train XGBoost model with MLflow tracking")
def train_model(X_train, y_train, X_val, y_val, dv: DictVectorizer) -> str:
    """
    Train XGBoost model and log to MLflow.

    Entrena XGBoost (clasificación) y registra en MLflow:
    - parámetros, métricas de clasificación (acc/prec/recall/f1/roc_auc),
    - threshold óptimo por F1,
    - plots ROC/PR, matriz de confusión,
    - preprocesador y paquete del modelo.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        dv: Fitted DictVectorizer

    Returns:
        MLflow run ID
    """
    logger = get_run_logger()
    
    # Ensure models directory exists
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)
    
    logger.info(f"Training with {X_train.shape[0]} samples, {X_train.shape[1]} features")

    with mlflow.start_run() as run:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # ===== Mejores hiperparámetros (de Optuna) + binaria + balanceo =====
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        best_params = {
            # Optuna 
            'max_depth': 9,
            'learning_rate': 0.06927659202446733,
            'subsample': 0.7384018543749307,
            'colsample_bytree': 0.7456148148748957,
            'gamma': 3.131133258823818,
            # binaria + tracking estable
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': spw,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=333,
            evals=[(dval, "validation")],
            early_stopping_rounds=50
        )

        y_pred_proba = booster.predict(dval)
        y_pred = (y_pred_proba > 0.5).astype(int)

         # ===== Métricas de clasificación =====
        # ===== Métricas de clasificación =====
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc = roc_auc_score(y_val, y_pred_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc)

        # ===== Threshold óptimo para F1 =====
        thresholds = np.linspace(0.1, 0.9, 81)
        f1s = [f1_score(y_val, (y_pred_proba >= t).astype(int)) for t in thresholds]
        best_t = float(thresholds[int(np.argmax(f1s))])
        mlflow.log_metric("best_threshold_f1", best_t)

        # ===== Plots: ROC y PR, matriz de confusión =====
        # ROC
        RocCurveDisplay.from_predictions(y_val, y_pred_proba)
        plt.savefig(models_folder / "roc.png", bbox_inches="tight")
        plt.close()

        # PR
        PrecisionRecallDisplay.from_predictions(y_val, y_pred_proba)
        plt.savefig(models_folder / "pr.png", bbox_inches="tight")
        plt.close()

        # Matriz de confusión
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots()
        ax.imshow(cm, interpolation='nearest')
        ax.set_title('Confusion matrix')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, str(z), ha='center', va='center')
        plt.savefig(models_folder / "confusion_matrix.png", bbox_inches="tight")
        plt.close()

        # Save preprocessor
        preprocessor_path = "models/preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)

        # Save complete model package for Flask API
        model_package = {
            'model': booster,
            'preprocessor': dv,
            'threshold_f1': best_t,
            'metrics': {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc},
            'params': best_params,
            'features_names': dv.feature_names_ if hasattr(dv, 'features_name_') else None
        }

        model_package_path = "models/model_complete.bin"
        with open(model_package_path, "wb") as f_out:
            pickle.dump(model_package, f_out)
        logger.info(f"Complete model package saved to {model_package_path}")
        
        try:
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
            mlflow.log_artifact(model_package_path, artifact_path="complete_model")
            mlflow.log_artifact(str(models_folder / "roc.png"), artifact_path="plots")
            mlflow.log_artifact(str(models_folder / "pr.png"), artifact_path="plots")
            mlflow.log_artifact(str(models_folder / "confusion_matrix.png"), artifact_path="plots")

            # Log model
            input_example = X_train[:1]
            mlflow.xgboost.log_model(
                booster, 
                name="models_mlflow", 
                model_format="json", 
                input_example=input_example,
                conda_env={
                    'channels':['defaults'],
                    'dependencies':[
                        'python=3.9',
                        'pip',
                        {'pip':['xgboost', 'mlflow']}
                    ],
                    'name':'mlflow-env'
                }
            )
            logger.info("Successfully logged model and preprocessor to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
            logger.info("Model artifacts saved locally in models/ directory")

        # Create Prefect artifact with model performance
        performance_data = [
            ["Accuracy", f"{acc:.4f}"],
            ["Precision", f"{prec:.4f}"],
            ["Recall", f"{rec:.4f}"],
            ["F1", f"{f1:.4f}"],
            ["ROC AUC", f"{roc:.4f}"],
            ["Best threshold (F1)", f"{best_t:.3f}"],
            ["Boost Rounds", 333],
            ["Learning Rate", best_params['learning_rate']],
            ["Max Depth", best_params['max_depth']],
            ["Num Boost Rounds", 30],
            ["MLflow Run ID", run.info.run_id]
        ]

        create_table_artifact(
            key="model-performance",
            table=performance_data,
            description=f"Model performance metrics - Accuracy: {acc:.4f}"
        )

        # Create markdown artifact with training summary
        markdown_content = f"""
        # Model Training Summary

        ## Performance
        - **Accuracy**: {acc:.4f}
        - **MLflow Run ID**: {run.info.run_id}
        - Precision: **{prec:.4f}**
        - Recall: **{rec:.4f}**
        - F1: **{f1:.4f}**
        - ROC AUC: **{roc:.4f}**
        - Best threshold (F1): **{best_t:.3f}**

        ## Parameters
        {best_params}

        ## Training Details
        - Boost Rounds: 333
        - Early Stopping: 50 rounds
        - Objective: {best_params['objective']}
        - MLflow Run ID: {run.info.run_id}
        """

        create_markdown_artifact(
            key="training-summary",
            markdown=markdown_content,
            description="Detailed training summary"
        )

        return run.info.run_id


@flow(name="Adult Income Classification Pipeline", description="End-to-end ML pipeline for adult income classification")
def income_classification_flow() -> str:
    """
    Main flow for adult income classification.

    Returns:
        MLflow run ID
    """
    # Load and prepare data
    df = load_and_prepare_data()
    df_processed = preprocess_data(df)
    
    # Split data
    df_train, df_val = train_test_split(df_processed, test_size=0.4, random_state=42,
                                         stratify=df_processed["income"])

    # Create features
    X_train, dv = create_features(df_train)
    X_val, _ = create_features(df_val, dv)

    # Prepare targets
    target = 'income'
    y_train = df_train[target].values.astype(int)
    y_val = df_val[target].values.astype(int)

    # Train model
    run_id = train_model(X_train, y_train, X_val, y_val, dv)

    # Create final pipeline artifact
    pipeline_summary = f"""
    # Pipeline Execution Summary

    ## Data
    - **Training Samples**: {len(y_train):,}
    - **Validation Samples**: {len(y_val):,}

    ## Results
    - **MLflow Run ID**: {run_id}
    - **MLflow Experiment**: adult-income-experiment-prefect

    ## Next Steps
    1. Review model performance in MLflow UI: http://localhost:5000
    2. Compare with previous runs
    3. Consider model deployment if performance is satisfactory
    """

    create_markdown_artifact(
        key="pipeline-summary",
        markdown=pipeline_summary,
        description="Complete pipeline execution summary"
    )

    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict adult income with Prefect + MLflow.')
    parser.add_argument('--mlflow-uri', type=str, help='MLflow tracking URI (overrides environment variable)')
    args = parser.parse_args()

    # Override MLflow URI if provided
    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
        setup_mlflow()

    try:
        # Run the flow
        run_id = income_classification_flow()
        logger.info("\n✅ Pipeline completed successfully!")
        logger.info(f"📊 MLflow run_id: {run_id}")
        logger.info(f"🔗 View results at: {mlflow.get_tracking_uri()}")

        # Save run ID for reference
        with open("prefect_run_id.txt", "w") as f:
            f.write(run_id)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise