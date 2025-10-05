#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional
from ucimlrepo import fetch_ucirepo

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

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
        
        # Combinar features y target en un solo DataFrame
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
    columnas_categoricas = ['workclass','education','marital-status','occupation','relationship','race','native-country']

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
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'binary:logistic',  # Para clasificación binaria
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred_proba = booster.predict(valid)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = (y_pred == y_val).mean()
        mlflow.log_metric("accuracy", accuracy)

        # Save preprocessor
        preprocessor_path = "models/preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)

        # Save complete model package for Flask API
        model_package = {
            'model' : booster,
            'preprocessor' : dv,
            'accuracy' : accuracy,
            'model_params' : best_params,
            'features_names': dv.feature_names_ if hasattr(dv, 'features_name_') else None
        }

        model_package_path = "models/model_complete.bin"
        with open(model_package_path, "wb") as f_out:
            pickle.dump(model_package, f_out)
        logger.info(f"Complete model package saved to {model_package_path}")
        
        try:
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
            mlflow.log_artifact(model_package_path, artifact_path="complete_model")
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
            ["Accuracy", f"{accuracy:.4f}"],
            ["Learning Rate", best_params['learning_rate']],
            ["Max Depth", best_params['max_depth']],
            ["Num Boost Rounds", 30],
            ["MLflow Run ID", run.info.run_id]
        ]

        create_table_artifact(
            key="model-performance",
            table=performance_data,
            description=f"Model performance metrics - Accuracy: {accuracy:.4f}"
        )

        # Create markdown artifact with training summary
        markdown_content = f"""
        # Model Training Summary

        ## Performance
        - **Accuracy**: {accuracy:.4f}
        - **MLflow Run ID**: {run.info.run_id}

        ## Parameters
        - Learning Rate: {best_params['learning_rate']}
        - Max Depth: {best_params['max_depth']}
        - Min Child Weight: {best_params['min_child_weight']}
        - Regularization Alpha: {best_params['reg_alpha']}
        - Regularization Lambda: {best_params['reg_lambda']}

        ## Training Details
        - Boost Rounds: 30
        - Early Stopping: 50 rounds
        - Objective: {best_params['objective']}
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
    df_train, df_val = train_test_split(df_processed, test_size=0.4, random_state=42)

    # Create features
    X_train, dv = create_features(df_train)
    X_val, _ = create_features(df_val, dv)

    # Prepare targets
    target = 'income'
    y_train = df_train[target].values
    y_val = df_val[target].values

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

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration using Prefect.')
    parser.add_argument('--mlflow-uri', type=str, help='MLflow tracking URI (overrides environment variable)')
    args = parser.parse_args()

    # Override MLflow URI if provided
    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
        setup_mlflow()

    try:
        # Run the flow
        run_id = income_classification_flow()
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 MLflow run_id: {run_id}")
        print(f"🔗 View results at: {mlflow.get_tracking_uri()}")

        # Save run ID for reference
        with open("prefect_run_id.txt", "w") as f:
            f.write(run_id)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise