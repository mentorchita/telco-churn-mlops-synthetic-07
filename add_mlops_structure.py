#!/usr/bin/env python3
"""
Скрипт для автоматичного додавання структури MLOps (модулі 2–4)
до репозиторію telco-churn-mlops-synthetic.

Запускати з кореня репозиторію:
    python add_mlops_structure.py
"""

import os
from pathlib import Path
import textwrap

# Коренева директорія проєкту (поточна директорія, де запущено скрипт)
ROOT = Path.cwd()

def create_dir(path: str):
    full_path = ROOT / path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"Створено директорію: {path}")

def create_file(path: str, content: str = ""):
    full_path = ROOT / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content))
    print(f"Створено файл: {path}")

def main():
    print("=== Початок додавання структури MLOps (модулі 2–4) ===\n")

    # ──────────────────────────────────────────────
    # Модуль 2: Version Control & Reproducibility
    # ──────────────────────────────────────────────

    # DVC
    create_dir("data/raw")
    create_dir("data/processed")
    create_dir("models")

    create_file("dvc.yaml", """\
        stages:
          generate:
            cmd: python src/generate_dataset_ext.py --output data/processed/churn_dataset.csv
            deps:
              - src/generate_dataset_ext.py
              - conf/config.yaml
            outs:
              - data/processed/churn_dataset.csv

          train:
            cmd: python pipelines/train.py
            deps:
              - data/processed/churn_dataset.csv
              - pipelines/train.py
              - conf/train.yaml
            outs:
              - models/churn_model.pkl

          predict:
            cmd: python pipelines/predict.py
            deps:
              - models/churn_model.pkl
              - data/processed/churn_dataset.csv
              - pipelines/predict.py
        """)

    create_file(".dvc/config", "# DVC remote config placeholder\n")

    # Hydra configs
    create_dir("conf")

    create_file("conf/config.yaml", """\
        defaults:
          - train: default

        data:
          path: data/processed/churn_dataset.csv
          test_size: 0.2

        model:
          type: xgboost
          params:
            n_estimators: 100
            max_depth: 5
            learning_rate: 0.1
        """)

    create_file("conf/train.yaml", "# Специфічні налаштування для тренування\n")
    create_file("conf/predict.yaml", "# Специфічні налаштування для передбачення\n")

    # ──────────────────────────────────────────────
    # Модуль 3: Experiment Tracking & Model Registry
    # ──────────────────────────────────────────────

    create_dir("mlflow")

    create_file("mlflow/mlflow_register.py", """\
        import mlflow

        # Приклад реєстрації моделі
        mlflow.set_tracking_uri("http://localhost:5000")  # або ваш сервер
        model_uri = "runs:/<run_id>/model"               # замінити на реальний
        mlflow.register_model(model_uri, "ChurnModel")
        print("Модель зареєстровано в MLflow Model Registry")
        """)

    # ──────────────────────────────────────────────
    # Модуль 4: Orchestration Pipelines
    # ──────────────────────────────────────────────

    create_dir("airflow/dags")
    create_dir("pipelines")

    create_file("airflow/dags/full_pipeline_dag.py", """\
        from airflow import DAG
        from airflow.operators.bash import BashOperator
        from datetime import datetime

        with DAG(
            dag_id="telco_churn_full_pipeline",
            start_date=datetime(2025, 1, 1),
            schedule_interval="@daily",
            catchup=False,
        ) as dag:

            generate = BashOperator(
                task_id="generate_data",
                bash_command="python src/generate_dataset_ext.py --output data/processed/churn_dataset.csv"
            )

            train = BashOperator(
                task_id="train_model",
                bash_command="python pipelines/train.py"
            )

            register = BashOperator(
                task_id="register_model",
                bash_command="python mlflow/mlflow_register.py"
            )

            generate >> train >> register
        """)

    create_file("pipelines/train.py", """\
        import mlflow
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier
        from hydra import compose, initialize
        from omegaconf import OmegaConf

        initialize(version_base=None, config_path="../conf")
        cfg = compose(config_name="config")

        with mlflow.start_run():
            df = pd.read_csv(cfg.data.path)
            X = df.drop("Churn", axis=1)
            y = df["Churn"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.data.test_size)

            model = XGBClassifier(**cfg.model.params)
            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.xgboost.log_model(model, "model")

            print(f"Accuracy: {accuracy:.4f}")
        """)

    create_file("pipelines/predict.py", """\
        import pickle
        import pandas as pd

        def predict(input_data_path: str, model_path: str = "models/churn_model.pkl"):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            df = pd.read_csv(input_data_path)
            # Приклад передбачення
            predictions = model.predict(df.drop("Churn", axis=1, errors="ignore"))
            print("Передбачені churn:", predictions)
        """)

    # Додаткові файли
    create_file("requirements-ml.txt", """\
        scikit-learn>=1.2.0
        xgboost>=1.7.0
        mlflow>=2.0.0
        hydra-core>=1.3.0
        pandas>=1.5.0
        numpy>=1.23.0
        """)

    print("\n=== Усі елементи успішно додано ===")
    print("Тепер можна запускати:")
    print("  make install-ml        # якщо додати в Makefile")
    print("  dvc repro              # відтворити пайплайн")
    print("  python pipelines/train.py")
    print("  airflow dags test full_pipeline_dag")

if __name__ == "__main__":
    main()