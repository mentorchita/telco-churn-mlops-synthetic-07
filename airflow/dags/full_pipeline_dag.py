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
