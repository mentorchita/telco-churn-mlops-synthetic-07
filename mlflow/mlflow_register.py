import mlflow

# Приклад реєстрації моделі
mlflow.set_tracking_uri("http://localhost:5000")  # або ваш сервер
model_uri = "runs:/<run_id>/model"               # замінити на реальний
mlflow.register_model(model_uri, "ChurnModel")
print("Модель зареєстровано в MLflow Model Registry")
