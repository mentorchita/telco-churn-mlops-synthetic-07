# ──────────────────────────────────────────────────────────────────────────────
# Makefile для проєкту telco-churn-mlops-synthetic
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help install dev install-dev generate generate-ext explore lint format clean clean-data docker-build docker-run docker-up down

# ──────────────────────────────────────────────────────────────────────────────
# Основні команди
# ──────────────────────────────────────────────────────────────────────────────

help: ## Показати цю довідку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Створити віртуальне середовище та встановити основні залежності
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt

dev: install-dev ## Встановити залежності для розробки (ruff, black, jupyter тощо)
install-dev:
	. venv/bin/activate && pip install -r requirements-dev.txt
	. venv/bin/activate && pre-commit install

generate: ## Згенерувати базовий датасет (оригінальний generate_dataset.py)
	. venv/bin/activate && python src/generate_dataset.py --samples 50000 --output data/telco_churn_demo.csv

generate-ext: ## Згенерувати розширений датасет (табличні + conversations + knowledge base)
	. venv/bin/activate && python src/generate_dataset_ext.py --samples 50000 --conv-samples 7500

explore: ## Запустити JupyterLab для дослідження даних
	. venv/bin/activate && jupyter lab notebooks/

lint: ## Перевірити код стилем (ruff + black --check)
	. venv/bin/activate && ruff check src/ notebooks/
	. venv/bin/activate && black --check src/ notebooks/

format: ## Автоматично відформатувати код (black + ruff --fix)
	. venv/bin/activate && black src/ notebooks/
	. venv/bin/activate && ruff check --fix src/ notebooks/

clean: ## Видалити тимчасові файли, venv, кеш
	rm -rf venv
	rm -rf __pycache__ *.pyc *.pyo .pytest_cache .ruff_cache
	rm -rf notebooks/.ipynb_checkpoints

clean-data: ## Видалити всі згенеровані дані
	rm -rf data/*.csv data/*.json

# ──────────────────────────────────────────────────────────────────────────────
# Docker команди (якщо використовуєте контейнеризацію)
# ──────────────────────────────────────────────────────────────────────────────

docker-build: ## Зібрати Docker-образ
	docker build -t telco-churn-generator:latest .

docker-run: ## Запустити генерацію всередині контейнера (розширена версія)
	docker run --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/config:/app/config \
		telco-churn-generator:latest \
		python src/generate_dataset_ext.py --samples 30000 --conv-samples 5000

docker-up: ## Запустити docker-compose (якщо є docker-compose.yml)
	docker compose up --build

docker-down: ## Зупинити та видалити контейнери
	docker compose down

# ──────────────────────────────────────────────────────────────────────────────
# Приклади використання з параметрами
# ──────────────────────────────────────────────────────────────────────────────

generate-small: ## Швидка генерація невеликого датасету для тестів
	. venv/bin/activate && python src/generate_dataset_ext.py --samples 10000 --conv-samples 1500

generate-demo: ## Генерація для демо на занятті (~30–50 тис. рядків)
	. venv/bin/activate && python src/generate_dataset_ext.py --samples 40000 --conv-samples 6000

# ──────────────────────────────────────────────────────────────────────────────
# Jupyter Notebook / Lab в Docker
# ──────────────────────────────────────────────────────────────────────────────

jupyter-up: ## Запустити JupyterLab у контейнері (порт 8888)
	docker compose up -d jupyter

jupyter-down: ## Зупинити Jupyter контейнер
	docker compose down jupyter

jupyter-logs: ## Показати логи Jupyter (корисний для отримання token)
	docker compose logs -f jupyter

jupyter-build: ## Перебудувати Jupyter (якщо змінили image або додали пакети)
	docker compose build jupyter

jupyter-bash: ## Зайти в bash всередину запущеного Jupyter контейнера
	docker compose exec jupyter bash

jupyter-clean: ## Видалити Jupyter контейнер та образ (якщо потрібно)
	docker compose down jupyter --rmi local
train:  ## Тренування моделі
	python pipelines/train.py

test:  ## Запуск тестів
	pytest tests/

deploy:  # Деплоймент
	docker compose up -d

install-api: ## Install API
	pip install -r requirements-api.txt

run-api: ## Run API
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build-api: ## Build API Docker image
	docker build -f Dockerfile.api -t churn-api:latest .

docker-run-api: ## Run API Docker container
	docker run -p 8000:8000 churn-api:latest

