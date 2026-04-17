FROM python:3.11-slim

WORKDIR /app

# Копіюємо тільки необхідне (краще для кешу)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/          
COPY .gitignore .               

# Якщо хочете запускати за замовчуванням саме розширений скрипт
CMD ["python", "src/generate_dataset_ext.py", "--samples", "50000", "--conv-samples", "7500"]