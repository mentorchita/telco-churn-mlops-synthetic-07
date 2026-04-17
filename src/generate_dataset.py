import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

def generate_telco_dataset_with_drift(
    n_samples: int = 50000,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    output_file: str = "synthetic_telco_churn_with_drift.csv"
):
    data = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days

    for i in range(n_samples):
        # Випадкова дата в діапазоні
        record_date = start + timedelta(days=random.randint(0, total_days))
        progress = (record_date - start).days / total_days  # 0.0 → 1.0 (2023 → 2024)

        # === ДРЕЙФ ПАРАМЕТРІВ ===
        # 1. Зростання Fiber optic
        fiber_prob = 0.40 + 0.25 * progress
        dsl_prob = 0.40 - 0.20 * progress
        no_inet_prob = 0.20 - 0.05 * progress

        # 2. Зменшення Electronic check
        echeck_prob = max(0.15, 0.40 - 0.25 * progress)

        # 3. Зростання довгострокових контрактів
        m2m_prob = max(0.30, 0.55 - 0.25 * progress)

        # 4. Зростання стримінгу
        streaming_boost = 0.3 * progress

        # 5. Зменшення SeniorCitizen серед нових клієнтів
        senior_prob = max(0.08, 0.18 - 0.12 * progress)

        # === Генерація клієнта ===
        gender = random.choice(["Male", "Female"])
        senior_citizen = 1 if random.random() < senior_prob else 0
        has_partner = random.choices(["Yes", "No"], weights=[52 + 10*progress, 48 - 10*progress])[0]
        has_dependents = "Yes" if random.random() < (0.3 - 0.1*progress) else "No"

        tenure = int(np.random.beta(2 + progress, 3 - 0.5*progress) * 72)
        tenure = max(0, min(tenure, 72))

        phone_service = "Yes" if random.random() < 0.92 else "No"
        internet_service = random.choices(
            ["DSL", "Fiber optic", "No"],
            weights=[dsl_prob, fiber_prob, no_inet_prob]
        )[0]

        # Додаткові послуги
        if internet_service == "No":
            secs = ["No internet service"] * 6
            online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies = secs
        else:
            base_yes = 0.5 + streaming_boost
            online_security = "Yes" if random.random() < (base_yes * 0.7) else "No"
            online_backup = "Yes" if random.random() < (base_yes * 0.8) else "No"
            device_protection = "Yes" if random.random() < (base_yes * 0.75) else "No"
            tech_support = "Yes" if random.random() < (base_yes * 0.6) else "No"
            streaming_tv = "Yes" if random.random() < (base_yes + 0.1) else "No"
            streaming_movies = "Yes" if random.random() < (base_yes + 0.1) else "No"

        multiple_lines = "No phone service" if phone_service == "No" else (
            "Yes" if random.random() < 0.45 + 0.1*progress else "No"
        )

        # Контракт
        contract = random.choices(
            ["Month-to-month", "One year", "Two year"],
            weights=[m2m_prob, (1-m2m_prob)*0.6, (1-m2m_prob)*0.4]
        )[0]

        paperless_billing = "Yes" if random.random() < 0.59 + 0.15*progress else "No"

        payment_method = random.choices(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            weights=[echeck_prob, 0.25, 0.25 + 0.1*progress, 0.25 + 0.15*progress]
        )[0]

        # Ціна
        base = 20.0
        if phone_service == "Yes":
            base += 25
            if multiple_lines == "Yes":
                base += 18
        if internet_service == "DSL":
            base += 50
        elif internet_service == "Fiber optic":
            base += 82 + 10*progress  # дорожче з часом

        extra_count = sum([online_security=="Yes", online_backup=="Yes", device_protection=="Yes",
                          tech_support=="Yes", streaming_tv=="Yes", streaming_movies=="Yes"])
        base += extra_count * (8 + 3*progress)

        if contract == "One year":
            base *= 0.94
        elif contract == "Two year":
            base *= 0.88 - 0.03*progress  # знижки трохи зменшуються

        monthly_charges = round(max(18.5, base + np.random.normal(0, 6)), 2)
        total_charges = round(monthly_charges * tenure * random.uniform(0.97, 1.03), 2)

        # Churn — знижується з часом (компанія покращує сервіс)
        churn_base = 0.45
        if contract == "Month-to-month": churn_base += 0.35
        if payment_method == "Electronic check": churn_base += 0.18
        if internet_service == "Fiber optic": churn_base += 0.08
        if tenure < 12: churn_base += 0.25 - tenure*0.02
        churn_base -= 0.20 * progress  # головний ефект покращення

        churn = "Yes" if random.random() < churn_base else "No"

        customer_id = f"{random.randint(1000,9999)}-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}"

        row = [customer_id, gender, senior_citizen, has_partner, has_dependents, tenure,
               phone_service, multiple_lines, internet_service, online_security, online_backup,
               device_protection, tech_support, streaming_tv, streaming_movies, contract,
               paperless_billing, payment_method, monthly_charges, total_charges, churn,
               record_date.strftime("%Y-%m-%d")]

        data.append(row)

    columns = ["customerID","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
               "MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
               "TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling",
               "PaymentMethod","MonthlyCharges","TotalCharges","Churn","RecordDate"]

    df = pd.DataFrame(data, columns=columns)
    df = df.sort_values("RecordDate").reset_index(drop=True)

    df.to_csv(output_file, index=False)
    print(f"Готово! Згенеровано {n_samples:,} записів з дрейфом за 2023–2024")
    print(f"Файл: {output_file}")
    print("\nРозподіл Churn по роках:")
    df['Year'] = pd.to_datetime(df['RecordDate']).dt.year
    print(df.groupby('Year')['Churn'].value_counts(normalize=True).unstack().round(3))

# === ЗАПУСК ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic Telco Churn dataset with drift")
    parser.add_argument("--samples", type=int, default=100000, help="Number of samples (default: 100000)")
    parser.add_argument("--output", type=str, default="data/telco_churn_full.csv", help="Output CSV path")
    args = parser.parse_args()

    generate_telco_dataset_with_drift(
        n_samples=args.samples,
        output_file=args.output
    )
