import pandas as pd
import numpy as np
import random
import json
import yaml
from faker import Faker
from datetime import datetime, timedelta
from pathlib import Path
import argparse

fake = Faker()
random.seed(42)
np.random.seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# COMPLAINT_TEMPLATES та RESOLUTION_TEMPLATES (повністю відновлені)
# ──────────────────────────────────────────────────────────────────────────────

COMPLAINT_TEMPLATES = {
    "billing_high": [
        "My bill is too high this month. I was charged ${amount} but my usual is around ${normal}. Can you explain?",
        "I'm shocked by my ${amount} bill! This is way more than my typical ${normal}. What happened?",
        "Why am I being charged ${amount}? My contract says ${normal}/month. Please fix this immediately.",
        "I noticed an unexpected increase in my bill to ${amount}. Last month it was ${normal}. What's the reason?",
    ],
    "service_slow": [
        "My internet has been incredibly slow for the past {days} days. I'm paying for {speed} but getting terrible speeds.",
        "The {service} service keeps buffering. This is unacceptable for what I'm paying.",
        "I can't work from home because the internet is so slow. When will this be fixed?",
        "Download/upload speeds are way below what I'm paying for. Can you check my line?",
    ],
    "service_outage": [
        "My {service} has been down since {time}. I need this fixed ASAP as I work from home.",
        "Complete service outage for {hours} hours now. No internet, no phone. What's going on?",
        "Still no {service} after {days} days! I'm paying for a service I'm not receiving.",
        "Internet is completely out in my area — is there an outage?",
    ],
    "contract_confusion": [
        "I thought I signed up for a {contract} contract, but my bill says {actual_contract}. Please clarify.",
        "I want to cancel but you're saying I have a contract until {date}. I was told it was month-to-month!",
        "Your sales rep promised me {feature} with my {contract} plan but I don't see it on my account.",
        "The contract terms on my account don't match what I agreed to. Can you review?",
    ],
    "want_to_cancel": [
        "I want to cancel my service. It's too expensive and I found a better deal elsewhere.",
        "Please cancel my account. I'm moving to a competitor who offers {feature} for less money.",
        "I've been a customer for {tenure} months but the service quality has declined. I'm leaving.",
        "I'm not satisfied anymore — please process my cancellation request.",
    ]
}

RESOLUTION_TEMPLATES = {
    "billing_high": [
        "I apologize for the billing confusion. I see there was a one-time charge for {reason}. I've applied a ${credit} credit to your account.",
        "You're right, that charge was incorrect. I've adjusted your bill back to ${normal} and credited the difference.",
        "Let me explain: the extra ${diff} was for {reason}. I can waive this charge as a one-time courtesy.",
        "I've reviewed your account and removed the unauthorized fee of ${diff}. Your next bill will reflect the correction.",
    ],
    "service_slow": [
        "I'm sorry about the speed issues. I've scheduled a technician visit for {date}. In the meantime, try resetting your router.",
        "I see there's network congestion in your area. We're upgrading infrastructure. I've applied a ${credit} credit for the inconvenience.",
        "I've run diagnostics and found an issue with your modem. We'll ship a new one overnight at no charge.",
        "Your line has been reprovisioned. Speeds should improve within the next 2 hours. I've credited ${credit}.",
    ],
    "service_outage": [
        "There's a known outage in your area due to {reason}. Estimated restoration time is {time}. I've credited your account for the downtime.",
        "I apologize for the disruption. The issue has been identified and our team is working on it. ETA: {time}.",
        "The outage was caused by {reason}. Service is now restored. I've applied a ${credit} credit to your next bill.",
        "Outage resolved — fiber splice repaired. Thank you for your patience. Credit of ${credit} applied.",
    ],
    "contract_confusion": [
        "I see the confusion. Your contract is actually {contract_type}. I've updated your account notes and confirmed your terms.",
        "You're correct - there was an error in how your contract was entered. I've corrected it to {contract_type} as agreed.",
        "I apologize for the miscommunication. Let me clarify your current contract terms: {details}.",
        "I've adjusted your plan to match the original agreement. No early termination fee will apply.",
    ],
    "want_to_cancel": [
        "I'm sorry to hear you want to leave. Before you go, let me offer you {offer} to stay. Would that work for you?",
        "I understand your frustration. I can offer you a special retention discount: {discount}% off for the next {months} months.",
        "I'd hate to see you go after {tenure} months. How about we upgrade you to our {plan} plan at your current price?",
        "As a valued customer, I'd like to offer you one free month + {discount}% off for 12 months if you stay.",
    ]
}


def load_config(config_path: str = "config/config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"Файл конфігурації {config_path} не знайдено → використовуємо значення за замовчуванням")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def generate_tabular_data(config: dict = None) -> pd.DataFrame:
    config = config or {}
    gen = config.get("generation", {})
    drift = config.get("drift", {})

    n_samples   = gen.get("samples", 50000)
    start_date  = gen.get("start_date", "2023-01-01")
    end_date    = gen.get("end_date", "2024-12-31")

    # Drift параметри (fallback на оригінальні значення)
    fiber_growth_rate      = drift.get("fiber_growth_rate", 0.25)
    dsl_decline_rate       = drift.get("dsl_decline_rate", 0.20)
    no_inet_decline        = drift.get("no_internet_decline", 0.05)
    echeck_decline_rate    = drift.get("echeck_decline_rate", 0.25)
    m2m_decline_rate       = drift.get("m2m_decline_rate", 0.25)
    streaming_boost_factor = drift.get("streaming_boost_factor", 0.3)
    senior_decline_rate    = drift.get("senior_decline_rate", 0.12)
    churn_base_decline     = drift.get("churn_base_decline", 0.20)

    data = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days

    for _ in range(n_samples):
        record_date = start + timedelta(days=random.randint(0, total_days))
        progress = (record_date - start).days / total_days

        fiber_prob = 0.40 + fiber_growth_rate * progress
        dsl_prob = 0.40 - dsl_decline_rate * progress
        no_inet_prob = 0.20 - no_inet_decline * progress

        echeck_prob = max(0.15, 0.40 - echeck_decline_rate * progress)
        m2m_prob = max(0.30, 0.55 - m2m_decline_rate * progress)
        streaming_boost = streaming_boost_factor * progress
        senior_prob = max(0.08, 0.18 - senior_decline_rate * progress)

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

        if internet_service == "No":
            secs = ["No internet service"] * 6
            online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies = secs
        else:
            base_yes = 0.5 + streaming_boost
            online_security   = "Yes" if random.random() < (base_yes * 0.7)  else "No"
            online_backup     = "Yes" if random.random() < (base_yes * 0.8)  else "No"
            device_protection = "Yes" if random.random() < (base_yes * 0.75) else "No"
            tech_support      = "Yes" if random.random() < (base_yes * 0.6)  else "No"
            streaming_tv      = "Yes" if random.random() < (base_yes + 0.1) else "No"
            streaming_movies  = "Yes" if random.random() < (base_yes + 0.1) else "No"

        multiple_lines = "No phone service" if phone_service == "No" else (
            "Yes" if random.random() < 0.45 + 0.1*progress else "No"
        )

        contract = random.choices(
            ["Month-to-month", "One year", "Two year"],
            weights=[m2m_prob, (1-m2m_prob)*0.6, (1-m2m_prob)*0.4]
        )[0]

        paperless_billing = "Yes" if random.random() < 0.59 + 0.15*progress else "No"

        payment_method = random.choices(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            weights=[echeck_prob, 0.25, 0.25 + 0.1*progress, 0.25 + 0.15*progress]
        )[0]

        base = 20.0
        if phone_service == "Yes":
            base += 25
            if multiple_lines == "Yes":
                base += 18
        if internet_service == "DSL":
            base += 50
        elif internet_service == "Fiber optic":
            base += 82 + 10*progress

        extra_count = sum([online_security=="Yes", online_backup=="Yes", device_protection=="Yes",
                           tech_support=="Yes", streaming_tv=="Yes", streaming_movies=="Yes"])
        base += extra_count * (8 + 3*progress)

        if contract == "One year":
            base *= 0.94
        elif contract == "Two year":
            base *= 0.88 - 0.03*progress

        monthly_charges = round(max(18.5, base + np.random.normal(0, 6)), 2)
        total_charges = round(monthly_charges * tenure * random.uniform(0.97, 1.03), 2)

        churn_base = 0.45
        if contract == "Month-to-month": churn_base += 0.35
        if payment_method == "Electronic check": churn_base += 0.18
        if internet_service == "Fiber optic": churn_base += 0.08
        if tenure < 12: churn_base += 0.25 - tenure*0.02
        churn_base -= churn_base_decline * progress

        churn = "Yes" if random.random() < churn_base else "No"

        customer_id = f"{random.randint(1000,9999)}-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}"

        row = {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": has_partner,
            "Dependents": has_dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
            "RecordDate": record_date.strftime("%Y-%m-%d")
        }
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values("RecordDate").reset_index(drop=True)
    return df


def generate_conversation(customer: dict) -> dict:
    issue_type = random.choice(list(COMPLAINT_TEMPLATES.keys()))
    complaint_template = random.choice(COMPLAINT_TEMPLATES[issue_type])

    if issue_type == "billing_high":
        normal = round(customer['MonthlyCharges'] * random.uniform(0.7, 0.85), 2)
        complaint = complaint_template.format(
            amount=customer['MonthlyCharges'],
            normal=normal
        )
    elif issue_type == "service_slow":
        speed_label = "fiber optic speeds" if customer['InternetService'] == "Fiber optic" else "DSL speeds"
        complaint = complaint_template.format(
            days=random.randint(2, 14),
            speed=speed_label,
            service=customer['InternetService'].lower()
        )
    elif issue_type == "service_outage":
        complaint = complaint_template.format(
            service=customer['InternetService'],
            time=random.choice(["this morning", "yesterday morning", "last night", "2 days ago"]),
            hours=random.randint(4, 72),
            days=random.randint(1, 7)
        )
    elif issue_type == "contract_confusion":
        complaint = complaint_template.format(
            contract=customer['Contract'].lower(),
            actual_contract=random.choice(["Month-to-month", "One year", "Two year"]),
            date=(datetime.now() + timedelta(days=random.randint(30, 730))).strftime("%B %d, %Y"),
            feature=random.choice(["free installation", "premium tech support", "streaming bundle"])
        )
    elif issue_type == "want_to_cancel":
        complaint = complaint_template.format(
            tenure=customer['tenure'],
            feature=random.choice(["faster internet", "better support", "lower monthly price"])
        )
    else:
        complaint = complaint_template

    # Resolution
    resolution_template = random.choice(RESOLUTION_TEMPLATES[issue_type])
    if issue_type == "billing_high":
        diff = round(customer['MonthlyCharges'] * random.uniform(0.15, 0.35), 2)
        resolution = resolution_template.format(
            reason=random.choice(["late fee", "equipment rental", "one-time upgrade charge"]),
            credit=diff,
            normal=round(customer['MonthlyCharges'] - diff, 2),
            diff=diff
        )
    elif issue_type == "service_slow":
        resolution = resolution_template.format(
            date=(datetime.now() + timedelta(days=random.randint(1, 7))).strftime("%B %d"),
            credit=random.choice([10, 15, 20, 25, 30])
        )
    elif issue_type == "service_outage":
        resolution = resolution_template.format(
            reason=random.choice(["fiber line damage", "power outage in the area", "equipment failure", "scheduled upgrade"]),
            time=random.choice(["within 4 hours", "by end of day", "within 24 hours", "by tomorrow morning"]),
            credit=random.choice([15, 20, 25, 30, 50])
        )
    elif issue_type == "contract_confusion":
        resolution = resolution_template.format(
            contract_type=customer['Contract'],
            details=f"{customer['Contract']} with auto-renewal, cancel anytime after term with 30 days notice"
        )
    elif issue_type == "want_to_cancel":
        resolution = resolution_template.format(
            offer=random.choice(["15% discount for 12 months", "free upgrade to Fiber", "one month free"]),
            discount=random.choice([10, 15, 20, 25]),
            months=random.choice([6, 12]),
            tenure=customer['tenure'],
            plan="Premium Fiber 1 Gbps"
        )
    else:
        resolution = resolution_template

    return {
        "customerID": customer["customerID"],
        "issue_type": issue_type,
        "complaint": complaint,
        "resolution": resolution,
        "RecordDate": customer["RecordDate"]
    }


def generate_knowledge_base(output_dir: str | Path):
    kb_data = [
        {"id": 1, "title": "How to reset your modem", "content": "1. Unplug the power cord from the modem. 2. Wait 30 seconds. 3. Plug it back in. 4. Wait for all lights to stabilize."},
        {"id": 2, "title": "Understanding your bill", "content": "Your monthly bill includes: base plan charge, equipment rental (if applicable), taxes, and any one-time fees. Check 'My Account' for detailed breakdown."},
        {"id": 3, "title": "Upgrading to Fiber optic", "content": "Fiber offers speeds up to 1 Gbps. Availability depends on your address. Contact support or check online to see if eligible."},
        {"id": 4, "title": "How to change payment method", "content": "Log in → My Account → Billing & Payments → Update Payment Method. We accept credit/debit cards, bank transfer, and electronic check."},
        {"id": 5, "title": "Troubleshooting slow internet", "content": "1. Restart modem/router. 2. Connect via Ethernet to test. 3. Check for background downloads. 4. Contact us if issue persists."},
        {"id": 6, "title": "Contract terms and cancellation", "content": "Month-to-month: cancel anytime. One/Two year: early termination fee may apply. 30-day notice required."},
        {"id": 7, "title": "Adding streaming services", "content": "You can add HBO, Netflix bundle, etc. in My Services. Some plans include free streaming options."},
        {"id": 8, "title": "Technical support hours", "content": "24/7 phone support. Chat available Mon–Fri 8 AM – 10 PM, weekends 9 AM – 8 PM."}
    ]

    output_dir = Path(output_dir)
    csv_path = output_dir / "knowledge_base.csv"
    json_path = output_dir / "knowledge_base.json"

    pd.DataFrame(kb_data).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)

    print(f"Knowledge base збережено: {csv_path} та {json_path} ({len(kb_data)} документів)")


# ──────────────────────────────────────────────────────────────────────────────
# Головний запуск
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерація розширеного Telco датасету: churn + support conversations + knowledge base")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Шлях до config.yaml (опціонально)")
    parser.add_argument("--samples", type=int, help="Кількість клієнтів (перевизначення)")
    parser.add_argument("--conv-samples", type=int, help="Кількість розмов support (перевизначення)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Директорія для збереження файлів")
    args = parser.parse_args()

    config = load_config(args.config)

    # Пріоритет: CLI > config.yaml > дефолт
    n_samples    = args.samples    or config.get("generation", {}).get("samples", 50000)
    conv_samples = args.conv_samples or config.get("generation", {}).get("conv_samples", 7500)
    output_dir   = args.output_dir or config.get("generation", {}).get("output_dir", "data")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Генерація: {n_samples:,} клієнтів + {conv_samples:,} розмов → {output_path}")

    # 1. Табличні дані
    df_customers = generate_tabular_data(config)
    customers_path = output_path / "telco_customers.csv"
    df_customers.to_csv(customers_path, index=False)
    print(f"Збережено {len(df_customers):,} клієнтів → {customers_path}")

    # Статистика churn drift
    df_customers['Year'] = pd.to_datetime(df_customers['RecordDate']).dt.year
    print("\nChurn rate по роках:")
    print(df_customers.groupby('Year')['Churn'].value_counts(normalize=True).unstack().round(3))

    # 2. Support conversations
    print("\nГенерація support conversations...")
    conv_data = []
    sampled_customers = df_customers.sample(n=conv_samples, replace=True)
    for _, customer in sampled_customers.iterrows():
        conv = generate_conversation(customer.to_dict())
        conv_data.append(conv)

    df_conversations = pd.DataFrame(conv_data)
    conv_path = output_path / "support_conversations.csv"
    df_conversations.to_csv(conv_path, index=False)
    print(f"Згенеровано та збережено {len(df_conversations):,} розмов → {conv_path}")

    # 3. Knowledge base
    print("\nГенерація knowledge base...")
    generate_knowledge_base(output_path)

    print("\nГотово! Дані підготовлені для MLOps / LLMOps демо.")