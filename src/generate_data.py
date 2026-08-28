"""
Generates a synthetic dataset of failed payment transactions,
modeled on realistic Indian payment failure patterns.
"""

import pandas as pd
import numpy as np

np.random.seed(42)  # ensures the same "random" data every time we run this — reproducibility

N_ROWS = 5000

payment_methods = ["UPI", "Card", "Netbanking", "Wallet"]
# UPI dominates Indian payments volume, so we weight it heavily
method_weights = [0.55, 0.25, 0.15, 0.05]

banks = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank", "PNB"]

failure_reasons_by_method = {
    "UPI": ["otp_timeout", "bank_server_down", "insufficient_funds", "network_error"],
    "Card": ["card_declined", "insufficient_funds", "otp_timeout", "bank_server_down"],
    "Netbanking": ["bank_server_down", "session_timeout", "insufficient_funds"],
    "Wallet": ["insufficient_funds", "network_error"],
}

rows = []
for i in range(N_ROWS):
    method = np.random.choice(payment_methods, p=method_weights)
    bank = np.random.choice(banks)
    failure_reason = np.random.choice(failure_reasons_by_method[method])
    amount = round(np.random.lognormal(mean=6.5, sigma=1.0), 2)  # realistic skew: many small, few large txns
    hour = np.random.randint(0, 24)
    is_repeat_customer = np.random.choice([0, 1], p=[0.4, 0.6])
    previous_failed_attempts = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])

    # --- Build retry success probability based on realistic logic ---
    base_prob = 0.5

    if failure_reason == "insufficient_funds":
        base_prob -= 0.25  # unlikely to succeed immediately on retry
    if failure_reason == "otp_timeout":
        base_prob += 0.2   # often just a user slip, retry usually works
    if failure_reason == "bank_server_down":
        base_prob -= 0.1   # depends on timing
    if failure_reason == "network_error":
        base_prob += 0.15  # transient, retry often works

    if is_repeat_customer:
        base_prob += 0.05  # repeat customers tend to have healthier accounts

    base_prob -= previous_failed_attempts * 0.12  # diminishing returns on repeated retries

    # bank server load tends to spike during peak hours (10am-1pm, 7pm-10pm)
    if hour in [10, 11, 12, 19, 20, 21]:
        base_prob -= 0.08

    base_prob = np.clip(base_prob, 0.02, 0.95)
    retry_successful = np.random.binomial(1, base_prob)

    rows.append({
        "transaction_id": f"txn_{i:05d}",
        "amount": amount,
        "payment_method": method,
        "bank": bank,
        "failure_reason": failure_reason,
        "hour_of_day": hour,
        "is_repeat_customer": is_repeat_customer,
        "previous_failed_attempts": previous_failed_attempts,
        "retry_successful": retry_successful,
    })

df = pd.DataFrame(rows)
df.to_csv("data/failed_transactions.csv", index=False)
print(f"Generated {len(df)} rows -> data/failed_transactions.csv")
print(df["retry_successful"].value_counts(normalize=True))