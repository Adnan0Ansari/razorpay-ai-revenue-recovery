"""
Takes a transaction + the model's predicted probability of retry success,
and decides a concrete action: AUTO_RETRY, HOLD_FOR_HUMAN_REVIEW, or
ESCALATE_ALTERNATE_CHANNEL. Includes explicit fallback logic for
low-confidence or risky cases.
"""

import pandas as pd
import joblib

# --- Load the trained model + encoders from Step 4 ---
model = joblib.load("src/retry_model.pkl")
encoders = joblib.load("src/encoders.pkl")

# --- Thresholds: the core "policy" of our decision engine ---
HIGH_CONFIDENCE_SUCCESS = 0.70   # above this, we trust the model to auto-retry
HIGH_CONFIDENCE_FAILURE = 0.25   # below this, we trust the model to NOT retry
MAX_AUTO_RETRY_ATTEMPTS = 2      # cap on automated retries per customer, regardless of confidence


def predict_probability(transaction: dict) -> float:
    """Runs a single transaction through the trained model and returns
    the probability (0 to 1) that a retry will succeed."""
    row = {
        "amount": transaction["amount"],
        "hour_of_day": transaction["hour_of_day"],
        "is_repeat_customer": transaction["is_repeat_customer"],
        "previous_failed_attempts": transaction["previous_failed_attempts"],
        "payment_method_enc": encoders["payment_method"].transform([transaction["payment_method"]])[0],
        "bank_enc": encoders["bank"].transform([transaction["bank"]])[0],
        "failure_reason_enc": encoders["failure_reason"].transform([transaction["failure_reason"]])[0],
    }
    X = pd.DataFrame([row])
    proba = model.predict_proba(X)[0][1]  # probability of class "1" (success)
    return proba


def decide_action(transaction: dict) -> dict:
    """Core decision logic: returns an action, confidence, and reasoning."""
    proba = predict_probability(transaction)

    # --- Fallback rule 1: too many prior attempts, don't keep auto-retrying ---
    if transaction["previous_failed_attempts"] >= MAX_AUTO_RETRY_ATTEMPTS:
        return {
            "action": "ESCALATE_ALTERNATE_CHANNEL",
            "confidence": round(proba, 3),
            "reason": f"Customer has already failed {transaction['previous_failed_attempts']} times. "
                      f"Avoiding repeated auto-retries; suggesting an alternate payment channel instead."
        }

    # --- Confident success: safe to automate ---
    if proba >= HIGH_CONFIDENCE_SUCCESS:
        return {
            "action": "AUTO_RETRY",
            "confidence": round(proba, 3),
            "reason": f"Model is confident ({proba:.0%}) this retry will succeed based on transaction pattern."
        }

    # --- Confident failure: don't waste a retry, but don't give up either ---
    if proba <= HIGH_CONFIDENCE_FAILURE:
        return {
            "action": "ESCALATE_ALTERNATE_CHANNEL",
            "confidence": round(proba, 3),
            "reason": f"Model is confident ({(1-proba):.0%}) a simple retry will fail; "
                      f"recommending an alternate approach instead of retrying blindly."
        }

    # --- Uncertain zone (the fallback case we designed deliberately) ---
    return {
        "action": "HOLD_FOR_HUMAN_REVIEW",
        "confidence": round(proba, 3),
        "reason": f"Model confidence ({proba:.0%}) is too close to a coin flip to act on automatically. "
                  f"Flagging for human review rather than guessing."
    }

def find_optimal_retry_hour(transaction: dict) -> dict:
    """Tests this transaction's success probability across all 24 hours
    and returns the hour with the highest predicted success rate."""
    best_hour = None
    best_proba = -1
    hourly_results = {}

    for hour in range(24):
        test_txn = transaction.copy()
        test_txn["hour_of_day"] = hour
        proba = predict_probability(test_txn)
        hourly_results[hour] = round(proba, 3)
        if proba > best_proba:
            best_proba = proba
            best_hour = hour

    current_hour = transaction["hour_of_day"]
    current_proba = hourly_results[current_hour]

    return {
        "current_hour": current_hour,
        "current_hour_confidence": current_proba,
        "recommended_hour": best_hour,
        "recommended_hour_confidence": round(best_proba, 3),
        "improvement": round(best_proba - current_proba, 3),
        "hourly_breakdown": hourly_results,
    }

if __name__ == "__main__":
    # --- Quick manual test with a few example transactions ---
    test_transactions = [
        {"amount": 500, "payment_method": "UPI", "bank": "HDFC", "failure_reason": "otp_timeout",
         "hour_of_day": 14, "is_repeat_customer": 1, "previous_failed_attempts": 0},
        {"amount": 12000, "payment_method": "Card", "bank": "SBI", "failure_reason": "insufficient_funds",
         "hour_of_day": 20, "is_repeat_customer": 0, "previous_failed_attempts": 3},
        {"amount": 800, "payment_method": "Netbanking", "bank": "Axis", "failure_reason": "bank_server_down",
         "hour_of_day": 11, "is_repeat_customer": 1, "previous_failed_attempts": 1},
    ]

    for txn in test_transactions:
        result = decide_action(txn)
        print(txn)
        print("->", result)
        print()