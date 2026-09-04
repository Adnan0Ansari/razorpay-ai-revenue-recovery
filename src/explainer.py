"""
Calls Google's Gemini API to turn a raw model decision into:
1. An internal, analyst-facing explanation
2. A customer-facing nudge message

Grounds the explanation in the model's actual feature importances,
rather than letting the LLM invent a reason from scratch.
"""

import os
from dotenv import load_dotenv
from google import genai

# --- Load the .env file into environment variables ---
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- The model's real feature importance ranking from Step 4 ---
FEATURE_IMPORTANCE_SUMMARY = (
    "failure_reason (40%), amount (19%), previous_failed_attempts (14%), "
    "hour_of_day (12%), bank (8%), payment_method (4%), is_repeat_customer (3%)"
)
FAILURE_REASON_GUIDANCE = {
    "insufficient_funds": "Customer should add funds to their account before retrying — this is the direct fix.",
    "otp_timeout": "Likely a simple user delay entering the OTP — retrying immediately usually works.",
    "bank_server_down": "A temporary bank-side issue — suggest retrying after a short wait, or a different bank/method.",
    "network_error": "Likely a connectivity glitch — retrying on a stable connection usually works.",
    "card_declined": "Could be bank-side restrictions — suggest checking with their bank or trying another card/method.",
    "session_timeout": "The session expired before completing — suggest retrying and completing the process more quickly.",
}

def explain_decision(transaction: dict, decision: dict) -> dict:
    """Sends the transaction + decision to Gemini, asks for grounded
    internal + customer-facing explanations."""

    prompt = f"""You are helping explain an automated payment-retry decision made by a
trained machine learning model at a payments company.

TRANSACTION DETAILS:
- Amount: ₹{transaction['amount']}
- Payment method: {transaction['payment_method']}
- Bank: {transaction['bank']}
- Failure reason: {transaction['failure_reason']}
- Hour of day: {transaction['hour_of_day']}:00
- Repeat customer: {"Yes" if transaction['is_repeat_customer'] else "No"}
- Previous failed attempts: {transaction['previous_failed_attempts']}

MODEL DECISION:
- Action: {decision['action']}
- Confidence (probability of retry success): {decision['confidence']}
- Rule-based reason: {decision['reason']}

GUIDANCE FOR THIS SPECIFIC FAILURE REASON:
{FAILURE_REASON_GUIDANCE.get(transaction['failure_reason'], "No specific guidance available.")}

THE MODEL'S FEATURE IMPORTANCE (what it actually relies on most, ranked):
{FEATURE_IMPORTANCE_SUMMARY}

Write two things, grounded ONLY in the data above (do not invent facts not given):

1. INTERNAL_EXPLANATION: A short, precise explanation (2-3 sentences) for a risk
analyst, referencing which factors likely drove this specific prediction.

2. CUSTOMER_MESSAGE: A short, friendly SMS/email-style message (2-3 sentences) to
send the customer, IF appropriate for this action. If the action is
HOLD_FOR_HUMAN_REVIEW, instead write "No customer message needed yet — pending review."

Respond in exactly this format, nothing else:
INTERNAL_EXPLANATION: <text>
CUSTOMER_MESSAGE: <text>
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        text = response.text
    except Exception as e:
        # --- Graceful fallback: don't crash the app if the LLM API fails ---
        return {
            "internal_explanation": f"[AI explanation temporarily unavailable — API error: {type(e).__name__}. "
                                     f"Rule-based reason: {decision['reason']}]",
            "customer_message": "We're reviewing your payment and will follow up shortly."
                                 if decision["action"] != "HOLD_FOR_HUMAN_REVIEW"
                                 else "No customer message needed yet — pending review.",
        }

    # --- Parse the two labeled sections out of the response ---
    internal = ""
    customer = ""
    if "INTERNAL_EXPLANATION:" in text and "CUSTOMER_MESSAGE:" in text:
        parts = text.split("CUSTOMER_MESSAGE:")
        internal = parts[0].replace("INTERNAL_EXPLANATION:", "").strip()
        customer = parts[1].strip()
    else:
        internal = text  # fallback: just return raw text if parsing fails

    return {"internal_explanation": internal, "customer_message": customer}


if __name__ == "__main__":
    from decision_engine import decide_action

    test_txn = {
        "amount": 500, "payment_method": "UPI", "bank": "HDFC",
        "failure_reason": "otp_timeout", "hour_of_day": 14,
        "is_repeat_customer": 1, "previous_failed_attempts": 0,
    }
    decision = decide_action(test_txn)
    explanation = explain_decision(test_txn, decision)

    print("DECISION:", decision)
    print()
    print("INTERNAL EXPLANATION:\n", explanation["internal_explanation"])
    print()
    print("CUSTOMER MESSAGE:\n", explanation["customer_message"])