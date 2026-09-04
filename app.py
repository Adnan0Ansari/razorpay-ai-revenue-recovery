"""
Streamlit dashboard: pick a failed transaction, run it through our
classifier -> decision engine -> LLM explainer pipeline, and display
the result.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))   # MUST come first

from decision_engine import decide_action, find_optimal_retry_hour, log_decision   # THEN this
from explainer import explain_decision

st.set_page_config(page_title="AI Revenue Recovery", layout="centered")
st.title("AI Revenue Recovery Dashboard")
st.caption("Predict, decide, and explain retry actions for failed payments.")

# --- Load our existing dataset so users can pick a real example ---
df = pd.read_csv("data/failed_transactions.csv")

st.subheader("1. Choose a failed transaction")

mode = st.radio("How do you want to pick a transaction?", ["Select from dataset", "Enter manually"])

if mode == "Select from dataset":
    row_index = st.number_input("Row index (0 to 4999)", min_value=0, max_value=len(df)-1, value=0)
    txn = df.iloc[int(row_index)].to_dict()
else:
    amount = st.number_input("Amount (₹)", min_value=1.0, value=500.0)
    payment_method = st.selectbox("Payment method", ["UPI", "Card", "Netbanking", "Wallet"])
    bank = st.selectbox("Bank", ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank", "PNB"])
    failure_reason = st.selectbox("Failure reason",
        ["otp_timeout", "insufficient_funds", "bank_server_down", "network_error", "card_declined", "session_timeout"])
    hour_of_day = st.slider("Hour of day", 0, 23, 14)
    is_repeat_customer = st.checkbox("Repeat customer?", value=True)
    previous_failed_attempts = st.slider("Previous failed attempts", 0, 5, 0)

    txn = {
        "amount": amount, "payment_method": payment_method, "bank": bank,
        "failure_reason": failure_reason, "hour_of_day": hour_of_day,
        "is_repeat_customer": int(is_repeat_customer),
        "previous_failed_attempts": previous_failed_attempts,
    }

st.write("**Selected transaction:**", txn)

# --- Run the pipeline when the user clicks the button ---
if st.button("Run Analysis"):
    with st.spinner("Running model, decision engine, and explainer..."):
        decision = decide_action(txn)
        explanation = explain_decision(txn, decision)
        log_decision(txn, decision, explanation)
    st.subheader("2. Result")

    # --- Color-coded action badge ---
    action_colors = {
        "AUTO_RETRY": "🟢",
        "HOLD_FOR_HUMAN_REVIEW": "🟡",
        "ESCALATE_ALTERNATE_CHANNEL": "🟠",
    }
    badge = action_colors.get(decision["action"], "")
    st.markdown(f"### {badge} {decision['action']}")
    st.write(f"**Confidence (probability of retry success):** {decision['confidence']:.0%}")
    st.write(f"**Rule-based reason:** {decision['reason']}")

    st.subheader("3. AI-Generated Explanation")
    st.write("**For the risk analyst:**")
    st.info(explanation["internal_explanation"])

    st.write("**For the customer:**")
    st.success(explanation["customer_message"])

    from decision_engine import find_optimal_retry_hour
    timing = find_optimal_retry_hour(txn)

    st.subheader("4. Optimal Retry Timing")
    if not timing["applicable"]:
        st.write(f"⏸️ {timing['reason_skipped']}")
    elif timing["improvement"] > 0.02:
        st.write(f"⏰ Retrying at **{timing['recommended_hour']}:00** instead of "
                 f"**{timing['current_hour']}:00** could improve success chances from "
                 f"**{timing['current_hour_confidence']:.0%}** to **{timing['recommended_hour_confidence']:.0%}**.")
        st.line_chart(pd.Series(timing["hourly_breakdown"]))
    else:
        st.write(f"Current time ({timing['current_hour']}:00) is already close to optimal.")
        st.line_chart(pd.Series(timing["hourly_breakdown"]))
    