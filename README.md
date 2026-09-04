# AI Revenue Recovery — Razorpay AI Buildathon 2026

An AI system that predicts, decides, and explains the best action for a failed payment retry — turning lost transactions back into recovered revenue.

## Problem

Failed payments (OTP timeouts, insufficient funds, bank server issues) often result in permanently lost revenue when customers don't retry. This system intelligently decides which failed payments are worth retrying, when, and how — instead of blindly retrying everything or giving up entirely.

## Pipeline

Failed transaction → ML Classifier (Random Forest, predicts retry success probability) → Decision Engine (turns probability + business rules into an action) → LLM Explainer (Gemini, generates grounded human-readable explanations) → Optimal Retry Timing (finds the best hour to retry, using the model itself)

## Key design decisions

- **Random Forest over deep learning**: interpretable feature importances, robust on a modest dataset, minimal tuning needed.
- **Explicit uncertainty handling**: when the model's confidence is in the 25–70% "coin flip" zone, the system flags the case for human review instead of guessing — directly addressing failure-recovery and auditability.
- **Grounded LLM explanations**: the LLM is given the model's real feature importances and failure-reason-specific guidance, so it explains decisions using real evidence rather than inventing plausible-sounding reasons.
- **Optimal retry timing**: reuses the trained model to test the same transaction across all 24 hours, recommending the best retry window.

## Setup

conda create -n razorpay-ai python=3.11 -y
conda activate razorpay-ai
pip install -r requirements.txt


Create a `.env` file in the project root:

GEMINI_API_KEY=your_key_here


## Usage

python src/generate_data.py
python src/classifier.py
streamlit run app.py


## Project structure

├── data/ # synthetic failed-transactions dataset
├── src/
│ ├── generate_data.py # creates the synthetic dataset
│ ├── classifier.py # trains the Random Forest retry-success model
│ ├── decision_engine.py # decision logic + optimal retry timing
│ └── explainer.py # LLM-based grounded explanations
├── app.py # Streamlit dashboard
└── docs/
└── architecture.md # full architecture write-up


## Results

- ROC-AUC: 0.736 on held-out test data
- Top predictive factor: failure reason (40% feature importance)