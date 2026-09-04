# Architecture — AI Revenue Recovery

## Overview

This system processes a failed payment transaction through four stages, each handled by the tool best suited for that specific job.

## 1. Data Layer

A synthetic dataset of 5,000 failed transactions, modeled on realistic Indian payment failure patterns (UPI, Card, Netbanking, Wallet failures across major banks). Success probabilities were deliberately engineered based on domain intuition (e.g. OTP timeouts are usually simple retries; insufficient funds are not), which we later validated the model independently rediscovered via feature importance analysis.

## 2. Prediction Layer — Random Forest Classifier

**Why this model:** handles mixed categorical/numeric features without heavy preprocessing, resists overfitting via bagging (200 trees, each trained on random data/feature subsets), and — critically — provides interpretable feature importances that later stages depend on.

**Why not an LLM for prediction:** structured tabular prediction is a statistics problem, not a language problem. An LLM would be slower, more expensive, and less reliable here than a purpose-built model.

**Evaluation:** 80/20 train/test split, ROC-AUC 0.736 on held-out data.

## 3. Decision Layer — Rule-Based Policy Engine

Converts the model's probability into an action:
- **AUTO_RETRY** (confidence ≥ 70%)
- **ESCALATE_ALTERNATE_CHANNEL** (confidence ≤ 25%, OR 2+ prior failed attempts — this business rule is checked before confidence, prioritizing customer experience over raw model optimism)
- **HOLD_FOR_HUMAN_REVIEW** (confidence 25–70%, genuinely uncertain)

This explicit uncertainty-handling is the system's core safeguard: it never forces an automated decision when the model itself is not confident.

## 4. Explanation Layer — Grounded LLM Reasoning

The LLM (Gemini) is given the transaction, the decision, the model's real feature importances, and failure-reason-specific guidance — then asked to generate (a) an internal analyst explanation and (b) a customer-facing message, using only the provided evidence. This prevents the LLM from hallucinating plausible-sounding but false explanations, and ensures every generated explanation is traceable back to real model output.

## 5. Optimal Retry Timing

Reuses the trained classifier to evaluate the same transaction across all 24 possible retry hours, recommending the hour with the highest predicted success probability — turning a binary retry decision into an actionable scheduling recommendation.

## Design Principles

1. **Right tool for each job**: ML for prediction, rules for policy, LLM for language — never blurred together.
2. **Explainability by construction**: every explanation is grounded in real model output, not generated freely.
3. **Honest uncertainty**: the system defers to humans when genuinely unsure, rather than forcing a guess.