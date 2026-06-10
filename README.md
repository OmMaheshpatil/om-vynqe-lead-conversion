# 🎯 Lead Conversion Prediction API

> An end-to-end Machine Learning pipeline and production-ready FastAPI service to predict B2B lead-to-customer conversion probability using behavioral and firmographic data.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/Assessment-Vynqe-purple)

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Project Overview](#1-project-overview) |
| 2 | [Problem Statement](#2-problem-statement) |
| 3 | [Dataset Description](#3-dataset-description) |
| 4 | [Approach & Methodology](#4-approach--methodology) |
| 5 | [Model Performance Results](#5-model-performance-results) |
| 6 | [Key Findings & Business Recommendations](#6-key-findings--business-recommendations) |
| 7 | [Setup & Installation](#7-setup--installation) |
| 8 | [Running the Pipelines](#8-running-the-pipelines) |
| 9 | [API Documentation & Examples](#9-api-documentation--examples) |
| 10 | [Repository Structure](#10-repository-structure) |
| 11 | [Limitations & Future Work](#11-limitations--future-work) |
| 12 | [Author & Contact](#12-author--contact) |

---

## 1. Project Overview

This project was built as part of the **AI/ML Engineer Assessment at Vynqe**. It addresses a core marketing-technology problem: scoring B2B leads based on their digital behaviour to help sales teams focus outreach on the highest-intent accounts.

The system is composed of three tightly integrated components:

| Component | File | Description |
|-----------|------|-------------|
| 🔧 Data Pipeline | `utils.py` | Raw data loading, target derivation, deduplication, and 11-feature engineering |
| 🤖 Training Pipeline | `train.py` | Multi-model comparison, stratified K-fold CV, threshold tuning, and artifact export |
| 🚀 API Service | `app.py` | Production-ready FastAPI with `/predict` and `/explain` endpoints |

---

## 2. Problem Statement

B2B sales cycles are long and expensive. Marketing teams capture thousands of leads, but only a small fraction convert to paying customers. Sales teams need a systematic method to **prioritize high-intent leads** and deprioritize cold prospects.

**Given:**

| Dataset | Description |
|---------|-------------|
| `leads.csv` | 2,025 unique leads with demographic and firmographic details |
| `interactions.csv` | 39,950 behavioral event logs (sessions, clicks, page views, timestamps) |

**Goal:** Build a reproducible ML model and REST API to predict the probability that a lead will convert into a customer.

### Target Variable Derivation

The dataset does **not** contain a `converted` column. It is derived from `account_type`:

| `account_type` value | Label | Converted |
|----------------------|-------|-----------|
| `Existing Customer` | Positive | ✅ `1` |
| `New Business` | Negative | ❌ `0` |
| `Partner` | Negative | ❌ `0` |

---

## 3. Dataset Description

### 📄 Leads Data (`leads.csv`)

| Property | Value |
|----------|-------|
| Raw rows | 2,045 |
| Unique leads (after dedup) | **2,025** |
| Duplicates removed | 20 |

| Key Column | Description |
|------------|-------------|
| `lead_id` | Unique lead identifier |
| `source` | Acquisition channel (Google, LinkedIn, Referral, Direct, etc.) |
| `company_size` | Company size tier (Small, Medium, Large, Enterprise) |
| `lead_segment` | Business segment (SMB, Startup, Mid-Market, Enterprise) |
| `account_type` | Target variable proxy |

### 📊 Interactions Data (`interactions.csv`)

| Property | Value |
|----------|-------|
| Raw rows | 40,000 |
| After cleaning | **39,950** |
| Future-dated rows removed | 50 |

| Key Column | Description |
|------------|-------------|
| `lead_id` | Foreign key linking to leads |
| `session_id` | Unique session identifier |
| `timestamp` | Date and time of event |
| `event_name` | Event type (`page_view`, `demo_request`, `pricing_page_view`, etc.) |
| `button_name` | Button clicked (e.g. `"WhatsApp Us"`, `"Request Demo"`) |
| `time_on_page_seconds` | Page engagement duration |
| `funnel_stage` | Stage in funnel (Awareness → Consideration → Evaluation → Decision) |

---

## 4. Approach & Methodology

### Step 1 — Data Preprocessing & Leakage Control

| Step | Action | Result |
|------|--------|--------|
| Deduplication | Remove duplicate `lead_id` rows | 20 rows removed |
| Temporal Cleaning | Drop future-dated interaction records | 50 rows removed |
| Feature Alignment | Train only on features available at inference time | No data leakage |

### Step 2 — Feature Engineering

We engineer **11 lead-level features** from raw interaction logs:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `session_count` | Behavioral | Total unique sessions per lead |
| 2 | `pages_visited` | Behavioral | Unique pages visited |
| 3 | `time_spent_minutes` | Behavioral | Total time on site (minutes) |
| 4 | `demo_requests` | Intent Signal | Count of `demo_request` events |
| 5 | `pricing_views` | Intent Signal | Count of `pricing_page_view` events |
| 6 | `whatsapp_clicks` | Intent Signal | Count of `"WhatsApp Us"` button clicks |
| 7 | `email_opens` | Intent Signal | Count of email_open events (zero variance — dropped in training) |
| 8 | `days_since_first_visit` | Recency | Days between first and last interaction |
| 9 | `source_encoded` | Firmographic | Label-encoded acquisition channel |
| 10 | `company_size_encoded` | Firmographic | Label-encoded company size tier |
| 11 | `segment_encoded` | Firmographic | Label-encoded business segment **(top feature)** |

### Step 3 — Model Training & Evaluation

Three classifiers were compared with the following configurations:

| Model | Key Configuration |
|-------|------------------|
| Logistic Regression | `max_iter=2000`, `class_weight=balanced`, scaled features |
| Random Forest | `n_estimators=200`, `max_depth=12`, `class_weight=balanced` |
| XGBoost | `n_estimators=300`, `scale_pos_weight` tuned for ~3.6:1 class imbalance |

- **Validation**: Stratified 80/20 train-test split + 5-fold stratified cross-validation
- **Threshold Tuning**: Grid search over [0.1, 0.9] to find the F1-maximizing threshold

---

## 5. Model Performance Results

### 5.1 Model Comparison (Default 0.5 Threshold)

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 0.6395 | 0.2583 | 0.3523 | 0.2981 | 0.5626 |
| Random Forest | 0.7605 | 0.3953 | 0.1932 | 0.2595 | 0.6588 |
| **XGBoost Best** | **0.7185** | **0.3375** | **0.3068** | **0.3214** | **0.6478** |

### 5.2 XGBoost — Optimized Performance

| Metric | Value |
|--------|-------|
| **Optimal Threshold** | `0.23` |
| **F1 Score (optimized)** | **0.4215** |
| F1 Score (default 0.5) | 0.3214 |
| Improvement | **+31.1%** |

**Confusion Matrix at Optimal Threshold (0.23):**

| | Predicted: Not Converted | Predicted: Converted |
|--|:------------------------:|:--------------------:|
| **Actual: Not Converted** | TN = 201 | FP = 116 |
| **Actual: Converted** | FN = 33 | TP = 55 |

### 5.3 Top Feature Importances (XGBoost)

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1st | `segment_encoded` | 0.1378 |
| 2nd | `pricing_views` | 0.1185 |
| 3rd | `demo_requests` | 0.1003 |
| 4th | `whatsapp_clicks` | 0.0983 |
| 5th | `days_since_first_visit` | 0.0970 |
| 6th | `source_encoded` | 0.0970 |
| 7th | `time_spent_minutes` | 0.0941 |
| 8th | `company_size_encoded` | 0.0877 |
| 9th | `pages_visited` | 0.0863 |
| 10th | `session_count` | 0.0831 |

---

## 6. Key Findings & Business Recommendations

| # | Finding | Recommendation |
|---|---------|----------------|
| 1 | **Segment is the #1 predictor** — Enterprise and Mid-Market leads convert at significantly higher rates | Prioritize Enterprise and Mid-Market in sales workflows |
| 2 | **Direct (38.1%) and Referral (30.7%)** channels far outperform Instagram (7.9%) | Shift budget from Instagram to referral programs and organic/direct |
| 3 | **`pricing_views` and `demo_requests`** are the top behavioral intent signals | Trigger immediate sales follow-up when these events occur |
| 4 | **Threshold 0.23** (vs default 0.5) improves F1 by 31% | Use 0.23 as the operational decision boundary for lead scoring |
| 5 | **Healthcare (28.9%) and Education (26.6%)** convert above the baseline average | Create industry-specific nurture sequences for these verticals |

---

## 7. Setup & Installation

### Prerequisites

- Python **3.8+** (tested on Python 3.13)
- `pip` package manager

### Installation Steps

**1. Clone the repository:**
```bash
git clone <your-repo-url>
cd <repo-name>
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | >= 2.0.3 | Data manipulation |
| `numpy` | >= 1.24.3 | Numerical operations |
| `scikit-learn` | >= 1.3.1 | ML models and preprocessing |
| `xgboost` | >= 2.0.0 | Gradient boosting classifier |
| `fastapi` | >= 0.104.1 | REST API framework |
| `uvicorn` | >= 0.24.0 | ASGI server |
| `pydantic` | >= 2.5.0 | Request/response validation |
| `matplotlib` | >= 3.8.0 | Feature importance plots |
| `seaborn` | >= 0.12.2 | Statistical visualizations |

---

## 8. Running the Pipelines

### Model Training

To retrain models, tune thresholds, and export all artifacts:

```bash
python train.py
```

**Outputs generated:**

| File | Location | Description |
|------|----------|-------------|
| `model.pkl` | Root | Serialized XGBoost bundle (model + scaler + encoders + threshold) |
| `model_metrics.json` | `outputs/` | Accuracy, F1, Precision, Recall, AUC-ROC for all models |
| `feature_importance.json` | `outputs/` | Ranked feature importance values |
| `feature_importance.png` | `outputs/` | Feature importance bar chart |

### Starting the FastAPI Server

**Option A — Direct Python:**
```bash
python app.py
```

**Option B — Uvicorn (recommended):**
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive API docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 9. API Documentation & Examples

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict conversion probability for a lead |
| `POST` | `/explain` | Get human-readable explanation of a prediction |

---

### POST /predict — Predict Conversion

**Input Fields:**

| Field | Type | Required | Example | Description |
|-------|------|:--------:|---------|-------------|
| `session_count` | `int` | Yes | `4` | Number of unique sessions |
| `total_interactions` | `int` | Yes | `15` | Total number of web events |
| `total_time_spent` | `float` | Yes | `420.0` | Total time on site (seconds) |
| `demo_requests` | `int` | Yes | `1` | Number of demo requests submitted |
| `pricing_views` | `int` | Yes | `3` | Number of pricing page views |
| `whatsapp_clicks` | `int` | Yes | `2` | Number of WhatsApp button clicks |
| `email_opens` | `int` | Yes | `0` | Number of email opens |
| `source` | `string` | Yes | `"Google"` | Acquisition channel |
| `company_size` | `string` | Yes | `"Medium"` | Company size tier |
| `segment` | `string` | Yes | `"Mid-Market"` | Business segment |

**Example Request:**
```json
{
  "session_count": 4,
  "total_interactions": 15,
  "total_time_spent": 420.0,
  "demo_requests": 1,
  "pricing_views": 3,
  "whatsapp_clicks": 2,
  "email_opens": 0,
  "source": "Google",
  "company_size": "Medium",
  "segment": "Mid-Market"
}
```

**Example Response:**

| Field | Type | Description |
|-------|------|-------------|
| `conversion_probability` | `float` | Predicted probability (0.0 to 1.0) |
| `confidence` | `string` | `low` / `medium` / `high` |
| `risk_level` | `string` | `low` / `medium` / `high` |
| `recommendation` | `string` | Actionable sales next step |

```json
{
  "conversion_probability": 0.76,
  "confidence": "high",
  "risk_level": "low",
  "recommendation": "Ready for sales outreach (High Priority)"
}
```

---

### POST /explain — Explain Prediction

**Example Request:**
```json
{
  "conversion_probability": 0.78,
  "demo_requests": 1,
  "pricing_views": 3,
  "session_count": 5,
  "email_opens": 0
}
```

**Example Response:**
```json
{
  "summary": "This lead shows exceptional interest with a demo request and strong engagement across multiple sessions.",
  "factors": [
    "Demo request submitted (high intent)",
    "Multiple pricing page visits (strong interest)",
    "Multiple sessions indicate strong engagement"
  ]
}
```

---

## 10. Repository Structure

```
vynqe-lead-conversion/
|
|-- README.md                  <- This file
|-- TASK.md                    <- Assessment brief
|-- analysis.md                <- Exploratory Data Analysis report
|-- requirements.txt           <- Python dependencies
|
|-- utils.py                   <- Data pipeline (loading, cleaning, features)
|-- train.py                   <- Model training and evaluation pipeline
|-- app.py                     <- FastAPI REST API service
|-- model.pkl                  <- Trained XGBoost model bundle
|
|-- data/
|   |-- leads.csv              <- Lead demographics (gitignored)
|   `-- interactions.csv       <- Behavioral event logs (gitignored)
|
`-- outputs/
    |-- model_metrics.json     <- All model evaluation metrics
    |-- feature_importance.json <- Ranked feature importances
    `-- feature_importance.png <- Feature importance bar chart
```

---

## 11. Limitations & Future Work

| Limitation | Impact | Proposed Fix |
|------------|--------|--------------|
| **Unseen categorical values at inference** | Unknown source/segment values default silently to 0 | Implement hash trick or frequency encoding |
| **Basic hyperparameter tuning** | Model may be sub-optimal | Use Optuna or Bayesian search for +2-5% F1 |
| **No sequential modeling** | Ignores the temporal order of user interactions | Apply LSTM or Transformer attention on event sequences |
| **Static threshold** | 0.23 is fixed; optimal value may shift over time | Add live threshold recalibration using recent conversion data |
| **No real-time retraining** | Model becomes stale as lead behavior evolves | Build an MLflow/Airflow scheduled retraining pipeline |

---

## 12. Author & Contact

| Field | Details |
|-------|---------|
| **Author** | Pranali |
| **Assessment** | AI/ML Engineer — Vynqe |
| **Contact** | hr@vynqe.com |

---

*Built for the Vynqe AI/ML Engineer Assessment*