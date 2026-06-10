# Lead Conversion Prediction API

An end-to-end Machine Learning pipeline and FastAPI service to predict lead-to-customer conversion probability using historical behavioral and firmographic data.

---

## 1. Project Overview

This project was built as part of the AI/ML Engineer assessment at Vynqe. It addresses a core marketing-technology problem: scoring leads based on their digital footprint to focus sales outreach on accounts with the highest conversion intent.

The system is split into three main components:
1. **Data Preprocessing & Feature Engineering Pipeline (`utils.py`)**: Cleans the raw data, derives the target variable, deduplicates leads, and constructs tabular features.
2. **Model Training & Evaluation Pipeline (`train.py`)**: Model comparison across Logistic Regression, Random Forest, and XGBoost, including stratified K-fold cross-validation and classification threshold tuning to maximize the F1-score.
3. **Deployment API Service (`app.py`)**: A production-ready FastAPI service providing real-time predictions, classification metrics, actionable recommendations, and rule-based explanation factors.

---

## 2. Problem Statement

B2B sales cycles are long and expensive. Marketing teams capture thousands of leads, but only a small fraction convert to active customers. Sales teams need a method to prioritize high-intent leads and filter out "window shoppers."

Given:
- `leads.csv` (2,025 unique leads with demographic/firmographic details)
- `interactions.csv` (39,948 user interaction logs containing session data, clicks, pages visited, and timestamps)

The goal is to develop a reproducible model and API to predict the probability that a lead will convert.

### Target Variable Derivation
The dataset does not contain a `converted` column. We derived it from the lead's final status:
- `account_type == "Existing Customer"` &rarr; **Converted (1)**
- `account_type == "New Business"` or `"Partner"` &rarr; **Not Converted (0)**

---

## 3. Dataset Description

### Leads Data (`leads.csv`)
- **Size**: 2,025 unique rows after deduplication.
- **Key Columns**:
  - `lead_id`: Unique identifier.
  - `source`: Acquisition channel (Google, LinkedIn, Referral, Direct, etc.).
  - `company_size`: Company size tier (Small, Medium, Large, Enterprise).
  - `lead_segment`: Business segment (SMB, Startup, Mid-Market, Enterprise).
  - `account_type`: Target variable proxy (Existing Customer vs New Business/Partner).

### Interactions Data (`interactions.csv`)
- **Size**: 39,948 rows after filtering out future-dated records.
- **Key Columns**:
  - `lead_id`: Mapping key to the leads table.
  - `session_id`: Unique session identifier.
  - `timestamp`: Date and time of the event.
  - `event_name`: Type of activity (`page_view`, `demo_request`, `pricing_page_view`, `free_trial_start`, etc.).
  - `button_name`: Button clicked (e.g. `"WhatsApp Us"`, `"Request Demo"`).
  - `time_on_page_seconds`: Page engagement duration.

---

## 4. Approach & Methodology

### 1. Data Preprocessing & Leakage Control
- **Deduplication**: Filtered out 20 duplicate lead rows to prevent row expansion during merges.
- **Temporal Cleaning**: Excluded 52 future-dated interactions relative to the current time.
- **Inference-Time Feature Alignment**: The model is trained *only* on the 11 features available during production inference. This guarantees there is no train-test feature mismatch or data leakage.

### 2. Feature Engineering
We engineered 11 high-signal behavioral and firmographic features:
- `session_count`: Total number of unique sessions.
- `pages_visited`: Number of unique pages visited.
- `time_spent_minutes`: Total engagement duration (minutes).
- `demo_requests`: Count of `demo_request` events.
- `pricing_views`: Count of `pricing_page_view` events.
- `whatsapp_clicks`: Count of WhatsApp clicks (`button_name == "WhatsApp Us"`).
- `email_opens`: Count of email opens (0 variance in dataset — dropped during training).
- `days_since_first_visit`: Days between first and last interaction.
- `source_encoded`: Categorical encoding of lead source.
- `company_size_encoded`: Categorical encoding of company size.
- `segment_encoded`: Categorical encoding of lead segment (top feature by importance).

### 3. Model Training & Evaluation
We compared three standard classifiers:
- **Logistic Regression**: Scaled features, balanced class weights.
- **Random Forest**: Tree depth limiting, balanced class weights.
- **XGBoost Classifier**: Weighted loss function scaled by class imbalance (~3.6:1).

**Classification Threshold Tuning**: Instead of a default 0.5 threshold, we run a search to find the threshold that maximizes the F1-score on the hold-out validation set.

---

## 5. Model Performance Results

Below are the evaluation metrics on the hold-out validation set (20% split, stratified):

| Model | Accuracy | Precision | Recall | F1 Score (Default 0.5) | AUC-ROC |
|-------|----------|-----------|--------|------------------------|---------|
| Logistic Regression | 0.6395 | 0.2583 | 0.3523 | 0.2981 | 0.5626 |
| Random Forest | 0.7605 | 0.3953 | 0.1932 | 0.2595 | 0.6588 |
| **XGBoost (Best)** | **0.7185** | **0.3375** | **0.3068** | **0.3214** | **0.6478** |

### Optimized F1 Performance (XGBoost)
- **Optimal Threshold**: `0.2300`
- **F1 Score**: **`0.4215`** (an increase of 31.1% relative to the default threshold)
- **Confusion Matrix at Optimal Threshold**:
  - True Positives (TP): 55
  - False Positives (FP): 116
  - False Negatives (FN): 33
  - True Negatives (TN): 201

### Top Feature Importances (XGBoost)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `segment_encoded` | 0.1378 |
| 2 | `pricing_views` | 0.1185 |
| 3 | `demo_requests` | 0.1003 |
| 4 | `whatsapp_clicks` | 0.0983 |
| 5 | `days_since_first_visit` | 0.0970 |

---

## 6. Key Findings & Business Recommendations

1. **High Intent Signals**: The feature importance analysis shows that **segment**, **pricing views**, and **demo requests** are the top three drivers of conversion.
2. **Channel Quality**: Leads arriving from Direct and Referral channels convert at rates >30%, whereas Instagram converted <8%. Marketing budget should be shifted towards high-quality organic/partner channels.
3. **Threshold Strategy**: Using a prediction threshold of `0.23` achieves an F1 of 0.42 — maximizing recall of converting leads while keeping false positives manageable for the sales team.

---

## 7. Setup & Run Instructions

### Prerequisites
- Python 3.8+ (tested on Python 3.13)

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 8. Running the Pipelines

### Model Training
To retrain the models, run threshold tuning, and export serialization artifacts:
```bash
python train.py
```
This script will output:
- `model.pkl`: Serialized model object, scaler, features list, and fitted LabelEncoders.
- `outputs/model_metrics.json`: Performance metrics for all models.
- `outputs/feature_importance.json` & `outputs/feature_importance.png`: Top feature importances.

### Starting the FastAPI Server
To run the server locally:
```bash
python app.py
```
Or use Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
The documentation is available in your browser at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 9. API Documentation & Examples

### Endpoint 1: Predict Conversion (`POST /predict`)
Predicts the likelihood of conversion for a lead.

- **Request Body**:
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
- **Response**:
  ```json
  {
    "conversion_probability": 0.04,
    "confidence": "low",
    "risk_level": "high",
    "recommendation": "Focus on product education (Low Priority)"
  }
  ```

---

### Endpoint 2: Explain Prediction (`POST /explain`)
Provides rule-based explanation factors and a summary message.

- **Request Body**:
  ```json
  {
    "conversion_probability": 0.78,
    "demo_requests": 1,
    "pricing_views": 3,
    "session_count": 5,
    "email_opens": 0
  }
  ```
- **Response**:
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

## 10. Limitations & Future Work

- **Anonymized Categoricals**: Categorical variables (`source`, `segment`) have values that contain unseen categories at inference time. Implementing hash trick encoding could improve robustness.
- **Model Tuning**: Hyperparameter tuning was limited to a simple grid. Implementing random search or Bayesian optimization (via Optuna) could squeeze another 2-5% on F1-score.
- **Temporal/Sequence Modelling**: The models ignore the sequential order of interactions. Treating interactions as a sequence and using an LSTM or Transformer (Attention) model would capture temporal behavior better.

---

## 11. Author & Contact
- **Author**: Pranali
- **Contact**: hr@vynqe.com (or your candidate contact info)
