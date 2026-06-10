# 🤖 AI/ML Engineer Assessment — Vynqe

## 🎯 Overview

This is a **real-world AI Systems challenge** designed to assess your ability to:
- Analyze behavioral data
- Build predictive ML models
- Deploy production-ready APIs
- Integrate LLM capabilities
- Write clean, documented code

You are acting as an AI/ML Engineer at a growing SaaS company tasked with predicting which leads are likely to convert into customers.

---

## 📋 Challenge Summary

**Duration**: 48 Hours
**Difficulty**: Intermediate to Advanced
**Tech Stack**: Python, FastAPI, ML (Scikit-learn/XGBoost/etc.)

You will:
1. Perform exploratory data analysis (EDA)
2. Build a conversion prediction model
3. Create a FastAPI service with inference endpoints
4. Add an LLM explanation layer
5. Document everything professionally

---

## 📦 What You're Given

Two CSV files representing real lead and interaction behavior:

### `leads.csv` (~2,000 rows)
Customer acquisition records with:
- lead_id, name, company, segment, company_size, location, source, created_at, converted

### `interactions.csv` (~40,000 rows)
Detailed behavioral logs with:
- interaction_id, lead_id, session_id, timestamp, page_name, event_type, event_name, duration_seconds, scroll_depth, funnel_stage, utm_source, utm_campaign, device, browser, converted

---

## 🎯 Your Tasks

### ✅ Task 1: Exploratory Data Analysis

Analyze both datasets and document findings:

**Deliverable**: `analysis.md`

Answer:

- Data shape, types, missing values
- Top lead sources by conversion rate
- Which engagement metrics correlate with conversion?
- Any outliers or anomalies?
- Behavioral segments you discover
- Temporal patterns (seasonal trends)
- Key insights for product/marketing teams

**Time**: 2-4 hours
**Tools**: Pandas, Matplotlib, Seaborn

---

### ✅ Task 2: Feature Engineering & Model Building

Build a predictive model for lead conversion.

**Deliverable**: `train.py`

Requirements:
- Load and preprocess data
- Engineer features (session-level, lead-level, recency/frequency)
- Train/test split (80/20)
- Test multiple models (Logistic Regression, Random Forest, XGBoost, etc.)
- Evaluate using: Accuracy, Precision, Recall, F1, AUC-ROC
- Save best model as `model.pkl`

**Expected Output**:
```json
{
  "best_model": "XGBoost",
  "accuracy": 0.78,
  "precision": 0.75,
  "recall": 0.72,
  "f1": 0.73,
  "auc_roc": 0.82
}
```

**Time**: 4-6 hours
**Tools**: Scikit-learn, XGBoost, Pandas

---

### ✅ Task 3: API Development

Build a FastAPI service with two endpoints.

**Deliverable**: `app.py`

#### Endpoint 1: `/predict` (POST)

**Input**:
```json
{
  "pages_visited": 14,
  "time_spent_minutes": 45,
  "demo_requests": 1,
  "whatsapp_clicks": 2,
  "pricing_views": 3,
  "email_opens": 5,
  "session_count": 4,
  "days_since_first_visit": 12,
  "source": "Google",
  "company_size": "Medium"
}
```

**Output**:
```json
{
  "lead_id": "L00123",
  "conversion_probability": 0.84,
  "confidence": "high",
  "risk_level": "low"
}
```

#### Endpoint 2: `/explain` (POST)

**Input**:
```json
{
  "conversion_probability": 0.84,
  "pages_visited": 14,
  "demo_requests": 1,
  "pricing_views": 3
}
```

**Output**:
```json
{
  "summary": "This lead demonstrates high purchase intent through multiple pricing page visits and a demo request. Engagement metrics indicate strong product interest."
}
```

Can use:
- Simple rule-based logic
- OpenAI API
- Gemini API
- Claude API

**Time**: 3-4 hours
**Tools**: FastAPI, Pydantic, Pickle

---

### ✅ Task 4: Code Quality & Documentation

**Deliverable**: `README.md`

Must include:

- Project overview
- Setup instructions
- How to train the model
- How to run the API
- API endpoint documentation
- Example requests/responses
- Key findings and insights
- Limitations and future improvements

---

## 📁 Expected Repository Structure

```
your-repo/
├── README.md
├── TASK.md (this file)
├── analysis.md
├── requirements.txt
├── train.py
├── app.py
├── model.pkl
├── data/
│   ├── leads.csv
│   ├── interactions.csv
│   └── sample_leads.csv
├── outputs/
│   ├── model_metrics.json
│   └── feature_importance.png
└── .gitignore
```

---

## 📤 Submission Requirements

See [SUBMISSION_FORMAT.md](./SUBMISSION_FORMAT.md)

**Key Points**:
- Push to GitHub
- Include commit history (minimum 10+ commits showing progression)
- Do NOT commit data files (use .gitignore)
- Include detailed README
- All code must be runnable
- Save model artifacts

---

## 🧪 Evaluation Criteria

| Dimension | Weight | Details |
|-----------|--------|---------|
| **EDA & Insights** | 20% | Data understanding, finding hidden patterns, clear communication |
| **Model Quality** | 25% | Feature engineering, model selection, evaluation metrics, validation |
| **API Implementation** | 20% | Correctness, error handling, request/response format |
| **Code Quality** | 15% | Clean code, functions, reusability, comments |
| **Documentation** | 10% | README, comments, setup reproducibility |
| **Git Practices** | 10% | Commit history, meaningful messages, clean repo |

---

## ⚠️ Common Mistakes (Auto-Rejection)

❌ Zero commit history (appears AI-generated)
❌ No README or incomplete documentation
❌ Data leakage (future information in training)
❌ No train/test split
❌ Hardcoded model outputs
❌ API doesn't run without modifications
❌ No explanation of model choice
❌ Missing error handling
❌ Committing large data files to repo

---

## 🚀 Timeline

| Phase | Time | Tasks |
|-------|------|-------|
| **Setup** | 1 hour | Clone, setup venv, explore data |
| **EDA** | 3 hours | Analysis, visualizations, insights.md |
| **Model Building** | 5 hours | Feature engineering, training, evaluation |
| **API Development** | 3 hours | Endpoints, testing, error handling |
| **Documentation** | 2 hours | README, comments, finalization |
| **Buffer** | 29 hours | Debugging, improvements, refinement |

---

## 💡 Hints (No Spoilers)

- Session-level features (count, duration, gaps) are very predictive
- Recency/frequency metrics matter
- Not all high engagement means high conversion
- Some segments behave differently
- Temporal patterns exist
- A/B testing data might be hidden in the interactions
- Feature importance will surprise you

---

## 🎯 Interview Questions (After Submission)

Be prepared to answer:

1. Why did you choose your specific model?
2. How did you handle missing values and outliers?
3. What features were most important? Why?
4. Did you check for data leakage?
5. How would you deploy this to production?
6. How would you monitor model performance?
7. What if conversion rate changed dramatically?
8. How would you scale this to 10M leads?
9. What's a limitation of your approach?
10. How would you improve the model?

---

## 📚 Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Pandas](https://pandas.pydata.org/)

---

## ❓ Questions?

Contact: **hr@vynqe.com**

---

## 🏆 What Makes a Strong Submission

✅ Clear exploratory analysis with visualizations
✅ Feature engineering that makes business sense
✅ Model with good evaluation metrics and explanation
✅ Working API with proper error handling
✅ Professional GitHub repository with clean history
✅ Can explain every decision made
✅ Identifies limitations and proposes improvements

---

**Good luck! We're excited to see your work.** 🚀
