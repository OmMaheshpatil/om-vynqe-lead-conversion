# Exploratory Data Analysis – Lead Conversion Prediction

## 1. Dataset Overview

| Dataset | Rows | Columns | Size |
|---------|------|---------|------|
| `leads.csv` | 2 045 (2 025 unique leads) | 21 | 443 KB |
| `interactions.csv` | 40 000 | 36 | 10.9 MB |

After deduplication and cleaning:
- **2 025 unique leads** (20 duplicate `lead_id` rows removed)
- **39 948 interactions** (52 future-dated rows removed)

---

## 2. Target Variable

The raw dataset does **not** contain a `converted` column. We derive it from `account_type`:

| account_type | Count | Converted Label |
|--------------|-------|-----------------|
| New Business | 1 283 | 0 |
| Existing Customer | 440 | 1 |
| Partner | 302 | 0 |

**Target distribution:** 440 positive (21.7%), 1 585 negative (78.3%)  
**Class imbalance ratio:** ~3.6:1 — moderate imbalance; stratified splitting and class-weight adjustments are warranted.

---

## 3. Lead Demographics

### 3.1 Source Channel

| Source | Total | Converted | Conv. Rate |
|--------|-------|-----------|------------|
| Google | 505 | 107 | 21.2% |
| LinkedIn | 402 | 89 | 22.1% |
| Facebook | 307 | 48 | 15.6% |
| Instagram | 203 | 16 | **7.9%** |
| Direct | 202 | 77 | **38.1%** |
| Referral | 163 | 50 | **30.7%** |
| Email Campaign | 140 | 35 | 25.0% |
| Organic Search | 103 | 18 | 17.5% |

**Key finding:** Direct traffic (38.1%) and Referrals (30.7%) convert at significantly higher rates, while Instagram (7.9%) trails far behind. This suggests brand-aware visitors or word-of-mouth leads have strongest intent.

### 3.2 Company Size

| Size | Total | Converted | Conv. Rate |
|------|-------|-----------|------------|
| Enterprise | 423 | 98 | 23.2% |
| Large | 606 | 135 | 22.3% |
| Medium | 595 | 133 | 22.4% |
| Small | 300 | 52 | **17.3%** |

Small companies convert at a notably lower rate. Enterprise, Large, and Medium are roughly comparable.

### 3.3 Lead Segment

| Segment | Total | Converted | Conv. Rate |
|---------|-------|-----------|------------|
| Enterprise | 447 | 105 | 23.5% |
| Mid-Market | 1 153 | 259 | 22.5% |
| Startup | 195 | 41 | 21.0% |
| SMB | 230 | 35 | **15.2%** |

SMB segment shows the weakest conversion at 15.2%.

### 3.4 Industry

| Industry | Total | Converted | Conv. Rate |
|----------|-------|-----------|------------|
| Healthcare | 152 | 44 | **28.9%** |
| Education | 143 | 38 | **26.6%** |
| Consulting | 216 | 50 | 23.1% |
| BFSI | 265 | 54 | 20.4% |
| Technology | 277 | 59 | 21.3% |
| SaaS | 276 | 60 | 21.7% |
| Real Estate | 147 | 26 | **17.7%** |
| Manufacturing | 200 | 37 | **18.5%** |

Healthcare and Education show markedly higher conversion rates.

### 3.5 Region

| Region | Total | Conv. Rate |
|--------|-------|------------|
| West | 1 239 | 22.1% |
| North | 162 | 22.8% |
| South | 531 | 21.1% |
| East | 93 | 18.3% |

Geographic differences are modest. West dominates in volume.

### 3.6 Device Type

| Device | Total | Conv. Rate |
|--------|-------|------------|
| Mobile | 1 338 | 22.7% |
| Desktop | 585 | 20.5% |
| Tablet | 102 | **15.7%** |

Mobile leads slightly outperform Desktop; Tablet lags behind.

---

## 4. Interaction Behaviour Analysis

### 4.1 Engagement Metrics (Converted vs Not Converted)

| Metric | Converted | Not Converted | Ratio |
|--------|-----------|---------------|-------|
| Time on page (s) | 150.5 | 145.9 | 1.03× |
| Scroll depth (%) | 55.2 | 55.4 | 1.00× |
| Click count | 6.24 | 6.07 | 1.03× |
| Session duration (s) | 913.0 | 857.2 | **1.07×** |
| Page depth | 3.63 | 3.42 | **1.06×** |
| Mouse activity | 0.64 | 0.64 | 1.01× |

Per-interaction differences are subtle, but session-level and aggregate features show stronger signals.

### 4.2 Session Volume

| Metric | Converted | Not Converted |
|--------|-----------|---------------|
| Mean sessions | 5.7 | 5.0 |
| Median sessions | 5 | 4 |

Converted leads have ~14% more sessions on average.

### 4.3 Funnel Stage Distribution

| Funnel Stage | Not Converted | Converted |
|--------------|---------------|-----------|
| Awareness | 10 500 | 3 016 |
| Consideration | 8 790 | 2 683 |
| Evaluation | 7 700 | 2 595 |
| Decision | 3 427 | 1 237 |

Converted leads have a higher proportion of **Decision** and **Evaluation** stage interactions (expected).

### 4.4 High-Signal Events

| Event | Not Converted | Converted | Conv. Share |
|-------|---------------|-----------|-------------|
| demo_request | 163 | 90 | **35.6%** |
| contact_form_submit | 86 | 52 | **37.7%** |
| free_trial_start | 77 | 52 | **40.3%** |
| pricing_page_view | 1 972 | 886 | **31.0%** |
| page_view | 20 009 | 6 253 | 23.8% |

**Key finding:** `free_trial_start`, `contact_form_submit`, and `demo_request` are the strongest intent signals — converted leads generate these events at nearly 2× the baseline rate.

### 4.5 Button Clicks

| Button | Not Converted | Converted | Conv. Share |
|--------|---------------|-----------|-------------|
| View Pricing | 3 890 | 1 399 | **26.4%** |
| Request Demo | 2 925 | 992 | **25.3%** |
| Contact Sales | 2 581 | 859 | **25.0%** |
| WhatsApp Us | 1 420 | 452 | **24.1%** |
| Download PDF | 3 772 | 1 041 | 21.6% |

Pricing, Demo, and Contact Sales buttons are disproportionately clicked by converting leads.

---

## 5. Missing Data Analysis

### Leads

| Column | Missing | % |
|--------|---------|---|
| city | 40 | 2.0% |
| browser | 101 | 5.0% |
| company_size | 101 | 5.0% |
| annual_revenue_band | 41 | 2.0% |

Missing data in leads is minimal and handled via imputation.

### Interactions

| Column | Missing | % |
|--------|---------|---|
| form_name / form_step | 32 177 | 80.4% |
| cta_type | 7 933 | 19.8% |
| utm_source | 7 565 | 18.9% |
| utm_medium | 7 747 | 19.4% |
| button_location | 6 696 | 16.7% |
| referrer_type | 6 517 | 16.3% |
| previous_session_gap_days | 6 261 | 15.7% |
| button_name | 3 864 | 9.7% |
| utm_campaign | 3 349 | 8.4% |
| browser | 1 904 | 4.8% |
| session_duration_seconds | 1 289 | 3.2% |
| page_name | 828 | 2.1% |

Form fields have ~80% missing (most interactions don't involve forms). UTM and referrer fields are conditionally missing (direct traffic). All handled via default fills in preprocessing.

---

## 6. Feature Engineering Summary

We engineer **38 features** at the lead level, grouped into categories:

### Behavioural Aggregations (14 features)
`session_count`, `total_interactions`, `total_time_spent`, `avg_time_per_page`, `avg_scroll_depth`, `max_scroll_depth`, `total_clicks`, `avg_page_depth`, `max_page_depth`, `unique_pages_visited`, `form_completed_count`, `avg_mouse_activity`, `is_return_visitor_flag`, `avg_session_gap_days`

### Intent Signal Counts (10 features)
`demo_requests`, `pricing_views`, `whatsapp_clicks`, `email_opens`, `document_downloads`, `webinar_registrations`, `free_trial_starts`, `contact_form_submits`, `case_study_views`, `blog_reads`

### Funnel Stage Features (3 features)
`max_funnel_stage`, `decision_stage_interactions`, `evaluation_stage_interactions`

### Encoded Categoricals (9 features)
`source_encoded`, `company_size_encoded`, `segment_encoded`, `region_encoded`, `device_type_encoded`, `industry_encoded`, `funding_stage_encoded`, `job_role_encoded`, `first_touch_channel_encoded`

### Firmographic (2 features)
`employee_count`, `company_age_years`

---

## 7. Feature Correlations with Target

Top positively correlated features:

| Feature | Correlation |
|---------|-------------|
| pricing_views | **0.123** |
| free_trial_starts | **0.096** |
| demo_requests | **0.093** |
| contact_form_submits | **0.087** |
| form_completed_count | **0.075** |
| max_funnel_stage | 0.070 |
| avg_page_depth | 0.063 |
| session_count | 0.062 |
| decision_stage_interactions | 0.062 |

> **Note:** `email_opens` has zero variance (no `email_open` events exist in the dataset) and will be dropped during training.

**Interpretation:** Intent-driven features (pricing views, trial starts, demo requests) are the strongest predictors, followed by engagement depth (funnel stage, page depth, session count).

---

## 8. Key Insights & Modelling Recommendations

1. **Intent features dominate:** `pricing_views`, `free_trial_starts`, and `demo_requests` are the top 3 predictors. Tree-based models can capture these non-linear relationships well.

2. **Class imbalance (3.6:1):** Use `class_weight='balanced'` or SMOTE during training. Evaluate with F1 and AUC-ROC rather than accuracy alone.

3. **Low individual correlations:** No single feature exceeds ρ = 0.13, suggesting the prediction problem benefits from ensemble methods that combine weak signals.

4. **Source channel matters:** Direct and Referral traffic convert at 2–5× the rate of Instagram. Source encoding captures meaningful signal.

5. **Firmographics are weak predictors:** `employee_count` (ρ = 0.018) and `company_age_years` (ρ = 0.033) add marginal value but are retained for model completeness.

6. **Drop `email_opens`:** Zero variance feature — no `email_open` events exist in the data.

7. **Model selection:** Given the tabular data with mixed feature types and moderate size, **XGBoost** and **LightGBM** are expected to perform best, with Logistic Regression as an interpretable baseline.
