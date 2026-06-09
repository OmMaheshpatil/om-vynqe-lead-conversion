"""
utils.py – Data loading, cleaning, feature engineering, and preprocessing
for the Vynqe Lead Conversion Prediction pipeline.

Key design decisions
--------------------
* The provided datasets do NOT contain a `converted` column.  We derive it
  robustly from `leads.account_type`:
      Existing Customer  →  1  (converted)
      everything else    →  0  (not converted)
* Column naming diverges from the sample solution (e.g. `lead_segment` vs
  `segment`, `time_on_page_seconds` vs `duration_seconds`).  We normalise
  names early so downstream code uses a consistent schema.
* All feature engineering is deterministic and fully reproducible.
"""

from __future__ import annotations

import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─── constants ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LEADS_PATH = os.path.join(DATA_DIR, "leads.csv")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions.csv")

TARGET_COL = "converted"

# Columns we never feed into a model (identifiers, free text, dates)
DROP_COLS = [
    "lead_id", "business_email", "created_at", "campaign",
    "city", "state", "interaction_id", "session_id", "visitor_id",
    "timestamp", "page_url",
]

# Features we engineer at the lead level
FEATURE_COLUMNS = [
    "session_count",
    "total_interactions",
    "total_time_spent",
    "avg_time_per_page",
    "avg_scroll_depth",
    "max_scroll_depth",
    "total_clicks",
    "avg_page_depth",
    "max_page_depth",
    "unique_pages_visited",
    "demo_requests",
    "pricing_views",
    "whatsapp_clicks",
    "email_opens",
    "document_downloads",
    "webinar_registrations",
    "free_trial_starts",
    "contact_form_submits",
    "case_study_views",
    "blog_reads",
    "form_completed_count",
    "avg_mouse_activity",
    "is_return_visitor_flag",
    "avg_session_gap_days",
    "max_funnel_stage",
    "decision_stage_interactions",
    "evaluation_stage_interactions",
    "source_encoded",
    "company_size_encoded",
    "segment_encoded",
    "region_encoded",
    "device_type_encoded",
    "industry_encoded",
    "funding_stage_encoded",
    "job_role_encoded",
    "first_touch_channel_encoded",
    "employee_count",
    "company_age_years",
]

# Funnel stage ordering for ordinal encoding
FUNNEL_ORDER = {
    "Awareness": 0,
    "Consideration": 1,
    "Evaluation": 2,
    "Decision": 3,
}

# ─── data loading ────────────────────────────────────────────────────────────

def load_raw_data(
    leads_path: str = LEADS_PATH,
    interactions_path: str = INTERACTIONS_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSVs and perform minimal type coercion."""
    leads = pd.read_csv(leads_path)
    interactions = pd.read_csv(interactions_path)

    # Parse dates
    leads["created_at"] = pd.to_datetime(leads["created_at"], errors="coerce")
    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"], errors="coerce"
    )

    return leads, interactions


# ─── target derivation ───────────────────────────────────────────────────────

def derive_target(leads: pd.DataFrame) -> pd.DataFrame:
    """
    If the `converted` column is missing, derive it from `account_type`.

    Logic:
        account_type == 'Existing Customer'  →  converted = 1
        otherwise                            →  converted = 0
    """
    if TARGET_COL in leads.columns:
        leads[TARGET_COL] = leads[TARGET_COL].astype(int)
        return leads

    if "account_type" not in leads.columns:
        raise ValueError(
            "Cannot derive target: neither 'converted' nor 'account_type' "
            "found in leads data."
        )

    leads[TARGET_COL] = (
        leads["account_type"].str.strip().str.lower() == "existing customer"
    ).astype(int)

    pos = leads[TARGET_COL].sum()
    neg = len(leads) - pos
    print(f"   Derived target from account_type: {pos} positive, {neg} negative")
    return leads


# ─── cleaning ────────────────────────────────────────────────────────────────

def clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Remove future timestamps and fill missing categorical values."""
    # Drop interactions with timestamps in the future
    now = pd.Timestamp.now()
    future_mask = interactions["timestamp"] > now
    n_future = future_mask.sum()
    if n_future > 0:
        interactions = interactions[~future_mask].copy()
        print(f"   Removed {n_future} future-dated interactions")

    # Fill missing categoricals with sensible defaults
    fill_map = {
        "utm_source": "direct",
        "utm_medium": "none",
        "utm_campaign": "none",
        "funnel_stage": "Awareness",
        "referrer_type": "unknown",
        "button_name": "none",
        "button_location": "none",
        "cta_type": "none",
        "form_name": "none",
        "traffic_source": "unknown",
    }
    for col, val in fill_map.items():
        if col in interactions.columns:
            interactions[col] = interactions[col].fillna(val)

    # Fill missing numerics with 0
    num_fills = {
        "event_value": 0,
        "form_step": 0,
        "session_duration_seconds": 0,
        "mouse_activity_score": 0,
        "previous_session_gap_days": 0,
    }
    for col, val in num_fills.items():
        if col in interactions.columns:
            interactions[col] = interactions[col].fillna(val)

    return interactions


def clean_leads(leads: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values and deduplicate leads data."""
    # Deduplicate leads – keep first occurrence
    n_before = len(leads)
    leads = leads.drop_duplicates(subset=["lead_id"], keep="first").copy()
    n_dropped = n_before - len(leads)
    if n_dropped > 0:
        print(f"   Removed {n_dropped} duplicate lead rows")

    fill_map = {
        "region": "unknown",
        "industry": "unknown",
        "funding_stage": "unknown",
        "employee_growth_band": "unknown",
        "annual_revenue_band": "unknown",
    }
    for col, val in fill_map.items():
        if col in leads.columns:
            leads[col] = leads[col].fillna(val)

    return leads


# ─── feature engineering ─────────────────────────────────────────────────────

def engineer_features(
    leads: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a rich feature matrix at the lead level by aggregating interaction
    behaviour and encoding lead-level categoricals.

    Returns a DataFrame indexed by lead_id with all FEATURE_COLUMNS present.
    """

    # --- Interaction aggregation ------------------------------------------------
    agg_features = interactions.groupby("lead_id").agg(
        session_count=("session_id", "nunique"),
        total_interactions=("interaction_id", "count"),
        total_time_spent=("time_on_page_seconds", "sum"),
        avg_time_per_page=("time_on_page_seconds", "mean"),
        avg_scroll_depth=("scroll_depth_percent", "mean"),
        max_scroll_depth=("scroll_depth_percent", "max"),
        total_clicks=("click_count", "sum"),
        avg_page_depth=("page_depth", "mean"),
        max_page_depth=("page_depth", "max"),
        unique_pages_visited=("page_name", "nunique"),
        form_completed_count=("form_completed", "sum"),
        avg_mouse_activity=("mouse_activity_score", "mean"),
        is_return_visitor_flag=("is_return_visitor", "max"),
        avg_session_gap_days=("previous_session_gap_days", "mean"),
    )

    # --- Event-based count features --------------------------------------------
    event_counts = _count_events(interactions)
    agg_features = agg_features.join(event_counts, how="left")

    # --- Funnel stage features -------------------------------------------------
    interactions["funnel_ordinal"] = interactions["funnel_stage"].map(FUNNEL_ORDER).fillna(0)
    funnel_feats = interactions.groupby("lead_id").agg(
        max_funnel_stage=("funnel_ordinal", "max"),
        decision_stage_interactions=("funnel_stage", lambda s: (s == "Decision").sum()),
        evaluation_stage_interactions=("funnel_stage", lambda s: (s == "Evaluation").sum()),
    )
    agg_features = agg_features.join(funnel_feats, how="left")

    # --- Merge with lead-level data --------------------------------------------
    df = leads[["lead_id"]].drop_duplicates().copy()
    df = df.merge(agg_features.reset_index(), on="lead_id", how="left")

    # Bring in lead-level fields
    lead_cols = [
        "lead_id", "source", "company_size", "lead_segment", "region",
        "device_type", "industry", "funding_stage", "job_role",
        "first_touch_channel", "employee_count", "company_age_years",
    ]
    existing_lead_cols = [c for c in lead_cols if c in leads.columns]
    lead_dedup = leads[existing_lead_cols].drop_duplicates(subset=["lead_id"], keep="first")
    df = df.merge(lead_dedup, on="lead_id", how="left")

    # --- Encode categoricals ---------------------------------------------------
    cat_encode_map = {
        "source": "source_encoded",
        "company_size": "company_size_encoded",
        "lead_segment": "segment_encoded",
        "region": "region_encoded",
        "device_type": "device_type_encoded",
        "industry": "industry_encoded",
        "funding_stage": "funding_stage_encoded",
        "job_role": "job_role_encoded",
        "first_touch_channel": "first_touch_channel_encoded",
    }
    label_encoders: dict[str, LabelEncoder] = {}
    for raw_col, enc_col in cat_encode_map.items():
        if raw_col in df.columns:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[raw_col].astype(str))
            label_encoders[raw_col] = le

    # Fill any remaining NaN with 0
    df = df.fillna(0)

    # Keep only feature columns (+ lead_id for joining target later)
    final_cols = ["lead_id"] + [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[final_cols]

    return df, label_encoders


def _count_events(interactions: pd.DataFrame) -> pd.DataFrame:
    """Count specific high-signal events per lead."""
    events = {
        "demo_requests": interactions["event_name"] == "demo_request",
        "pricing_views": interactions["event_name"] == "pricing_page_view",
        "whatsapp_clicks": interactions["button_name"] == "WhatsApp Us",
        "email_opens": interactions["event_name"] == "email_open",
        "document_downloads": interactions["event_name"] == "document_download",
        "webinar_registrations": interactions["event_name"] == "webinar_registration",
        "free_trial_starts": interactions["event_name"] == "free_trial_start",
        "contact_form_submits": interactions["event_name"] == "contact_form_submit",
        "case_study_views": interactions["event_name"] == "case_study_view",
        "blog_reads": interactions["event_name"] == "blog_read",
    }
    result = pd.DataFrame(index=interactions["lead_id"].unique())
    for feat_name, mask in events.items():
        result[feat_name] = interactions.loc[mask].groupby("lead_id").size()
    return result.fillna(0)


# ─── high-level pipeline function ────────────────────────────────────────────

def load_and_preprocess_data(
    leads_path: str = LEADS_PATH,
    interactions_path: str = INTERACTIONS_PATH,
) -> Tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """
    End-to-end data pipeline.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (lead-level).
    y : pd.Series
        Binary target.
    feature_names : list[str]
        Ordered list of feature column names used.
    label_encoders : dict
        Fitted LabelEncoders for categorical columns.
    """
    print("=" * 60)
    print("DATA LOADING & PREPROCESSING")
    print("=" * 60)

    # 1. Load
    print("\n1. Loading raw data...")
    leads, interactions = load_raw_data(leads_path, interactions_path)
    print(f"   Leads: {leads.shape[0]} rows, {leads.shape[1]} cols")
    print(f"   Interactions: {interactions.shape[0]} rows, {interactions.shape[1]} cols")

    # 2. Derive target
    print("\n2. Deriving target variable...")
    leads = derive_target(leads)
    print(f"   Target distribution: {leads[TARGET_COL].value_counts().to_dict()}")

    # 3. Clean
    print("\n3. Cleaning data...")
    interactions = clean_interactions(interactions)
    leads = clean_leads(leads)
    print(f"   After cleaning: {interactions.shape[0]} interactions")

    # 4. Feature engineering
    print("\n4. Engineering features...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, label_encoders = engineer_features(leads, interactions)
    feature_names = [c for c in X.columns if c != "lead_id"]
    print(f"   Created {len(feature_names)} features")

    # 5. Align target
    y = leads.set_index("lead_id").loc[X["lead_id"], TARGET_COL].values
    y = pd.Series(y, name=TARGET_COL)
    X = X.drop(columns=["lead_id"])

    print(f"\n   Final X shape: {X.shape}")
    print(f"   Final y distribution: {y.value_counts().to_dict()}")
    print("=" * 60)

    return X, y, feature_names, label_encoders


# ─── quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, features, encoders = load_and_preprocess_data()
    print(f"\nFeature columns ({len(features)}):")
    for f in features:
        print(f"  • {f}")
    print(f"\nLabel encoders: {list(encoders.keys())}")
    print(f"Target balance: {y.mean():.2%} positive")
