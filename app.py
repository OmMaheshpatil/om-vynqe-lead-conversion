"""
app.py - FastAPI application for Lead Conversion Prediction.

Endpoints:
    GET  /                  - Health check / welcome
    POST /predict           - Predict conversion probability for a single lead
    POST /explain           - Predict with feature explanation

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

# App setup

app = FastAPI(
    title="Lead Conversion Prediction API",
    description=(
        "ML-powered API to predict whether a B2B lead will convert, "
        "based on behavioural and firmographic features."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup 

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
METRICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs", "model_metrics.json"
)

model_bundle: dict = {}
model_metrics: dict = {}


def load_model():
    """Load the trained model bundle from disk."""
    global model_bundle, model_metrics

    if not os.path.exists(MODEL_PATH):
        print(f"Warning: model.pkl not found at {MODEL_PATH}. Run train.py first.")
        model_bundle = {}
        return

    with open(MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            model_metrics = json.load(f)

    print(f"Model loaded: {model_bundle.get('model_name', 'unknown')}")
    print(f"Features: {len(model_bundle.get('feature_names', []))}")


@app.on_event("startup")
async def startup_event():
    load_model()


# Request / Response schemas 

class PredictionInput(BaseModel):
    """Input schema for /predict endpoint"""
    pages_visited: int = 1
    time_spent_minutes: float = 0.0
    demo_requests: int = 0
    pricing_views: int = 0
    whatsapp_clicks: int = 0
    email_opens: int = 0
    session_count: int = 1
    days_since_first_visit: int = 0
    source: str = "Direct"  # "Google", "Referral", "LinkedIn", etc.
    company_size: str = "Small"  # "Small", "Medium", "Enterprise", "Large"
    segment: Optional[str] = "Mid-Market"  # "Startup", "SMB", "Enterprise", "Mid-Market"

    model_config = {"json_schema_extra": {
        "examples": [{
            "pages_visited": 14,
            "time_spent_minutes": 45.0,
            "demo_requests": 1,
            "pricing_views": 3,
            "whatsapp_clicks": 2,
            "email_opens": 0,
            "session_count": 4,
            "days_since_first_visit": 12,
            "source": "Google",
            "company_size": "Medium",
            "segment": "Mid-Market"
        }]
    }}


class PredictionOutput(BaseModel):
    """Output schema for /predict endpoint"""
    conversion_probability: float
    confidence: str
    risk_level: str
    recommendation: str


class ExplanationInput(BaseModel):
    """Input schema for /explain endpoint"""
    conversion_probability: float
    demo_requests: int = 0
    pricing_views: int = 0
    session_count: int = 0
    email_opens: int = 0


class ExplanationOutput(BaseModel):
    """Output schema for /explain endpoint"""
    summary: str
    factors: List[str]


# Helper functions 

def prepare_features(input_data: PredictionInput) -> np.ndarray:
    """Convert input data to encoded feature vector for the model."""
    if not model_bundle:
        raise RuntimeError("Model bundle is not loaded.")

    encoders = model_bundle["label_encoders"]
    feature_names = model_bundle["feature_names"]

    # Safe encoding of categoricals using fitted label encoders
    def safe_encode(encoder_key: str, val: str) -> int:
        encoder = encoders.get(encoder_key)
        if encoder is None:
            return 0
        try:
            # Handle standard formatting
            val_clean = str(val).strip()
            # Try exact match first
            if val_clean in encoder.classes_:
                return int(encoder.transform([val_clean])[0])
            # Try case-insensitive matching
            for c in encoder.classes_:
                if c.lower() == val_clean.lower():
                    return int(encoder.transform([c])[0])
            # Fallback to mapping as unseen category or first class
            return 0
        except Exception:
            return 0

    source_encoded = safe_encode("source", input_data.source)
    size_encoded = safe_encode("company_size", input_data.company_size)
    segment_encoded = safe_encode("lead_segment", input_data.segment)

    # Reconstruct the feature vector in exact order expected by the model
    # Note: 'email_opens' might be dropped if it had zero variance during training
    raw_feats = {
        "pages_visited": input_data.pages_visited,
        "time_spent_minutes": input_data.time_spent_minutes,
        "demo_requests": input_data.demo_requests,
        "pricing_views": input_data.pricing_views,
        "whatsapp_clicks": input_data.whatsapp_clicks,
        "email_opens": input_data.email_opens,
        "session_count": input_data.session_count,
        "days_since_first_visit": input_data.days_since_first_visit,
        "source_encoded": source_encoded,
        "company_size_encoded": size_encoded,
        "segment_encoded": segment_encoded,
    }

    # Only include features the model was trained on
    features = [raw_feats[f] for f in feature_names]
    return np.array([features])


def get_confidence_level(probability: float) -> str:
    """Classify confidence based on probability."""
    if probability >= 0.75:
        return "high"
    elif probability >= 0.50:
        return "medium"
    else:
        return "low"


def get_risk_level(probability: float) -> str:
    """Classify risk level (inverse of conversion probability)."""
    if probability >= 0.70:
        return "low"
    elif probability >= 0.40:
        return "medium"
    else:
        return "high"


# Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Lead Conversion Prediction API is running",
        "endpoints": ["/predict", "/explain", "/docs"]
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict lead conversion probability based on behaviour and firmographics.
    """
    if not model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")

    try:
        features = prepare_features(input_data)
        model = model_bundle["model"]
        scaler = model_bundle["scaler"]
        use_scaled = model_bundle.get("use_scaled", False)

        # Scale if model expects scaled inputs (e.g. Logistic Regression)
        if use_scaled:
            features = scaler.transform(features)

        # Predict probability
        probability = float(model.predict_proba(features)[0][1])

        # Get threshold-based label
        threshold = model_bundle.get("optimal_threshold", 0.5)
        is_converted = probability >= threshold

        # Classify confidence and risk
        confidence = get_confidence_level(probability)
        risk = get_risk_level(probability)

        # Generate recommendation
        if is_converted:
            recommendation = "Ready for sales outreach (High Priority)"
        elif probability >= threshold * 0.7:
            recommendation = "Nurture with targeted content (Medium Priority)"
        else:
            recommendation = "Focus on product education (Low Priority)"

        return PredictionOutput(
            conversion_probability=round(probability, 2),
            confidence=confidence,
            risk_level=risk,
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain", response_model=ExplanationOutput)
async def explain(input_data: ExplanationInput):
    """
    Get human-readable explanation of conversion prediction.
    """
    try:
        factors = []

        # Analyze high-intent signals
        if input_data.demo_requests > 0:
            factors.append("Demo request submitted (high intent)")

        if input_data.pricing_views >= 3:
            factors.append("Multiple pricing page visits (strong interest)")
        elif input_data.pricing_views >= 1:
            factors.append("Pricing page visited (moderate interest)")

        if input_data.session_count >= 5:
            factors.append("Multiple sessions indicate strong engagement")
        elif input_data.session_count >= 2:
            factors.append("Return visitor showing continued interest")

        if input_data.email_opens >= 3:
            factors.append("High email engagement")

        # Build summary
        threshold = model_bundle.get("optimal_threshold", 0.5)
        prob = input_data.conversion_probability

        if prob >= threshold:
            if input_data.demo_requests > 0:
                summary = "This lead shows exceptional interest with a demo request and strong engagement across multiple sessions."
            else:
                summary = "This lead demonstrates high conversion likelihood through consistent engagement and pricing interest."
        elif prob >= threshold * 0.7:
            summary = "This lead shows moderate interest. Continued nurturing and engagement could improve conversion likelihood."
        else:
            summary = "This lead requires more engagement before conversion is likely. Focus on product education and use case alignment."

        # Add fallback factor if empty
        if not factors:
            factors.append("Lead engagement patterns are currently standard/low-risk")

        return ExplanationOutput(
            summary=summary,
            factors=factors
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


#Run service 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
