import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from app.spotify_auth import (
    build_login_url,
    exchange_code_for_token,
    generate_state,
    spotify_get,
)

app = FastAPI(title="Spotify Artist Churn Predictor")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "app" / "model.pkl"

STATE_STORE: set[str] = set()
CURRENT_TOKEN: dict | None = None


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model file not found. Run `python app/model.py` before starting the API."
        )

    return joblib.load(MODEL_PATH)


MODEL_BUNDLE = load_model_bundle()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.get("/")
def root():
    return {
        "message": "Spotify Artist Churn Predictor API",
        "available_endpoints": [
            "/health",
            "/login",
            "/callback",
            "/collect",
            "/features",
            "/predict",
        ],
    }


@app.get("/features")
def get_model_features():
    return {
        "features": MODEL_BUNDLE["feature_columns"],
        "target": "churned_artist",
        "meaning": "Predicts artist listening disengagement risk.",
        "note": "Send these exact feature names as JSON fields to /predict.",
    }


@app.post("/predict")
def predict_churn(payload: dict[str, Any] = Body(...)):
    model = MODEL_BUNDLE["model"]
    feature_columns = MODEL_BUNDLE["feature_columns"]

    missing_features = [col for col in feature_columns if col not in payload]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required model features.",
                "missing_features": missing_features,
                "expected_features": feature_columns,
            },
        )

    try:
        input_values = np.array(
            [[float(payload[col]) for col in feature_columns]],
            dtype=float,
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="All feature values must be numeric.",
        )

    prediction = int(model.predict(input_values)[0])
    probability = float(model.predict_proba(input_values)[0][1])

    return {
        "churned_artist": bool(prediction),
        "churn_probability": round(probability, 3),
        "used_features": feature_columns,
        "interpretation": (
            "The model predicts this user is at risk of losing interest in this artist."
            if prediction == 1
            else "The model predicts this artist is likely still retained for this user."
        ),
    }


@app.get("/login")
def login():
    state = generate_state()
    STATE_STORE.add(state)
    login_url = build_login_url(state)
    return RedirectResponse(login_url)


@app.get("/callback")
def callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
):
    global CURRENT_TOKEN

    if error:
        raise HTTPException(status_code=400, detail=f"Spotify authorization error: {error}")

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    if not state or state not in STATE_STORE:
        raise HTTPException(status_code=400, detail="Invalid or missing OAuth state")

    STATE_STORE.remove(state)

    try:
        CURRENT_TOKEN = exchange_code_for_token(code)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": "Spotify authorization successful",
        "next_step": "Open http://127.0.0.1:8000/collect to fetch top artists",
    }


@app.get("/collect")
def collect_top_artists():
    if CURRENT_TOKEN is None:
        raise HTTPException(
            status_code=401,
            detail="No Spotify token found. First open /login and authorize the app.",
        )

    access_token = CURRENT_TOKEN["access_token"]

    try:
        profile = spotify_get("/me", access_token)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    spotify_user_id = profile.get("account_id") or profile.get("id") or "unknown_user"
    anonymized_user_id = hashlib.sha256(spotify_user_id.encode("utf-8")).hexdigest()[:12]

    time_ranges = ["long_term", "medium_term", "short_term"]
    top_artists_by_range = {}

    for time_range in time_ranges:
        try:
            data = spotify_get(
                "/me/top/artists",
                access_token,
                params={
                    "time_range": time_range,
                    "limit": 50,
                    "offset": 0,
                },
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        top_artists_by_range[time_range] = data

    output = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "user_id": anonymized_user_id,
        "spotify_profile": {
            "display_name": profile.get("display_name"),
            "country": profile.get("country"),
            "product": profile.get("product"),
        },
        "top_artists": top_artists_by_range,
    }

    output_path = RAW_DATA_DIR / f"top_artists_{anonymized_user_id}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    return {
        "message": "Top artists collected successfully",
        "user_id": anonymized_user_id,
        "saved_to": str(output_path),
        "long_term_count": len(top_artists_by_range["long_term"].get("items", [])),
        "medium_term_count": len(top_artists_by_range["medium_term"].get("items", [])),
        "short_term_count": len(top_artists_by_range["short_term"].get("items", [])),
    }