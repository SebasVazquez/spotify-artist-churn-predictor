from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "spotify_artist_churn_dataset.csv"
MODEL_PATH = BASE_DIR / "app" / "model.pkl"
MODEL_FEATURES_PATH = BASE_DIR / "app" / "model_features.json"


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run python app/model.py first.")

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle.get("model")
        feature_columns = bundle.get("feature_columns")
    else:
        model = bundle
        feature_columns = None

    if feature_columns is None:
        with open(MODEL_FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_columns = json.load(f)

    return model, feature_columns


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Processed dataset not found. Run python app/features.py first.")

    return pd.read_csv(DATASET_PATH)


def get_user_churn_risk(user_id: str) -> dict:
    df = load_dataset()
    model, feature_columns = load_model_bundle()

    user_df = df[df["user_id"].astype(str) == str(user_id)].copy()

    if user_df.empty:
        return {
            "user_id": user_id,
            "found": False,
            "message": "User not found in processed dataset.",
            "max_churn_probability": None,
            "average_churn_probability": None,
        }

    X = user_df[feature_columns].copy()

    probabilities = model.predict_proba(X)[:, 1]

    return {
        "user_id": user_id,
        "found": True,
        "max_churn_probability": float(np.max(probabilities)),
        "average_churn_probability": float(np.mean(probabilities)),
        "candidate_artists": int(len(user_df)),
    }


def build_user_artist_matrix(df: pd.DataFrame):
    value_col = "historical_affinity_score"

    if value_col not in df.columns:
        value_col = "medium_rank_score"

    matrix_df = df.pivot_table(
        index="user_id",
        columns="artist_id",
        values=value_col,
        aggfunc="max",
        fill_value=0.0
    )

    return matrix_df


def recommend_for_user(user_id: str, top_n: int = 5, risk_threshold: float = 0.5) -> dict:
    df = load_dataset()
    risk = get_user_churn_risk(user_id)

    if not risk["found"]:
        return risk

    # Recommendation only for at-risk users.
    # We use max probability because this project predicts churn at user-artist level.
    if risk["max_churn_probability"] < risk_threshold:
        return {
            "user_id": user_id,
            "at_risk": False,
            "churn_probability": round(risk["max_churn_probability"], 3),
            "message": "User is not considered high risk. No retention recommendation returned.",
            "recommendations": [],
        }

    matrix_df = build_user_artist_matrix(df)

    if str(user_id) not in matrix_df.index.astype(str).tolist():
        return {
            "user_id": user_id,
            "at_risk": True,
            "message": "User not available in recommendation matrix.",
            "recommendations": [],
        }

    # Align index as string for lookup
    matrix_df.index = matrix_df.index.astype(str)

    user_item_matrix = matrix_df.values

    n_users, n_items = user_item_matrix.shape

    if n_users < 2 or n_items < 2:
        return {
            "user_id": user_id,
            "at_risk": True,
            "message": "Not enough users/items to build SVD recommendations.",
            "recommendations": [],
        }

    n_components = min(3, n_users - 1, n_items - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_latent = svd.fit_transform(user_item_matrix)
    item_latent = svd.components_

    predicted_scores = np.dot(user_latent, item_latent)

    user_idx = list(matrix_df.index).index(str(user_id))
    user_scores = predicted_scores[user_idx].copy()

    already_seen = user_item_matrix[user_idx] > 0
    user_scores[already_seen] = -999

    top_indices = np.argsort(user_scores)[::-1][:top_n]

    artist_lookup = (
        df[["artist_id", "artist_name"]]
        .drop_duplicates()
        .set_index("artist_id")["artist_name"]
        .to_dict()
    )

    artist_ids = matrix_df.columns.tolist()

    recommendations = []

    for idx in top_indices:
        artist_id = artist_ids[idx]
        score = user_scores[idx]

        if score <= -999:
            continue

        recommendations.append({
            "artist_id": artist_id,
            "artist_name": artist_lookup.get(artist_id, "Unknown Artist"),
            "recommendation_score": round(float(score), 4),
        })

    return {
        "user_id": user_id,
        "at_risk": True,
        "churn_probability": round(risk["max_churn_probability"], 3),
        "average_churn_probability": round(risk["average_churn_probability"], 3),
        "method": "SVD collaborative filtering on user-artist affinity matrix",
        "recommendations": recommendations,
    }