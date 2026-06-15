import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_artist_churn_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "app" / "model.pkl"
FEATURES_PATH = PROJECT_ROOT / "app" / "model_features.json"
COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "feature_selection_comparison.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "model_metrics.json"


TARGET_COL = "churned_artist"

# No usamos short_rank ni appears_short_term porque definen el target.
# Eso sería data leakage.
FEATURE_COLS = [
    "appears_long_term",
    "appears_medium_term",
    "rank_change_long_to_medium",
    "historical_best_rank",
    "historical_avg_rank",
    "historical_presence_count",
    "is_long_only",
    "is_medium_only",
    "is_present_in_both_historical_windows",
    "long_rank_score",
    "medium_rank_score",
    "historical_affinity_score",
]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATASET_PATH)

    missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()

    return X, y


def run_feature_selection(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    comparison = pd.DataFrame({"feature": X.columns})

    # 1A. Variance Threshold
    variance_selector = VarianceThreshold(threshold=0.01)
    variance_selector.fit(X)
    comparison["variance"] = X.var().values
    comparison["filter_variance_keep"] = variance_selector.get_support()

    # 1B. ANOVA F-test
    k = min(5, X.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=k)
    anova_selector.fit(X, y)

    anova_scores = anova_selector.scores_
    comparison["anova_score"] = anova_scores
    comparison["filter_anova_rank"] = (
        pd.Series(anova_scores, index=X.columns)
        .rank(ascending=False, method="min")
        .astype(int)
        .values
    )
    comparison["filter_anova_top5"] = anova_selector.get_support()

    # 1C. Correlation redundancy check
    corr_matrix = X.corr().abs()
    high_corr_flags = []

    for feature in X.columns:
        other_corrs = corr_matrix.loc[feature].drop(feature)
        high_corr_flags.append(bool((other_corrs > 0.9).any()))

    comparison["has_high_correlation_gt_0_9"] = high_corr_flags

    # 2. RFE Wrapper with Logistic Regression
    rfe_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    rfe = RFE(
        estimator=rfe_pipeline.named_steps["model"],
        n_features_to_select=k,
    )

    # RFE needs scaled values because Logistic Regression is scale-sensitive.
    X_scaled = StandardScaler().fit_transform(X)
    rfe.fit(X_scaled, y)

    comparison["rfe_selected"] = rfe.support_
    comparison["rfe_rank"] = rfe.ranking_

    # 3. Decision Tree importance
    dt = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    dt.fit(X, y)

    dt_importances = dt.feature_importances_
    comparison["dt_importance"] = dt_importances
    comparison["dt_rank"] = (
        pd.Series(dt_importances, index=X.columns)
        .rank(ascending=False, method="min")
        .astype(int)
        .values
    )

    # 4. Random Forest importance
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X, y)

    rf_importances = rf.feature_importances_
    comparison["rf_importance"] = rf_importances
    comparison["rf_rank"] = (
        pd.Series(rf_importances, index=X.columns)
        .rank(ascending=False, method="min")
        .astype(int)
        .values
    )

    # Combined decision score.
    # Más puntos = más evidencia de que la feature importa.
    comparison["selection_score"] = 0
    comparison["selection_score"] += comparison["filter_variance_keep"].astype(int)
    comparison["selection_score"] += comparison["filter_anova_top5"].astype(int)
    comparison["selection_score"] += comparison["rfe_selected"].astype(int)
    comparison["selection_score"] += (comparison["dt_rank"] <= 5).astype(int)
    comparison["selection_score"] += (comparison["rf_rank"] <= 5).astype(int)

    comparison["decision"] = np.where(
        comparison["selection_score"] >= 3,
        "Keep",
        np.where(comparison["selection_score"] == 2, "Optional", "Drop"),
    )

    comparison = comparison.sort_values(
        by=["selection_score", "rf_importance", "anova_score"],
        ascending=[False, False, False],
    )

    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8")

    return comparison


def train_final_model(X: pd.DataFrame, y: pd.Series, comparison: pd.DataFrame) -> dict:
    selected_features = comparison.loc[
        comparison["decision"].isin(["Keep", "Optional"]),
        "feature",
    ].tolist()

    # Fallback defensivo por si el dataset es chico y la selección queda muy estricta.
    if len(selected_features) < 4:
        selected_features = (
            comparison.sort_values("rf_importance", ascending=False)["feature"]
            .head(6)
            .tolist()
        )

    X_selected = X[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)

    cv_metrics = {}

    if n_splits >= 2:
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

        scoring = ["accuracy", "precision", "recall", "f1"]

        cv_result = cross_validate(
            model,
            X_selected,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise",
        )

        cv_metrics = {
            metric: float(cv_result[f"test_{metric}"].mean())
            for metric in scoring
        }

    bundle = {
        "model": model,
        "feature_columns": selected_features,
    }

    joblib.dump(bundle, MODEL_PATH)

    with open(FEATURES_PATH, "w", encoding="utf-8") as file:
        json.dump(selected_features, file, indent=2)

    metrics = {
        "rows": int(len(X)),
        "target_distribution": y.value_counts(normalize=True).to_dict(),
        "selected_features": selected_features,
        "test_metrics": test_metrics,
        "cross_validation_metrics": cv_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


if __name__ == "__main__":
    X, y = load_dataset()

    print("Rows:", len(X))
    print("Features:", list(X.columns))
    print()
    print("Target balance:")
    print(y.value_counts(normalize=True))
    print()

    comparison = run_feature_selection(X, y)

    print("Feature selection comparison:")
    print(
        comparison[
            [
                "feature",
                "filter_anova_rank",
                "rfe_selected",
                "dt_rank",
                "rf_rank",
                "selection_score",
                "decision",
            ]
        ]
    )
    print()

    metrics = train_final_model(X, y, comparison)

    print("Selected features:")
    print(metrics["selected_features"])
    print()
    print("Test metrics:")
    print(metrics["test_metrics"])
    print()
    print("Cross-validation metrics:")
    print(metrics["cross_validation_metrics"])
    print()
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved feature comparison to: {COMPARISON_PATH}")