from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "spotify_artist_churn_dataset.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
FIGURE_DIR = BASE_DIR / "data" / "processed" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "churned_artist"

ORIGINAL_SELECTED_FEATURES = [
    "medium_rank_score",
    "historical_affinity_score",
    "historical_best_rank",
    "is_long_only",
    "rank_change_long_to_medium",
    "historical_avg_rank",
    "appears_medium_term",
]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required = ["user_id", "artist_id", "artist_name", TARGET_COL]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def run_pca_analysis(df: pd.DataFrame) -> dict:
    print("\n=== PCA ANALYSIS ===")

    X = df[ORIGINAL_SELECTED_FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    # Elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o"
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Elbow Plot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    elbow_path = FIGURE_DIR / "pca_elbow.png"
    plt.savefig(elbow_path, dpi=160)
    plt.close()

    # 2D scatter
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        X_pca2[:, 0],
        X_pca2[:, 1],
        c=y,
        alpha=0.7
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA Scatter Plot Colored by Churn Label")
    plt.tight_layout()
    scatter_path = FIGURE_DIR / "pca_scatter.png"
    plt.savefig(scatter_path, dpi=160)
    plt.close()

    # Choose number of components that explain at least 90% variance
    n_components_90 = int(np.argmax(cumulative_variance >= 0.90) + 1)

    pca_metrics = {
        "explained_variance_ratio": pca.explained_variance_ratio_.round(4).tolist(),
        "cumulative_explained_variance": cumulative_variance.round(4).tolist(),
        "n_components_90_variance": n_components_90,
        "pca_elbow_plot": str(elbow_path),
        "pca_scatter_plot": str(scatter_path),
        "pc1_pc2_variance": float(cumulative_variance[1]) if len(cumulative_variance) >= 2 else None,
    }

    with open(OUTPUT_DIR / "pca_metrics.json", "w", encoding="utf-8") as f:
        json.dump(pca_metrics, f, indent=4)

    print(f"PCA elbow plot saved to: {elbow_path}")
    print(f"PCA scatter plot saved to: {scatter_path}")
    print(f"Components for 90% variance: {n_components_90}")

    return pca_metrics


def build_network_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== NETWORK ANALYSIS ===")

    G = nx.Graph()

    # Bipartite graph:
    # user nodes connected to artist nodes if the artist appeared in the user's historical data.
    for _, row in df.iterrows():
        user_node = f"user::{row['user_id']}"
        artist_node = f"artist::{row['artist_id']}"

        weight = row.get("historical_affinity_score", 1.0)
        if pd.isna(weight):
            weight = 1.0

        G.add_node(user_node, node_type="user")
        G.add_node(
            artist_node,
            node_type="artist",
            artist_name=row.get("artist_name", "unknown")
        )
        G.add_edge(user_node, artist_node, weight=float(weight))

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G, weight="weight")

    network_rows = []

    for _, row in df.iterrows():
        user_node = f"user::{row['user_id']}"
        artist_node = f"artist::{row['artist_id']}"

        network_rows.append({
            "user_id": row["user_id"],
            "artist_id": row["artist_id"],
            "user_degree_centrality": degree.get(user_node, 0.0),
            "user_betweenness_centrality": betweenness.get(user_node, 0.0),
            "user_pagerank": pagerank.get(user_node, 0.0),
            "artist_degree_centrality": degree.get(artist_node, 0.0),
            "artist_betweenness_centrality": betweenness.get(artist_node, 0.0),
            "artist_pagerank": pagerank.get(artist_node, 0.0),
        })

    network_df = pd.DataFrame(network_rows)

    df_network = df.merge(
        network_df,
        on=["user_id", "artist_id"],
        how="left"
    )

    network_path = OUTPUT_DIR / "spotify_artist_churn_dataset_with_network.csv"
    df_network.to_csv(network_path, index=False)

    # Graph visualization
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=0.35)

    node_colors = []
    node_sizes = []

    for node in G.nodes():
        node_type = G.nodes[node].get("node_type")
        if node_type == "user":
            node_colors.append(0)
            node_sizes.append(180)
        else:
            node_colors.append(1)
            node_sizes.append(70)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title("User-Artist Interaction Network")
    plt.axis("off")
    plt.tight_layout()

    graph_path = FIGURE_DIR / "network_graph.png"
    plt.savefig(graph_path, dpi=160)
    plt.close()

    print(f"Network dataset saved to: {network_path}")
    print(f"Network graph saved to: {graph_path}")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    return df_network


def evaluate_model(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    scores = cross_validate(
        clf,
        X,
        y,
        cv=cv,
        scoring=scoring,
        error_score="raise"
    )

    return {
        "setup": label,
        "accuracy": float(np.mean(scores["test_accuracy"])),
        "precision": float(np.mean(scores["test_precision"])),
        "recall": float(np.mean(scores["test_recall"])),
        "f1": float(np.mean(scores["test_f1"])),
    }


def compare_original_pca_network(df_network: pd.DataFrame) -> pd.DataFrame:
    print("\n=== MODEL COMPARISON ===")

    y = df_network[TARGET_COL].astype(int)

    # Setup 1: original selected features
    X_original = df_network[ORIGINAL_SELECTED_FEATURES].copy()

    # Setup 2: PCA components
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)

    pca = PCA(n_components=0.90)
    X_pca = pca.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(
        X_pca,
        columns=[f"pca_component_{i+1}" for i in range(X_pca.shape[1])]
    )

    # Setup 3: original + network features
    network_features = [
        "user_degree_centrality",
        "user_betweenness_centrality",
        "user_pagerank",
        "artist_degree_centrality",
        "artist_betweenness_centrality",
        "artist_pagerank",
    ]

    X_network = df_network[ORIGINAL_SELECTED_FEATURES + network_features].copy()

    results = [
        evaluate_model(X_original, y, "original_selected_features"),
        evaluate_model(X_pca_df, y, "pca_components_90_variance"),
        evaluate_model(X_network, y, "original_plus_network_features"),
    ]

    comparison_df = pd.DataFrame(results)
    comparison_path = OUTPUT_DIR / "model_comparison_unit9.csv"
    comparison_df.to_csv(comparison_path, index=False)

    with open(OUTPUT_DIR / "model_comparison_unit9.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(comparison_df.to_string(index=False))
    print(f"Model comparison saved to: {comparison_path}")

    return comparison_df


def updated_feature_selection_with_network(df_network: pd.DataFrame) -> pd.DataFrame:
    print("\n=== UPDATED FEATURE SELECTION WITH NETWORK FEATURES ===")

    candidate_features = [
        "medium_rank_score",
        "historical_affinity_score",
        "historical_best_rank",
        "is_long_only",
        "rank_change_long_to_medium",
        "historical_avg_rank",
        "appears_medium_term",
        "user_degree_centrality",
        "user_betweenness_centrality",
        "user_pagerank",
        "artist_degree_centrality",
        "artist_betweenness_centrality",
        "artist_pagerank",
    ]

    X = df_network[candidate_features].copy()
    y = df_network[TARGET_COL].astype(int)

    # Filter method: ANOVA F-score
    f_scores, p_values = f_classif(X, y)
    filter_rank = pd.Series(f_scores, index=candidate_features).rank(
        ascending=False,
        method="min"
    ).astype(int)

    # RFE
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced")
    rfe = RFE(estimator=logistic, n_features_to_select=7)
    rfe.fit(StandardScaler().fit_transform(X), y)
    rfe_selected = pd.Series(rfe.support_, index=candidate_features)

    # Decision Tree importance
    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    dt.fit(X, y)
    dt_rank = pd.Series(dt.feature_importances_, index=candidate_features).rank(
        ascending=False,
        method="min"
    ).astype(int)

    # Random Forest importance
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X, y)
    rf_rank = pd.Series(rf.feature_importances_, index=candidate_features).rank(
        ascending=False,
        method="min"
    ).astype(int)

    rows = []

    for feature in candidate_features:
        score = 0

        if filter_rank[feature] <= 7:
            score += 1
        if rfe_selected[feature]:
            score += 1
        if dt_rank[feature] <= 7:
            score += 1
        if rf_rank[feature] <= 7:
            score += 1

        rows.append({
            "feature": feature,
            "filter_anova_rank": int(filter_rank[feature]),
            "rfe_selected": bool(rfe_selected[feature]),
            "dt_rank": int(dt_rank[feature]),
            "rf_rank": int(rf_rank[feature]),
            "selection_score": score,
            "decision": "Keep" if score >= 2 else "Drop",
        })

    result = pd.DataFrame(rows).sort_values(
        by=["selection_score", "rf_rank"],
        ascending=[False, True]
    )

    output_path = OUTPUT_DIR / "feature_selection_comparison_unit9_network.csv"
    result.to_csv(output_path, index=False)

    print(result.to_string(index=False))
    print(f"Updated feature selection table saved to: {output_path}")

    return result


def main():
    df = load_dataset()

    run_pca_analysis(df)
    df_network = build_network_features(df)
    compare_original_pca_network(df_network)
    updated_feature_selection_with_network(df_network)

    print("\nUnit 9 extension analysis completed successfully.")


if __name__ == "__main__":
    main()