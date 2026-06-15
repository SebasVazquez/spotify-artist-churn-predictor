import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Spotify Artist Churn Predictor\n",
                "\n",
                "This notebook documents the exploratory analysis, feature generation, feature selection, and model evaluation for the Spotify Artist Churn Predictor project.\n",
                "\n",
                "The goal is to predict whether a Spotify user is losing interest in an artist by comparing long-term, medium-term, and short-term listening affinity."
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Project Context\n",
                "\n",
                "The project uses the Spotify Web API to collect each authorized user's top artists across three time ranges:\n",
                "\n",
                "- `long_term`: historical listening preference\n",
                "- `medium_term`: medium-term listening preference\n",
                "- `short_term`: recent listening preference\n",
                "\n",
                "Each row in the dataset represents a `user + artist` relationship.\n",
                "\n",
                "The churn label is defined as:\n",
                "\n",
                "`churned_artist = 1` if the artist appeared in `long_term` or `medium_term`, but does not appear in `short_term`.\n",
                "\n",
                "This represents artist listening disengagement risk, not subscription cancellation."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "PROJECT_ROOT = Path('..').resolve()\n",
                "DATASET_PATH = PROJECT_ROOT / 'data' / 'processed' / 'spotify_artist_churn_dataset.csv'\n",
                "COMPARISON_PATH = PROJECT_ROOT / 'data' / 'processed' / 'feature_selection_comparison.csv'\n",
                "METRICS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'model_metrics.json'\n",
                "\n",
                "df = pd.read_csv(DATASET_PATH)\n",
                "comparison = pd.read_csv(COMPARISON_PATH)\n",
                "\n",
                "with open(METRICS_PATH, 'r', encoding='utf-8') as file:\n",
                "    metrics = json.load(file)\n",
                "\n",
                "df.head()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Dataset Overview"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Rows:', len(df))\n",
                "print('Columns:', len(df.columns))\n",
                "print('\\nColumns:')\n",
                "print(df.columns.tolist())\n",
                "\n",
                "print('\\nUsers collected:', df['user_id'].nunique())\n",
                "print('Artists collected:', df['artist_id'].nunique())"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Churn Label Balance"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class_balance = df['churned_artist'].value_counts(normalize=True).rename('proportion')\n",
                "display(class_balance)\n",
                "\n",
                "df['churned_artist'].value_counts().sort_index().plot(kind='bar')\n",
                "plt.title('Class Distribution: Retained vs Churned Artist')\n",
                "plt.xlabel('churned_artist')\n",
                "plt.ylabel('Count')\n",
                "plt.show()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Feature Engineering\n",
                "\n",
                "The raw Spotify fields are not used directly as the final modeling features. Instead, rankings and appearances across time ranges are transformed into behavioral indicators.\n",
                "\n",
                "Important engineered features include:\n",
                "\n",
                "- `medium_rank_score`: normalized score based on medium-term artist rank.\n",
                "- `historical_affinity_score`: average historical strength of the artist for the user.\n",
                "- `historical_best_rank`: best rank achieved in long-term or medium-term windows.\n",
                "- `rank_change_long_to_medium`: change in rank between long-term and medium-term windows.\n",
                "- `is_long_only`: whether the artist only appeared in the long-term window.\n",
                "- `appears_medium_term`: whether the artist appeared in the medium-term window.\n",
                "\n",
                "`short_rank` and `appears_short_term` are not used as model features because they define the target label and would create data leakage."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_cols = [\n",
                "    'appears_long_term',\n",
                "    'appears_medium_term',\n",
                "    'rank_change_long_to_medium',\n",
                "    'historical_best_rank',\n",
                "    'historical_avg_rank',\n",
                "    'historical_presence_count',\n",
                "    'is_long_only',\n",
                "    'is_medium_only',\n",
                "    'is_present_in_both_historical_windows',\n",
                "    'long_rank_score',\n",
                "    'medium_rank_score',\n",
                "    'historical_affinity_score',\n",
                "]\n",
                "\n",
                "df[feature_cols].describe().T"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Exploratory Analysis"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df.groupby('churned_artist')[['medium_rank_score', 'historical_affinity_score', 'historical_best_rank', 'historical_avg_rank']].mean()"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df.boxplot(column='historical_affinity_score', by='churned_artist')\n",
                "plt.title('Historical Affinity Score by Churn Label')\n",
                "plt.suptitle('')\n",
                "plt.xlabel('churned_artist')\n",
                "plt.ylabel('historical_affinity_score')\n",
                "plt.show()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Feature Selection\n",
                "\n",
                "The project applies four required feature selection approaches:\n",
                "\n",
                "1. Filter methods: Variance Threshold, ANOVA F-test, and correlation check.\n",
                "2. Wrapper method: Recursive Feature Elimination using Logistic Regression.\n",
                "3. Decision Tree feature importance.\n",
                "4. Random Forest feature importance.\n",
                "\n",
                "The table below consolidates the rankings and final decision for each feature."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(comparison[[\n",
                "    'feature',\n",
                "    'filter_anova_rank',\n",
                "    'rfe_selected',\n",
                "    'dt_rank',\n",
                "    'rf_rank',\n",
                "    'selection_score',\n",
                "    'decision'\n",
                "]])"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "comparison.sort_values('rf_importance', ascending=False).plot(\n",
                "    x='feature',\n",
                "    y='rf_importance',\n",
                "    kind='bar',\n",
                "    legend=False\n",
                ")\n",
                "plt.title('Random Forest Feature Importance')\n",
                "plt.xlabel('Feature')\n",
                "plt.ylabel('Importance')\n",
                "plt.xticks(rotation=75, ha='right')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Final Selected Features"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Selected features:')\n",
                "for feature in metrics['selected_features']:\n",
                "    print('-', feature)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Model Evaluation\n",
                "\n",
                "The final model is a Random Forest classifier trained on the selected features. Random Forest was chosen because it is stable, handles non-linear relationships, and also provides feature importance values."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Rows:', metrics['rows'])\n",
                "print('\\nTarget distribution:')\n",
                "print(metrics['target_distribution'])\n",
                "\n",
                "print('\\nTest metrics:')\n",
                "print(metrics['test_metrics'])\n",
                "\n",
                "print('\\nCross-validation metrics:')\n",
                "print(metrics['cross_validation_metrics'])"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Interpretation: Which Features Matter Most and Why?\n",
                "\n",
                "The most important features are related to medium-term artist rank and historical affinity. This suggests that churn risk is strongly connected to whether the artist remained relevant in the user's more recent medium-term listening behavior.\n",
                "\n",
                "- `medium_rank_score` was the strongest feature across several methods. A high medium-term rank means the artist was still relevant recently, reducing churn risk.\n",
                "- `historical_affinity_score` captures how strongly the artist appeared in the user's historical listening profile.\n",
                "- `historical_best_rank` indicates how important the artist ever was to the user.\n",
                "- `is_long_only` helps detect artists that were historically relevant but disappeared from medium-term listening.\n",
                "- `rank_change_long_to_medium` captures decline or movement in user interest.\n",
                "\n",
                "Overall, the model shows that churn is not random: artists are more likely to be classified as churned when their presence weakens or disappears between long-term and medium-term listening windows."
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Retention Strategy\n",
                "\n",
                "If this model were used by a music streaming platform, it could support personalized retention actions. For example:\n",
                "\n",
                "- If an artist has high churn probability, the platform could recommend a new release from that artist.\n",
                "- If a user's affinity toward a genre is declining, the system could suggest similar artists or curated playlists.\n",
                "- If the user historically liked an artist but stopped listening recently, the platform could surface concert announcements, new albums, or related content.\n",
                "\n",
                "Ethically, the prediction should be treated as a probability, not as a confirmed fact. The model should support recommendations without manipulating or pressuring users."
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path("notebooks") / "eda_and_selection.ipynb"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(notebook, file, ensure_ascii=False, indent=2)

print(f"Notebook created at: {output_path}")