# Spotify Artist Churn Predictor

## Project Overview

This project is a Dockerized machine learning web application that predicts **artist listening disengagement risk** using data collected from the Spotify Web API.

The goal is to predict whether a Spotify user is likely to stop listening to an artist based on changes between long-term, medium-term, and short-term listening affinity.

This project was developed for the **Introduction to Data Science** final project.

---

## Problem Definition

Spotify does not directly provide a `churned_artist` label. Therefore, this project defines churn as a behavioral signal derived from user listening patterns.

Each row in the dataset represents:

```text
user + artist
```

The target variable is:

```text
churned_artist = 1
```

if the artist appeared in the user's `long_term` or `medium_term` top artists, but does **not** appear in the user's `short_term` top artists.

This means that the artist was historically relevant to the user, but is no longer present in recent listening behavior.

This project predicts **artist disengagement**, not Spotify subscription cancellation.

---

## Data Source

The data was collected using the **Spotify Web API**.

For each authorized user, the application collects top artists from three time ranges:

```text
/me/top/artists?time_range=long_term&limit=50
/me/top/artists?time_range=medium_term&limit=50
/me/top/artists?time_range=short_term&limit=50
```

The project uses OAuth authorization with the following Spotify scope:

```text
user-top-read
```

The collected users are anonymized before being saved.

---

## Final Dataset

The final dataset contains:

```text
Users: 5
Rows: 328
Target: churned_artist
```

Class balance:

```text
churned_artist = 1 → 55.79%
churned_artist = 0 → 44.21%
```

This balance is acceptable for binary classification and does not require major class imbalance correction.

---

## Feature Engineering

Raw Spotify API data is transformed into behavioral features. The model does not use artist names directly.

Generated features include:

| Feature                                 | Meaning                                                |
| --------------------------------------- | ------------------------------------------------------ |
| `long_rank`                             | Artist rank in long-term listening                     |
| `medium_rank`                           | Artist rank in medium-term listening                   |
| `short_rank`                            | Artist rank in short-term listening                    |
| `appears_long_term`                     | Whether the artist appears in long-term top artists    |
| `appears_medium_term`                   | Whether the artist appears in medium-term top artists  |
| `appears_short_term`                    | Whether the artist appears in short-term top artists   |
| `rank_change_long_to_medium`            | Change in rank from long-term to medium-term           |
| `historical_best_rank`                  | Best historical rank between long-term and medium-term |
| `historical_avg_rank`                   | Average historical rank                                |
| `historical_presence_count`             | Number of historical windows where the artist appears  |
| `is_long_only`                          | Artist appears only in long-term history               |
| `is_medium_only`                        | Artist appears only in medium-term history             |
| `is_present_in_both_historical_windows` | Artist appears in both long-term and medium-term       |
| `long_rank_score`                       | Normalized long-term rank score                        |
| `medium_rank_score`                     | Normalized medium-term rank score                      |
| `historical_affinity_score`             | Average historical affinity score                      |

Important note:

```text
short_rank and appears_short_term are not used as model features.
```

They define the target label and using them would cause data leakage.

---

## Feature Selection

The project applies four feature selection methods:

1. **Filter Methods**

   * Variance Threshold
   * ANOVA F-test
   * Correlation check

2. **Wrapper Method**

   * Recursive Feature Elimination with Logistic Regression

3. **Decision Tree Importance**

   * Feature importance from a single Decision Tree

4. **Random Forest Importance**

   * Feature importance from a Random Forest model

The final comparison table is saved at:

```text
data/processed/feature_selection_comparison.csv
```

---

## Final Selected Features

The final model uses the following selected features:

```text
medium_rank_score
historical_affinity_score
historical_best_rank
is_long_only
rank_change_long_to_medium
historical_avg_rank
appears_medium_term
```

These features were selected because they showed the strongest importance across the feature selection methods.

---

## Model

The final model is a **Random Forest Classifier**.

Random Forest was selected because:

* it handles non-linear relationships;
* it works well with mixed ranking and binary features;
* it provides feature importance scores;
* it is more stable than a single Decision Tree.

The trained model is saved as:

```text
app/model.pkl
```

The selected model features are saved as:

```text
app/model_features.json
```

---

## Model Performance

Final test metrics:

```text
Accuracy: 0.7317
Precision: 0.7609
Recall: 0.7609
F1-score: 0.7609
```

Cross-validation metrics:

```text
Accuracy: 0.6950
Precision: 0.7497
Recall: 0.7114
F1-score: 0.7193
```

The cross-validation F1-score is the most useful metric here because the project predicts a binary class and needs a balance between precision and recall.

---

## Project Structure

```text
spotify-churn-predictor/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── spotify_auth.py
│   ├── scraper.py
│   ├── features.py
│   ├── model.py
│   ├── model.pkl
│   └── model_features.json
├── notebooks/
│   └── eda_and_selection.ipynb
├── data/
│   ├── raw/
│   └── processed/
│       ├── spotify_artist_churn_dataset.csv
│       ├── feature_selection_comparison.csv
│       └── model_metrics.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8000/callback
SPOTIFY_SCOPE=user-top-read
```

Do not upload `.env` to GitHub.

---

## Running Locally

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the API locally:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Test health endpoint:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

---

## Running with Docker

Build and run the application:

```powershell
docker compose up --build
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Test the health endpoint:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## API Endpoints

### `GET /health`

Checks if the API is running and if the model is loaded.

Example:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

---

### `GET /features`

Returns the features expected by the model.

Example:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/features" | ConvertTo-Json -Depth 5
```

Example response:

```json
{
  "features": [
    "medium_rank_score",
    "historical_affinity_score",
    "historical_best_rank",
    "is_long_only",
    "rank_change_long_to_medium",
    "historical_avg_rank",
    "appears_medium_term"
  ],
  "target": "churned_artist",
  "meaning": "Predicts artist listening disengagement risk."
}
```

---

### `POST /predict`

Predicts whether a user is at risk of losing interest in an artist.

Example request:

```powershell
$body = @{
    medium_rank_score = 0.80
    historical_affinity_score = 0.85
    historical_best_rank = 5
    is_long_only = 0
    rank_change_long_to_medium = 10
    historical_avg_rank = 8
    appears_medium_term = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

Example response:

```json
{
  "churned_artist": false,
  "churn_probability": 0.234,
  "used_features": [
    "medium_rank_score",
    "historical_affinity_score",
    "historical_best_rank",
    "is_long_only",
    "rank_change_long_to_medium",
    "historical_avg_rank",
    "appears_medium_term"
  ],
  "interpretation": "The model predicts this artist is likely still retained for this user."
}
```

---

## Data Collection Flow

To collect data from a Spotify user:

1. Add the user to the Spotify Developer Dashboard allowlist if the app is in Development Mode.
2. Open:

```text
http://127.0.0.1:8000/login
```

3. The user authorizes the application.
4. Open:

```text
http://127.0.0.1:8000/collect
```

5. A raw JSON file is saved in:

```text
data/raw/
```

Each file represents one anonymized Spotify user.

---

## Rebuilding the Dataset

After collecting new raw JSON files, regenerate the processed dataset:

```powershell
python app/features.py
```

This creates:

```text
data/processed/spotify_artist_churn_dataset.csv
```

---

## Training the Model

Train the model and run feature selection:

```powershell
python app/model.py
```

This creates or updates:

```text
app/model.pkl
app/model_features.json
data/processed/feature_selection_comparison.csv
data/processed/model_metrics.json
```

After retraining, rebuild Docker:

```powershell
docker compose down
docker compose up --build
```

---

## Notebook

The notebook is located at:

```text
notebooks/eda_and_selection.ipynb
```

It includes:

* project overview;
* dataset overview;
* churn label definition;
* exploratory data analysis;
* feature engineering explanation;
* four feature selection methods;
* selected features;
* model metrics;
* business interpretation.

---

## Main Findings

The most important features were related to medium-term rank and historical affinity.

The strongest predictors were:

```text
medium_rank_score
historical_affinity_score
historical_best_rank
is_long_only
rank_change_long_to_medium
```

Interpretation:

* Artists that remain strong in the medium-term window are less likely to be churned.
* Artists that only appear in long-term listening history are more likely to represent past interest.
* A large change between long-term and medium-term rank may indicate declining affinity.
* Historical rank strength helps distinguish core artists from weak or occasional listening preferences.

Overall, the model suggests that artist churn is mainly associated with the weakening or disappearance of listening affinity across time windows.

---

## Ethical Considerations

The model predicts probability, not certainty.

A user classified as at risk of losing interest in an artist should not be treated as a confirmed churn case. Predictions should support recommendation systems in a transparent and non-manipulative way.

Possible ethical concerns include:

* over-personalization;
* manipulating recommendations to push content;
* misclassifying user preferences;
* using inferred behavior without clear user awareness.

The model should be used to improve recommendations, not to pressure users.

---

## Limitations

This project has several limitations:

1. The dataset contains only 5 authorized users.
2. The target label is engineered, not directly provided by Spotify.
3. Top artists are based on Spotify's calculated affinity, not raw play counts.
4. The model predicts artist disengagement, not subscription churn.
5. The results may not generalize to the broader Spotify population.

Despite these limitations, the project demonstrates the complete data science workflow: data collection, feature engineering, feature selection, model training, API deployment, and Dockerization.

---

## Author

Sebastian Vazquez - 5250382

## Course

Introduction to Data Science
