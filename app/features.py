import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

MISSING_RANK = 999
MAX_RANK = 50


def _extract_artist_rows(raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Converts one user's raw Spotify top artists JSON into user-artist rows.
    One row = one user + one artist.
    """

    user_id = raw_data["user_id"]
    top_artists = raw_data["top_artists"]

    artist_map: dict[str, dict[str, Any]] = {}

    for time_range in ["long_term", "medium_term", "short_term"]:
        items = top_artists.get(time_range, {}).get("items", [])

        for rank, artist in enumerate(items, start=1):
            artist_id = artist["id"]

            if artist_id not in artist_map:
                artist_map[artist_id] = {
                    "user_id": user_id,
                    "artist_id": artist_id,
                    "artist_name": artist.get("name"),
                    "long_rank": MISSING_RANK,
                    "medium_rank": MISSING_RANK,
                    "short_rank": MISSING_RANK,
                    "appears_long_term": 0,
                    "appears_medium_term": 0,
                    "appears_short_term": 0,
                }

            prefix = time_range.split("_")[0]
            artist_map[artist_id][f"{prefix}_rank"] = rank
            artist_map[artist_id][f"appears_{time_range}"] = 1

    rows = []

    for artist in artist_map.values():
        # Base del dataset:
        # usamos artistas que aparecen en long_term o medium_term.
        # Los artistas que aparecen solo en short_term son nuevos intereses,
        # no casos válidos para churn histórico.
        if not (artist["appears_long_term"] or artist["appears_medium_term"]):
            continue

        # Target:
        # si era relevante antes pero no aparece recientemente, lo marcamos como churn risk.
        artist["churned_artist"] = int(artist["appears_short_term"] == 0)

        # Features históricas, sin usar short_term directamente para evitar data leakage.
        artist["rank_change_long_to_medium"] = (
            artist["medium_rank"] - artist["long_rank"]
            if artist["appears_long_term"] and artist["appears_medium_term"]
            else MISSING_RANK
        )

        artist["historical_best_rank"] = min(
            artist["long_rank"],
            artist["medium_rank"],
        )

        historical_rank_sum = 0
        historical_presence_count = (
            artist["appears_long_term"] + artist["appears_medium_term"]
        )

        if artist["appears_long_term"]:
            historical_rank_sum += artist["long_rank"]

        if artist["appears_medium_term"]:
            historical_rank_sum += artist["medium_rank"]

        artist["historical_avg_rank"] = (
            historical_rank_sum / historical_presence_count
        )

        artist["historical_presence_count"] = historical_presence_count

        artist["is_long_only"] = int(
            artist["appears_long_term"] == 1
            and artist["appears_medium_term"] == 0
        )

        artist["is_medium_only"] = int(
            artist["appears_long_term"] == 0
            and artist["appears_medium_term"] == 1
        )

        artist["is_present_in_both_historical_windows"] = int(
            artist["appears_long_term"] == 1
            and artist["appears_medium_term"] == 1
        )

        # Scores normalizados:
        # rank 1 = score alto; rank 50 = score bajo; missing = 0.
        artist["long_rank_score"] = (
            (MAX_RANK + 1 - artist["long_rank"]) / MAX_RANK
            if artist["appears_long_term"]
            else 0
        )

        artist["medium_rank_score"] = (
            (MAX_RANK + 1 - artist["medium_rank"]) / MAX_RANK
            if artist["appears_medium_term"]
            else 0
        )

        artist["historical_affinity_score"] = (
            artist["long_rank_score"] + artist["medium_rank_score"]
        ) / historical_presence_count

        rows.append(artist)

    return rows


def build_dataset_from_raw(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    all_rows = []

    json_files = sorted(raw_dir.glob("top_artists_*.json"))

    if not json_files:
        raise FileNotFoundError(f"No raw JSON files found in {raw_dir}")

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        rows = _extract_artist_rows(raw_data)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise ValueError("Dataset is empty after processing raw files.")

    output_path = PROCESSED_DATA_DIR / "spotify_artist_churn_dataset.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    return df


if __name__ == "__main__":
    dataset = build_dataset_from_raw()

    print(dataset.head())
    print()
    print("Rows:", len(dataset))
    print("Columns:", list(dataset.columns))
    print()
    print("Class balance:")
    print(dataset["churned_artist"].value_counts(normalize=True))
    print()
    print("Feature variance:")
    feature_cols = [
        "long_rank",
        "medium_rank",
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
    print(dataset[feature_cols].var(numeric_only=True).sort_values())