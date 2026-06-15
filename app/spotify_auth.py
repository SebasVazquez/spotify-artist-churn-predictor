import os
import base64
import secrets
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SPOTIFY_SCOPE = os.getenv("SPOTIFY_SCOPE", "user-top-read user-read-private")

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE_URL = "https://api.spotify.com/v1"


def validate_env() -> None:
    missing = []
    for name, value in {
        "SPOTIFY_CLIENT_ID": SPOTIFY_CLIENT_ID,
        "SPOTIFY_CLIENT_SECRET": SPOTIFY_CLIENT_SECRET,
        "SPOTIFY_REDIRECT_URI": SPOTIFY_REDIRECT_URI,
    }.items():
        if not value:
            missing.append(name)

    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


def build_login_url(state: str) -> str:
    validate_env()

    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SPOTIFY_SCOPE,
        "state": state,
        "show_dialog": "true",
    }

    return f"{AUTH_URL}?{urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    validate_env()

    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
    }

    response = requests.post(TOKEN_URL, headers=headers, data=data, timeout=20)

    if response.status_code != 200:
        raise RuntimeError(
            f"Spotify token error {response.status_code}: {response.text}"
        )

    return response.json()


def spotify_get(endpoint: str, access_token: str, params: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        headers=headers,
        params=params,
        timeout=20,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Spotify API error {response.status_code}: {response.text}"
        )

    return response.json()


def generate_state() -> str:
    return secrets.token_urlsafe(16)