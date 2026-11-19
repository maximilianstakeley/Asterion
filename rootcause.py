
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1", "true", "yes", "on"
    }


ASTERION_ENV = os.getenv(
    "ASTERION_ENV",
    "development"
).strip().lower()

if ASTERION_ENV not in {"development", "production", "test"}:
    raise RuntimeError("Invalid ASTERION_ENV")

API_TOKEN = os.getenv("ASTERION_API_TOKEN")

if ASTERION_ENV == "production":
    if not API_TOKEN or len(API_TOKEN) < 32:
        raise RuntimeError(
            "Production requires ASTERION_API_TOKEN "
            "with at least 32 characters."
        )

elif ASTERION_ENV == "development":
    if not API_TOKEN:
        API_TOKEN = "local-development-token-change-me"

elif ASTERION_ENV == "test":
    if not API_TOKEN:
        API_TOKEN = "asterion-test-token-only"


DB_PATH = Path(
    os.getenv(
        "ASTERION_DB_PATH",
        "./data/asterion.db"
    )
).expanduser()

MODEL_DIR = Path(
    os.getenv(
        "ASTERION_MODEL_DIR",
        "./artifacts"
    )
).expanduser()

DRY_RUN = env_bool("ASTERION_DRY_RUN", True)

MIN_PROMOTION_F1 = float(
    os.getenv("ASTERION_MIN_PROMOTION_F1", "0.70")
)

if not 0.0 <= MIN_PROMOTION_F1 <= 1.0:
    raise ValueError("MIN_PROMOTION_F1 must be between 0 and 1")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
