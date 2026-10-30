import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def env_bool(name, default):
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

API_TOKEN = os.getenv("ASTERION_API_TOKEN", "local-development-token-change-me")
DB_PATH = Path(os.getenv("ASTERION_DB_PATH", "./data/hunterseeker.db"))
MODEL_DIR = Path(os.getenv("ASTERION_MODEL_DIR", "./artifacts"))
DRY_RUN = env_bool("ASTERION_DRY_RUN", True)
MIN_PROMOTION_F1 = float(os.getenv("ASTERION_MIN_PROMOTION_F1", "0.70"))
for p in (DB_PATH.parent, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)
