"""Settings and paths shared by the collection pipeline."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube"

DATASET_NAME = "covid_vaccine_videos"

# Search settings
MAX_RESULTS_PER_QUERY = 25
RELEVANCE_LANGUAGE = "en"
PUBLISHED_AFTER = None

MAX_DURATION_S = 3600

SEARCH_QUERIES: list[str] = [
    "covid vaccine side effects",
    "covid vaccine truth",
    "covid vaccine dangers",
    "covid vaccine mRNA risks",
    "covid vaccine microchip", 
    "covid vaccine infertility",
]
