"""Settings and paths shared by the collection pipeline."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube"

DATASET_NAME = "covid_vaccine_videos"

MAX_RESULTS_PER_QUERY = 50
RELEVANCE_LANGUAGE = "en"
PUBLISHED_AFTER = None

MAX_DURATION_S = 3600

SEARCH_QUERIES_V1: list[str] = [
    "covid vaccine side effects",
    "covid vaccine truth",
    "covid vaccine dangers",
    "covid vaccine mRNA risks",
    "covid vaccine microchip",
    "covid vaccine infertility",
]

SEARCH_QUERIES_V2: list[str] = [
    "covid vaccine heart inflammation young men",
    "covid vaccine myocarditis athletes",
    "covid booster heart problems",
    "covid vaccine blood clots",
    "astrazeneca vaccine clotting risk",
    "covid vaccine stroke risk",
    "natural immunity vs covid vaccine",
    "already had covid do i need the vaccine",
    "covid vaccine immunity wanes booster",
    "covid vaccine pregnancy safety",
    "covid vaccine menstrual cycle changes",
]

SEARCH_QUERIES_V3_MIRROR: list[str] = list(SEARCH_QUERIES_V1)

SEARCH_QUERIES_V3_UNCOVERED: list[str] = [
    "covid vaccine graphene oxide",
    "covid vaccine depopulation plandemic",
    "covid vaccine excess deaths",
    "covid vaccine turbo cancer",
    "covid vaccine 5G",
    "covid vaccine shedding",
    "ivermectin covid treatment suppressed",
]

SEARCH_QUERIES_V3: list[str] = (
    SEARCH_QUERIES_V3_MIRROR + SEARCH_QUERIES_V3_UNCOVERED
)

SEARCH_QUERIES_V4_TOPIC: list[str] = list(SEARCH_QUERIES_V2)

SEARCH_QUERIES: list[str] = SEARCH_QUERIES_V4_TOPIC
