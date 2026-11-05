"""Settings and paths shared by the collection pipeline."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "youtube"

DATASET_NAME = "covid_vaccine_videos"

# Search settings
# 50 rather than 25 for the second dev pull: the exclusion list by then held 376
# already-seen ids, so the six mirror queries returned only 39 unseen videos at
# 25 each. Deeper paging on the same queries keeps the mirror faithful, where
# adding synonym queries would change the distribution it is meant to mirror.
MAX_RESULTS_PER_QUERY = 50
RELEVANCE_LANGUAGE = "en"
PUBLISHED_AFTER = None

MAX_DURATION_S = 3600

# The original pull, which produced the 124 transcribed videos now held out as
# the test set (data/labels/ground_truth.json). Do not re-run for a dev set.
SEARCH_QUERIES_V1: list[str] = [
    "covid vaccine side effects",
    "covid vaccine truth",
    "covid vaccine dangers",
    "covid vaccine mRNA risks",
    "covid vaccine microchip",
    "covid vaccine infertility",
]

# Second pull, for the first dev set: queries aimed at the four corpus topics,
# chosen because myocarditis and natural_immunity rested on 4 videos each in v1.
# Phrased as speakers phrase them, not clinically. See docs/routing-experiments.md.
SEARCH_QUERIES_V2: list[str] = [
    # myocarditis — 4 videos in v1
    "covid vaccine heart inflammation young men",
    "covid vaccine myocarditis athletes",
    "covid booster heart problems",
    # blood_clots — 6 videos in v1
    "covid vaccine blood clots",
    "astrazeneca vaccine clotting risk",
    "covid vaccine stroke risk",
    # natural_immunity — 4 videos in v1
    "natural immunity vs covid vaccine",
    "already had covid do i need the vaccine",
    "covid vaccine immunity wanes booster",
    # infertility — thinner by video than it looks, 10 videos but many duplicates
    "covid vaccine pregnancy safety",
    "covid vaccine menstrual cycle changes",
]

# Third pull, for the second dev set. Two groups: the six v1 queries verbatim, so
# the dev set matches the test set's query distribution, plus queries aimed at
# narratives the evidence corpus does not cover, so the web-search fallback is
# actually exercised. Rationale in docs/web-search.md.
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

# Which pull a collection run uses.
# Fourth pull. The topic queries from the first dev set, re-run deliberately to
# close two gaps the mirror-only set left: chunk size is barely discriminating on
# broad windows, and the thin topics had 1-2 videos each. Rationale and the
# measurements that motivated it are in docs/dev-sets.md.
SEARCH_QUERIES_V4_TOPIC: list[str] = list(SEARCH_QUERIES_V2)

SEARCH_QUERIES: list[str] = SEARCH_QUERIES_V4_TOPIC
