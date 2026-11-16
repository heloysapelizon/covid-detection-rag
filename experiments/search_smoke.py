"""peek at what a single search query returns."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

QUERY = sys.argv[1] if len(sys.argv) > 1 else "covid vaccine side effects"
MAX_RESULTS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
RELEVANCE_LANGUAGE = "en"

api_key = os.getenv("YOUTUBE_API_KEY")
if not api_key:
    sys.exit("YOUTUBE_API_KEY not set — add it to .env")

youtube = build("youtube", "v3", developerKey=api_key)

print(f"query: {QUERY!r}  (max {MAX_RESULTS} results, 100 quota units)\n")

resp = youtube.search().list(
    q=QUERY,
    part="snippet",
    type="video",
    maxResults=MAX_RESULTS,
    relevanceLanguage=RELEVANCE_LANGUAGE,
).execute()

items = resp.get("items", [])
for item in items:
    sn = item["snippet"]
    vid = item["id"]["videoId"]
    print(f"  {vid}  {sn.get('title', '')[:70]}")
    print(f"      {sn.get('channelTitle', '')}  |  {sn.get('publishedAt', '')[:10]}")
    desc = (sn.get("description") or "").replace("\n", " ")
    print(f"      {desc[:100]}\n")

print(f"{len(items)} result(s).")
