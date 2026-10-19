""" gathers call of YouTube Data API v3

    search.list  100 units per call (up to 50 results)
    videos.list    1 unit per call  (up to 50 ids)
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .config import (
    MAX_RESULTS_PER_QUERY,
    PROJECT_ROOT,
    PUBLISHED_AFTER,
    RELEVANCE_LANGUAGE,
)
from .models import Video

__all__ = [
    "HttpError",
    "build_client",
    "fetch_videos_by_id",
    "parse_duration",
    "search_videos",
]

# PT1H2M3S / PT4M13S / PT45S — every component is optional, and the whole time
# section is too: live streams report "P0D" with no "T" part at all.
_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


def parse_duration(iso: str | None) -> int | None:
    """Seconds from an ISO 8601 duration like "PT1H2M3S". None if unparseable.

    Live streams report "P0D", which yields 0 — callers treat that as unknown.
    """
    if not iso:
        return None
    m = _DURATION_RE.match(iso)
    if not m:
        return None
    parts = {k: int(v) for k, v in m.groupdict(default="0").items()}
    return (
        parts["days"] * 86400
        + parts["hours"] * 3600
        + parts["minutes"] * 60
        + parts["seconds"]
    )


def build_client(api_key: str | None = None):
    """Return an authenticated YouTube client.

    Reads YOUTUBE_API_KEY from .env in the project root unless a key is passed.
    """
    if api_key is None:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY not set — add it to .env in the project root.\n"
            "Create one at https://console.cloud.google.com/ "
            "(enable 'YouTube Data API v3' -> create an API key)."
        )
    return build("youtube", "v3", developerKey=api_key)


def search_videos(
    client,
    query: str,
    max_results: int = MAX_RESULTS_PER_QUERY,
) -> list[Video]:
    """Return candidate videos for one query. Costs 100 quota units.

    search.list returns truncated descriptions (~140 chars) and no statistics or
    duration; fetch_videos_by_id fills those in.
    """
    params = {
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": max(1, min(50, max_results)),
    }
    if RELEVANCE_LANGUAGE:
        params["relevanceLanguage"] = RELEVANCE_LANGUAGE
    if PUBLISHED_AFTER:
        params["publishedAfter"] = PUBLISHED_AFTER

    resp = client.search().list(**params).execute()
    videos = []
    for item in resp.get("items", []):
        sn = item["snippet"]
        videos.append(Video(
            video_id=item["id"]["videoId"],
            title=sn.get("title", ""),
            description=sn.get("description", ""),
            channel=sn.get("channelTitle", ""),
            channel_id=sn.get("channelId", ""),
            published_at=sn.get("publishedAt", ""),
            query=query,
        ))
    return videos


def fetch_videos_by_id(
    client,
    video_ids: list[str],
    query: str,
) -> tuple[list[Video], list[str]]:
    """Fetch full metadata for explicit video ids. 1 quota unit per 50 ids.

    Returns (videos, missing_ids). Unlike search.list this returns untruncated
    descriptions plus statistics and duration, so it is both the link-list path
    and the way search results get completed.

    `query` is stored as each record's provenance.
    """
    if not video_ids:
        return [], []

    out: list[Video] = []
    found: set[str] = set()

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        resp = client.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch),
        ).execute()
        for item in resp.get("items", []):
            sn = item["snippet"]
            st = item.get("statistics", {})
            cd = item.get("contentDetails", {})
            found.add(item["id"])
            out.append(Video(
                video_id=item["id"],
                title=sn.get("title", ""),
                description=sn.get("description", ""),
                channel=sn.get("channelTitle", ""),
                channel_id=sn.get("channelId", ""),
                published_at=sn.get("publishedAt", ""),
                query=query,
                view_count=int(st["viewCount"]) if "viewCount" in st else None,
                like_count=int(st["likeCount"]) if "likeCount" in st else None,
                comment_count=int(st["commentCount"]) if "commentCount" in st else None,
                duration_s=parse_duration(cd.get("duration")),
            ))
        time.sleep(0.2)

    missing = [v for v in video_ids if v not in found]
    return out, missing
