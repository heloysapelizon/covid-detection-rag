"""The Video record.

Its field order is the dataset's column order, and `transcribe` reads these names,
so changing them changes the dataset contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Video:
    video_id: str
    title: str
    description: str
    channel: str
    channel_id: str
    published_at: str
    query: str
    url: str = ""
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    duration_s: Optional[int] = None

    def __post_init__(self):
        if not self.url:
            self.url = f"https://www.youtube.com/watch?v={self.video_id}"
