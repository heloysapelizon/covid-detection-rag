"""Everything that touches faster-whisper: loading a model and using it.

Model loading and transcription live together because they share one dependency
and are never used apart — a model exists only to be transcribed with, and
transcription is meaningless without one. Isolating them here means the rest of
the package imports without pulling in the inference stack.
"""
from __future__ import annotations

import time
from pathlib import Path

from faster_whisper import WhisperModel

from .config import (
    COMPUTE_TYPE,
    DEVICE,
    LANG_PROBABILITY_MIN,
    MODEL_SIZE,
    WORD_TIMESTAMPS,
)


def load_model(
    model_size: str = MODEL_SIZE,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
    verbose: bool = True,
) -> WhisperModel:
    """Load a faster-whisper model.

    The first load of a given size downloads its weights (~460 MB for "small")
    into ~/.cache/huggingface; later loads read that cache.
    """
    t0 = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    if verbose:
        print(f"Loaded '{model_size}' ({compute_type}, {device}) "
              f"in {time.time() - t0:.1f}s")
    return model


def transcribe_audio(
    model: WhisperModel,
    wav_path: Path,
    word_timestamps: bool = WORD_TIMESTAMPS,
) -> tuple[str, str, dict, list[dict]]:
    """Transcribe one WAV file.

    Returns (text, status, meta, segments). segments is a list of
    {"start", "end", "text"} in seconds (plus "words" when word_timestamps is set),
    preserving where each passage is spoken so a flagged claim can be traced back
    to the video. text is the segments joined, kept for convenience. meta carries the detected language and
    probability for auditing which videos were dropped and why.

    Language is detected before transcription, so non-target audio costs a
    detection pass rather than full inference.
    """
    try:
        segments, info = model.transcribe(
            str(wav_path),
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=word_timestamps,
        )
        meta = {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_s": round(info.duration, 1),
        }

        # The dataset is English-only: anything else is dropped here rather than
        # transcribed, so non-English audio costs only a detection pass.
        if info.language != "en":
            return "", f"skipped_language ({info.language})", meta, []
        if info.language_probability < LANG_PROBABILITY_MIN:
            return "", f"skipped_language_uncertain ({info.language_probability:.2f})", meta, []

        # segments is a generator; consuming it is what actually runs inference.
        out: list[dict] = []
        for seg in segments:
            piece = seg.text.strip()
            if not piece:
                continue
            entry = {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": piece}
            if word_timestamps and seg.words:
                entry["words"] = [
                    {"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()}
                    for w in seg.words
                ]
            out.append(entry)

        text = " ".join(s["text"] for s in out).strip()
        return text, ("ok" if text else "empty"), meta, out

    except Exception as e:
        return "", f"transcribe_error: {type(e).__name__}", {}, []
