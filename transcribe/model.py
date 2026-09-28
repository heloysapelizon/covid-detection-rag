"""Load the Whisper model."""
from __future__ import annotations

import time

from faster_whisper import WhisperModel

from .config import COMPUTE_TYPE, DEVICE, MODEL_SIZE


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
