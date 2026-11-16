"""Time N adjudications on real test-set windows, to compare machines."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import LLAMA_URL, USE_GRAMMAR, USE_JSON_SCHEMA  # noqa: E402
from rag.index import EvidenceIndex  # noqa: E402
from rag.llm import adjudicate  # noqa: E402
from rag.pipeline import segment_claims  # noqa: E402
from experiments.run_experiment import load_test_set  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
print(f"server {LLAMA_URL}  grammar={USE_GRAMMAR}  json_schema={USE_JSON_SCHEMA}")
index = EvidenceIndex.load()
windows = segment_claims(load_test_set()[0]["transcript"])[:n]
times = []
for i, w in enumerate(windows, 1):
    passages = index.search(w)
    t = time.time()
    r = adjudicate(w, passages)
    times.append(time.time() - t)
    print(f"  [{i}/{len(windows)}] {times[-1]:5.1f} s  {r['verdict']}")
mean = sum(times) / len(times)
print(f"\nmean {mean:.1f} s/call   (workstation RTX 2080: ~20 s/call)")
print(f"est. full run: ~{mean * 1588 / 3600:.1f} h corpus-only, "
      f"~{mean * 1.8 * 1588 / 3600:.1f} h with Tavily")
