import json, sys, shutil, collections
p = sys.argv[1]; drop = "--drop-errors" in sys.argv
d = json.load(open(p, encoding="utf-8")); vids = d["videos"]
c = collections.Counter(w["verdict"] for r in vids.values() for w in r["results"])
win = sum(r["windows"] for r in vids.values())
bad = [v for v, r in vids.items() if any(w["verdict"] == "ERROR" for w in r["results"])]
print(f"{len(vids)} videos, {win} windows, verdicts: {dict(c)}")
print(f"{len(bad)} video(s) with ERROR windows")
for v in bad[:5]:
    e = next(w for w in vids[v]["results"] if w["verdict"] == "ERROR")
    print(f"  {v}: {e['reason'][:150]}")
m = d.get("run", {}).get("minutes")
if m and win:
    s = m * 60 / win
    print(f"{s:.1f} s/window -> ~{s*1588/3600:.1f} h corpus-only, ~{s*1.35*1588/3600:.1f} h Tavily (valid for a non-resumed run)")
if drop and bad:
    shutil.copy(p, p + ".bak")
    for v in bad: del vids[v]
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"dropped {len(bad)} (backup {p}.bak); rerun the same command to redo them")
