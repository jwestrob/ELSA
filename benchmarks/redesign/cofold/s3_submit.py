"""Budget-guarded Biohub submission for screen 3 (task budget: <= 2,500 items in total, <= 2,000 in any 24 h).

Every item this task sends is recorded in work/s3/biohub_ledger.json (UTC time, items, what). A submission is
refused when the total or the rolling 24-hour count would exceed the budget.
Usage:
  python s3_submit.py submit --jobs JOBS --out OUT [--limit N] --what LABEL
  python s3_submit.py raw    --jobs JOBS --out OUT --what LABEL       (payload verification, one item)
  python s3_submit.py status
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import biohub_cofold as bc  # noqa: E402

LEDGER = HERE / "work" / "s3" / "biohub_ledger.json"
MAX_TOTAL, MAX_24H = 2500, 2000


def load():
    return json.load(open(LEDGER)) if LEDGER.exists() else []


def counts(led, now=None):
    now = now or time.time()
    tot = sum(e["items"] for e in led)
    d24 = sum(e["items"] for e in led if now - e["epoch"] < 86400)
    return tot, d24


def record(n, what):
    led = load()
    led.append(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), epoch=time.time(), items=n, what=what))
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    json.dump(led, open(LEDGER, "w"), indent=1)


def guard(n):
    tot, d24 = counts(load())
    if tot + n > MAX_TOTAL or d24 + n > MAX_24H:
        sys.exit(f"REFUSED: {n} items would bring the total to {tot + n} (max {MAX_TOTAL}) and the 24-h count to "
                 f"{d24 + n} (max {MAX_24H})")
    print(f"budget ok: total {tot} -> {tot + n}; last 24 h {d24} -> {d24 + n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--jobs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int)
    s.add_argument("--what", required=True)
    s.add_argument("--model", default=bc.MODEL)
    r = sub.add_parser("raw")
    r.add_argument("--jobs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--what", required=True)
    r.add_argument("--model", default=bc.MODEL)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "status":
        led = load()
        tot, d24 = counts(led)
        print(json.dumps(dict(total=tot, last_24h=d24, entries=led), indent=1))
        return
    if a.cmd == "raw":
        guard(1)
        record(1, a.what)
        bc.cmd_raw(a)
        return
    st = bc.load_state(a.out)
    jobs = [j for j in bc.load_jobs(a.jobs) if j["fold_id"] not in st["tasks"]]
    n = min(len(jobs), a.limit) if a.limit else len(jobs)
    if n == 0:
        print("nothing to submit")
        return
    guard(n)
    try:
        bc.cmd_submit(a)
    finally:
        st2 = bc.load_state(a.out)
        sent = len(st2["tasks"]) - len(st["tasks"])
        record(sent, a.what)
        print(f"ledger: +{sent} ({a.what})", flush=True)


if __name__ == "__main__":
    main()
