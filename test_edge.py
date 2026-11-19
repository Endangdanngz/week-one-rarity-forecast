"""Edge-case tests for main.py (not run by the grader).

Builds hostile cases - empty or missing history, a single read, all-zero
rates, a rate at exactly 100, nulls sprinkled through the reads, a trophy
that is null everywhere, an unsorted history, truncated rate lists, missing
player counts, malformed reads, zero trophies - runs the real entry point
(`python3 -P -s main.py`) over them in a temp directory, and validates every
output file: one row per trophy, ids matching the case, five finite
quantiles in [0, 100], non-decreasing, at most 65,536 bytes.

Usage: python3 test_edge.py
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

BASE_TROPHIES = [
    {"id": 1, "name": "Vice Got a Price!", "description": "Congratulations! You're a Leonida Legend.",
     "tier": "platinum", "points": 300, "secret": False},
    {"id": 2, "name": "Just Like Paradise", "description": "Complete Gone Fishing.",
     "tier": "bronze", "points": 15, "secret": True},
    {"id": 3, "name": "Paradise Lost", "description": "Complete Chapter 1.",
     "tier": "bronze", "points": 15, "secret": True},
    {"id": 4, "name": "A Long Road", "description": "Attain 100% Completion.",
     "tier": "gold", "points": 90, "secret": False},
    {"id": 5, "name": "Cunning Stunts", "description": "Complete 250 Stunts.",
     "tier": "silver", "points": 30, "secret": False},
]


def read(at, players, rates):
    return {"slot": at, "at": at, "players_tracked": players,
            "trophies_earned": 100, "rates": rates}


def case(cid, history, trophies=None, horizon=6510):
    return {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": cid,
        "at": "2026-11-21T11:30:00Z",
        "horizon_minutes": horizon,
        "target": "2026-11-26T00:00:00Z",
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19", "source": "https://example.invalid/"},
        "trophies": BASE_TROPHIES if trophies is None else trophies,
        "history": history,
        "notes": "edge test",
    }


def build_cases():
    cases = {}
    cases["empty-history"] = case("empty-history", [])
    nohist = case("missing-history", [])
    del nohist["history"]
    cases["missing-history"] = nohist
    cases["one-read"] = case("one-read", [
        read("2026-11-19T00:30:00Z", 638, [0.63, 100, 1.25, 0.63, 0.63])])
    cases["all-zero"] = case("all-zero", [
        read("2026-11-19T00:30:00Z", 5000, [0, 0, 0, 0, 0]),
        read("2026-11-19T06:30:00Z", 9000, [0, 0, 0, 0, 0]),
        read("2026-11-20T06:30:00Z", 20000, [0, 0.1, 0, 0, 0])])
    cases["hundred-normal-branch"] = case("hundred-normal-branch", [
        read("2026-11-19T00:30:00Z", 638, [0.63, 100, 1.25, 0.63, 0.63]),
        read("2026-11-19T06:30:00Z", 3868, [0.31, 100, 0.85, 0.21, 0.18]),
        read("2026-11-20T06:30:00Z", 30000, [0.12, 99.9, 4.5, 0.1, 0.4]),
        read("2026-11-21T06:30:00Z", 55000, [0.08, 99.5, 9.8, 0.07, 0.7])])
    cases["nulls-sprinkled"] = case("nulls-sprinkled", [
        read("2026-11-19T00:30:00Z", 638, [0.63, None, 1.25, None, 0.63]),
        read("2026-11-19T06:30:00Z", 3868, [None, 98.0, None, None, None]),
        read("2026-11-20T06:30:00Z", 30000, [0.12, 99.0, 4.5, None, 0.4])])
    shuffled = [
        read("2026-11-20T06:30:00Z", 30000, [0.12, 99.9, 4.5, 0.1, 0.4]),
        read("2026-11-19T00:30:00Z", 638, [0.63, 100, 1.25, 0.63, 0.63]),
        read("2026-11-21T06:30:00Z", 55000, [0.08, 99.5, 9.8, 0.07, 0.7]),
        read("2026-11-19T06:30:00Z", 3868, [0.31, 100, 0.85, 0.21, 0.18]),
    ]
    cases["unsorted-history"] = case("unsorted-history", shuffled)
    cases["truncated-rates"] = case("truncated-rates", [
        read("2026-11-19T00:30:00Z", 638, [0.63, 100]),
        read("2026-11-20T06:30:00Z", 30000, [0.12]),
        read("2026-11-21T06:30:00Z", 55000, [])])
    noplayers = case("missing-players", [
        read("2026-11-19T00:30:00Z", None, [0.63, 100, 1.25, 0.63, 0.63]),
        read("2026-11-20T06:30:00Z", None, [0.12, 99.9, 4.5, 0.1, 0.4])])
    cases["missing-players"] = noplayers
    malformed = case("malformed-reads", [
        {"slot": "not-a-date", "at": "also-not-a-date", "rates": [1, 2, 3, 4, 5]},
        read("2026-11-19T00:30:00Z", 638, [0.63, 100, 1.25, 0.63, 0.63]),
        {"at": "2026-11-20T00:30:00Z"},
        read("2026-11-21T06:30:00Z", 55000, [0.08, 99.5, 9.8, 0.07, 0.7]),
    ])
    cases["malformed-reads"] = malformed
    cases["zero-trophies"] = case("zero-trophies", [], trophies=[])
    cases["dilution"] = case("dilution", [
        read("2026-11-19T00:30:00Z", 638, [0.63, 100, 1.25, 0.63, 0.63]),
        read("2026-11-19T12:30:00Z", 5000, [0.2, 99.0, 1.0, 0.2, 0.2]),
        read("2026-11-20T12:30:00Z", 40000, [0.05, 97.0, 0.9, 0.05, 0.1]),
        read("2026-11-21T06:30:00Z", 60000, [0.04, 96.0, 0.85, 0.04, 0.09])])
    return cases


def validate(case_obj, out_path):
    raw = out_path.read_bytes()
    assert len(raw) <= 65536, f"{out_path.name}: {len(raw)} bytes > 65536"
    out = json.loads(raw)
    assert set(out.keys()) == {"trophies"}, f"{out_path.name}: keys {set(out.keys())}"
    want_ids = [t["id"] for t in case_obj["trophies"]]
    got_ids = [r["id"] for r in out["trophies"]]
    assert sorted(got_ids) == sorted(want_ids) and len(got_ids) == len(want_ids), \
        f"{out_path.name}: ids {got_ids} != {want_ids}"
    for row in out["trophies"]:
        q = row["q"]
        assert len(q) == 5, f"{out_path.name}: row {row}"
        for x in q:
            assert isinstance(x, (int, float)) and math.isfinite(x) and 0.0 <= x <= 100.0, \
                f"{out_path.name}: bad quantile {x}"
        assert all(q[i] <= q[i + 1] for i in range(4)), f"{out_path.name}: not monotone {q}"
    return out


def main():
    cases = build_cases()
    with tempfile.TemporaryDirectory() as td:
        in_dir = Path(td) / "in"
        out_dir = Path(td) / "out"
        in_dir.mkdir()
        for cid, c in cases.items():
            (in_dir / f"{cid}.in").write_text(json.dumps(c), encoding="utf-8")
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(in_dir), MAKEFUN_OUTPUT_DIR=str(out_dir))
        proc = subprocess.run([sys.executable, "-P", "-s", str(HERE / "main.py")],
                              env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"entry crashed: {proc.stderr}"
        for cid, c in cases.items():
            out = validate(c, out_dir / f"{cid}.out")
            print(f"PASS {cid:<24} rows={len(out['trophies'])}")
        # Sanity: in the dilution case the platinum median must fall.
        dil = {r["id"]: r["q"] for r in validate(cases["dilution"], out_dir / "dilution.out")["trophies"]}
        assert dil[1][2] < 0.04 + 0.05, f"dilution platinum median {dil[1][2]}"
        print("PASS dilution direction (platinum median falls as the base grows)")
    print("ALL EDGE TESTS PASSED")


if __name__ == "__main__":
    main()
