#!/usr/bin/env python3
"""Run every registered GEMM impl over the shape set, then ingest into the perf DB.

    python3 tools/sweep.py                 # default shapes, all impls
    python3 tools/sweep.py --impls tiled cublas --quick

Writes results/<host>-<timestamp>.jsonl and ingests it with the device manifest.
Pause background training on this machine before running (see CLAUDE.md rule 6).
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
RESULTS = ROOT / "results"

# impl name -> reps. Slow tiers get fewer reps on big shapes (see max_flops).
IMPLS = {
    "cublas": {"reps": 50},
    "naive": {"reps": 5, "max_flops": 2 * 2048 ** 3},  # skip naive above 2048^3
    "tiled": {"reps": 20},
    "regblock_64x64x8_4x4": {"reps": 20},
    "regblock_128x128x8_8x8": {"reps": 20},
    "regblock_128x64x16_8x4": {"reps": 20},
    "mma_128x128x32_2x4": {"reps": 20},
    "mma_128x64x32_2x2": {"reps": 20},
    "mma_64x128x32_2x4": {"reps": 20},
    "mma_64x64x32_2x2": {"reps": 20},
}

# Square shapes for the roofline, plus Qwen2.5-0.5B projection shapes
# (hidden 896, MLP 4864) at decode (M=1, 16) and prefill (M=512).
SQUARE = [(s, s, s) for s in (256, 512, 1024, 2048, 4096)]
QWEN = [(m, n, k) for m in (1, 16, 512)
        for (n, k) in ((896, 896), (4864, 896), (896, 4864))]
QUICK = [(512, 512, 512), (1, 896, 896), (1024, 1024, 1024)]


def run_one(impl: str, m: int, n: int, k: int, reps: int) -> dict | None:
    cmd = [str(BUILD / "bench_gemm"), "--impl", impl, "--m", str(m), "--n", str(n),
           "--k", str(k), "--reps", str(reps), "--warmup", "3", "--check"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    if not line:
        print(f"  ! {impl} {m}x{n}x{k}: no output\n{p.stderr}", file=sys.stderr)
        return None
    row = json.loads(line)
    status = "ok" if row["pass"] else "FAIL"
    print(f"  {impl:>8} {m:>5}x{n:>5}x{k:>5}  {row['median_us']:>10.1f} us  "
          f"{row['tflops']:>7.2f} TFLOPS  {status}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", nargs="+", default=list(IMPLS))
    ap.add_argument("--quick", action="store_true", help="3 shapes, for iterating")
    args = ap.parse_args()

    if not (BUILD / "bench_gemm").exists():
        sys.exit("build/bench_gemm not found; run cmake --build build first")
    RESULTS.mkdir(exist_ok=True)
    host = socket.gethostname()
    manifest = RESULTS / f"manifest-{host}.json"
    if not manifest.exists():
        manifest.write_text(subprocess.check_output([str(BUILD / "aegis_manifest")], text=True))

    shapes = QUICK if args.quick else SQUARE + QWEN
    out = RESULTS / f"{host}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    failures = 0
    with out.open("w") as f:
        for (m, n, k) in shapes:
            for impl in args.impls:
                spec = IMPLS[impl]
                if 2 * m * n * k > spec.get("max_flops", float("inf")):
                    continue
                row = run_one(impl, m, n, k, spec["reps"])
                if row:
                    f.write(json.dumps(row) + "\n")
                    failures += 0 if row["pass"] else 1

    subprocess.run([sys.executable, str(ROOT / "tools/perfdb.py"), "ingest", str(out),
                    "--manifest", str(manifest)], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/perfdb.py"), "winners"], check=True)
    print(f"\nresults: {out}")
    if failures:
        sys.exit(f"{failures} correctness failures; failing rows are excluded from winners")


if __name__ == "__main__":
    main()
