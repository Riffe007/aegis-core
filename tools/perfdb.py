#!/usr/bin/env python3
"""AEGIS perf DB: the shared spine every AEGIS project reads or writes.

    perfdb.py init                              create the schema
    perfdb.py ingest RESULTS.jsonl --manifest M  add one run's measurements
    perfdb.py winners                           recompute best kernel per (arch, shape)
    perfdb.py report [--arch 121]               % of cuBLAS per shape

Schema is frozen for v1. Forge, Fabric, Custos and SACFormer add their own
tables (plans, requests, train_runs) later; do not rename these columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("AEGIS_PERFDB", "results/aegis_perf.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  device_id TEXT NOT NULL,
  sm_arch INTEGER NOT NULL,
  gpu_name TEXT,
  cuda_ver INTEGER,
  driver_ver INTEGER,
  git_sha TEXT,
  power_mode TEXT,
  clocks_locked INTEGER,
  temp_start_c REAL,
  measured_bw_gbps REAL,
  started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kernels (
  kernel_id TEXT PRIMARY KEY,           -- sha1(op|impl|config)
  op TEXT NOT NULL,
  impl TEXT NOT NULL,
  config_json TEXT NOT NULL,
  regs INTEGER,
  smem_bytes INTEGER,
  local_bytes INTEGER,
  ptx_hash TEXT,
  sass_path TEXT
);
CREATE TABLE IF NOT EXISTS measurements (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  kernel_id TEXT NOT NULL REFERENCES kernels(kernel_id),
  dtype TEXT NOT NULL,
  m INTEGER, n INTEGER, k INTEGER,
  median_us REAL, p10_us REAL, p90_us REAL,
  tflops REAL, gbps REAL,
  reps INTEGER,
  checked INTEGER, pass INTEGER,
  max_abs_err REAL, rel_err REAL,
  power_w REAL
);
CREATE INDEX IF NOT EXISTS idx_meas_shape ON measurements(m, n, k, dtype);
CREATE TABLE IF NOT EXISTS winners (
  sm_arch INTEGER NOT NULL,
  op TEXT NOT NULL,
  dtype TEXT NOT NULL,
  shape_bucket TEXT NOT NULL,
  kernel_id TEXT NOT NULL REFERENCES kernels(kernel_id),
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  median_us REAL,
  pct_of_cublas REAL,
  PRIMARY KEY (sm_arch, op, dtype, shape_bucket)
);
"""


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    return con


def shape_bucket(m: int, n: int, k: int) -> str:
    """Exact shape for v1. Bucketing (e.g. M rounded to pow2) comes with the tuner."""
    return f"{m}x{n}x{k}"


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def power_mode() -> str | None:
    """Jetson: nvpmodel mode. Elsewhere: None."""
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True, stderr=subprocess.DEVNULL)
        return out.strip().splitlines()[0]
    except Exception:
        return None


def gpu_temp_c() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return float(out.strip().splitlines()[0])
    except Exception:
        pass
    # Jetson thermal zones (millidegrees); take the GPU zone if named, else max.
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            t = float((z / "temp").read_text()) / 1000.0
            name = (z / "type").read_text().strip().lower()
            if "gpu" in name:
                return t
            temps.append(t)
        except Exception:
            continue
    return max(temps) if temps else None


def kernel_id(op: str, impl: str, config: dict) -> str:
    key = f"{op}|{impl}|{json.dumps(config, sort_keys=True)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def cmd_init(args) -> None:
    connect(args.db).close()
    print(f"initialized {args.db}")


def cmd_ingest(args) -> None:
    rows = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    if not rows:
        sys.exit("no rows in results file")
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else {}
    first = rows[0]
    con = connect(args.db)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    con.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, manifest.get("device_id", socket.gethostname()), first["sm_arch"],
         first.get("device"), first.get("cuda_runtime"), first.get("cuda_driver"),
         git_sha(), power_mode(), None, args.temp_start if args.temp_start is not None else gpu_temp_c(),
         manifest.get("measured_d2d_bw_gbps"), datetime.now(timezone.utc).isoformat()))
    for r in rows:
        kid = kernel_id(r["op"], r["impl"], r["config"])
        con.execute(
            "INSERT OR IGNORE INTO kernels(kernel_id, op, impl, config_json, regs, smem_bytes, local_bytes) "
            "VALUES (?,?,?,?,?,?,?)",
            (kid, r["op"], r["impl"], json.dumps(r["config"], sort_keys=True),
             r.get("regs"), r.get("smem_bytes"), r.get("local_bytes")))
        s = r["shape"]
        con.execute(
            "INSERT INTO measurements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, kid, r["dtype"], s["M"], s["N"], s["K"], r["median_us"], r["p10_us"],
             r["p90_us"], r["tflops"], r["gbps"], r["reps"], int(r["checked"]), int(r["pass"]),
             r["max_abs_err"], r["rel_err"], r.get("power_w")))
    con.commit()
    print(f"run {run_id}: ingested {len(rows)} measurements")
    con.close()


def _latest_per_kernel_shape(con, arch: int | None):
    """Most recent measurement per (arch, kernel, shape); only checked+passing rows
    are eligible, except cuBLAS which is the reference itself."""
    q = """
    SELECT r.sm_arch, k.op, m.dtype, m.m, m.n, m.k, k.impl, k.kernel_id, m.run_id,
           m.median_us, m.tflops, k.regs, k.local_bytes, m.checked, m.pass
    FROM measurements m JOIN runs r USING(run_id) JOIN kernels k USING(kernel_id)
    WHERE (k.impl = 'cublas' OR (m.checked = 1 AND m.pass = 1))
    """ + ("AND r.sm_arch = ?" if arch else "") + " ORDER BY r.started_at"
    latest = {}
    for row in con.execute(q, (arch,) if arch else ()):
        key = (row[0], row[1], row[2], row[3], row[4], row[5], row[7])
        latest[key] = row  # later runs overwrite earlier ones
    return list(latest.values())


def cmd_winners(args) -> None:
    con = connect(args.db)
    rows = _latest_per_kernel_shape(con, args.arch)
    groups: dict = {}
    for r in rows:
        groups.setdefault((r[0], r[1], r[2], r[3], r[4], r[5]), []).append(r)
    n = 0
    for (arch, op, dtype, m, nn, k), rs in groups.items():
        cublas = [r for r in rs if r[6] == "cublas"]
        ours = [r for r in rs if r[6] != "cublas"]
        if not ours:
            continue
        best = min(ours, key=lambda r: r[9])
        pct = (cublas[0][9] / best[9] * 100.0) if cublas else None
        con.execute("INSERT OR REPLACE INTO winners VALUES (?,?,?,?,?,?,?,?)",
                    (arch, op, dtype, shape_bucket(m, nn, k), best[7], best[8], best[9], pct))
        n += 1
    con.commit()
    print(f"wrote {n} winners")


def cmd_report(args) -> None:
    con = connect(args.db)
    rows = _latest_per_kernel_shape(con, args.arch)
    table: dict = {}
    for r in rows:
        table.setdefault((r[0], r[3], r[4], r[5]), {})[r[6]] = r
    impls = sorted({r[6] for r in rows if r[6] != "cublas"})
    widths = [max(11, len(i)) for i in impls]  # full impl names, never truncated
    hdr = f"{'arch':>5} {'M':>6} {'N':>6} {'K':>6} {'cuBLAS us':>10} {'TFLOPS':>7} " + \
          " ".join(f"{i:>{w}}" for i, w in zip(impls, widths))
    print(hdr)
    print("-" * len(hdr))
    for (arch, m, n, k), byimpl in sorted(table.items()):
        cb = byimpl.get("cublas")
        cells = []
        for i, w in zip(impls, widths):
            r = byimpl.get(i)
            if r is None:
                cells.append(f"{'-':>{w}}")
            elif cb:
                cells.append(f"{cb[9] / r[9] * 100:>{w - 1}.1f}%")
            else:
                cells.append(f"{r[9]:>{w - 2}.1f}us")
        cbs = f"{cb[9]:>10.1f} {cb[10]:>7.2f}" if cb else f"{'-':>10} {'-':>7}"
        print(f"{arch:>5} {m:>6} {n:>6} {k:>6} {cbs} " + " ".join(cells))
    print("\nCells are % of cuBLAS speed (100% = parity, >100% = faster than cuBLAS).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    p = sub.add_parser("ingest")
    p.add_argument("results")
    p.add_argument("--manifest")
    p.add_argument("--run-id")
    p.add_argument("--temp-start", type=float)
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("winners")
    p.add_argument("--arch", type=int)
    p.set_defaults(fn=cmd_winners)
    p = sub.add_parser("report")
    p.add_argument("--arch", type=int)
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
