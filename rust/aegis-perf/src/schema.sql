
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
  tied INTEGER NOT NULL DEFAULT 0,  -- runner-up median inside the winner's p10-p90
  tie_set TEXT,                     -- JSON list of impls in the tie (incl. the pick)
  pick_reason TEXT,                 -- "fastest", or which tie-break decided
  PRIMARY KEY (sm_arch, op, dtype, shape_bucket)
);
CREATE TABLE IF NOT EXISTS microbench (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  probe TEXT NOT NULL,
  param_json TEXT NOT NULL,
  value REAL,
  unit TEXT NOT NULL,
  superseded INTEGER NOT NULL DEFAULT 0,  -- 1 once a later run re-measured this probe
  superseded_by TEXT                      -- run_id of that later run
);
CREATE INDEX IF NOT EXISTS idx_micro_probe ON microbench(probe);
