
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
  power_w REAL,
  -- Timing protocol v2 (src/bench/harness.h). Rows from before it are v1.
  protocol_version INTEGER NOT NULL DEFAULT 1,
  batch_n INTEGER,             -- launches per timed window (v1: 1)
  timer_res_ns REAL,           -- measured cudaEvent resolution
  cache_mode TEXT,             -- 'cold' | 'warm' (v2); 'flush_per_launch' or NULL (v1)
  buffer_sets INTEGER,         -- K buffer sets rotated inside a cold window
  window_ms REAL,              -- median timed window
  allow_busy INTEGER NOT NULL DEFAULT 0,  -- idle guard found the GPU busy; --allow-busy ran anyway
  idle_util_pct REAL           -- idle guard's mean GPU utilization
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
  protocol_version INTEGER NOT NULL DEFAULT 1,  -- of the measurements it was picked from
  warm_median_us REAL,              -- the pick's cache_mode=warm median (v2), if measured
  pct_of_cublaslt_best REAL,        -- cublaslt_best median / pick median x 100 (v2)
  tie_method TEXT,                  -- 'paired' (bootstrap CI), 'paired_swap' (CI < 1: runner-up won), 'band' (p10-p90)
  ratio_ci_lo REAL,                 -- paired: runner-up / fastest, 95% CI of the median ratio
  ratio_ci_hi REAL,
  lt_ratio_ci_lo REAL,              -- paired: cublaslt_best / fastest, 95% CI
  lt_ratio_ci_hi REAL,
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
-- Paired tie tests (aegis_tune, protocol v2): `pairs` alternating windows of
-- a (the fastest cold config) and b; ratio = b / a per pair; 95% percentile
-- bootstrap CI of the median ratio. Tied when the CI contains 1.0.
CREATE TABLE IF NOT EXISTS paired (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  op TEXT NOT NULL,
  dtype TEXT NOT NULL,
  m INTEGER, n INTEGER, k INTEGER,
  a_impl TEXT NOT NULL,
  b_impl TEXT NOT NULL,
  role TEXT,                  -- 'runner_up' | 'cublaslt_best'
  pairs INTEGER,
  ratio_median REAL,
  ratio_ci_lo REAL,
  ratio_ci_hi REAL,
  tied INTEGER,
  bootstrap_resamples INTEGER,
  bootstrap_seed INTEGER,
  protocol_version INTEGER,
  cache_mode TEXT,
  allow_busy INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_paired_shape ON paired(run_id, m, n, k);
