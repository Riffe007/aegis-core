//! `roofline`: an SVG roofline per arch from the perf DB.
//!
//! Roofs come from measurements on the same device (CLAUDE.md rule 8):
//! include/aegis/arch/sm_<arch>.json (emit-arch): memory = dram_bandwidth_gbps,
//! compute = mma_peak.tflops. Nothing derived from them is stored.
//! Points: per shape, our winner and cublaslt_best (latest v2 cold rows), at
//! compulsory arithmetic intensity 2MNK / (2 (MK + KN + MN)) FLOP/B.
use std::path::Path;

use anyhow::{bail, Context, Result};
use crate::params::{self, ArchParams};
use plotters::prelude::*;
use rusqlite::Connection;

use crate::db;

struct Point {
    label: String,
    ai: f64,
    tflops: f64,
}

fn points(con: &Connection, arch: i64) -> Result<Vec<Point>> {
    let mut stmt = con.prepare(
        "SELECT w.shape_bucket, k.impl, w.median_us, \
                (SELECT m.median_us FROM measurements m JOIN kernels kc USING(kernel_id) JOIN runs r USING(run_id) \
                 WHERE kc.impl = 'cublaslt_best' AND r.sm_arch = w.sm_arch AND m.dtype = w.dtype \
                   AND m.protocol_version = w.protocol_version AND m.allow_busy = 0 \
                   AND (m.protocol_version = 1 OR m.cache_mode = 'cold') \
                   AND (m.m || 'x' || m.n || 'x' || m.k) = w.shape_bucket \
                 ORDER BY r.started_at DESC LIMIT 1) \
         FROM winners w JOIN kernels k USING(kernel_id) WHERE w.sm_arch = ? AND w.op = 'gemm'",
    )?;
    let mut out = Vec::new();
    for row in stmt.query_map([arch], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, f64>(2)?, r.get::<_, Option<f64>>(3)?))
    })? {
        let (bucket, impl_, ours_us, cublas_us) = row?;
        let d: Vec<f64> = bucket.split('x').map(str::parse).collect::<Result<_, _>>().context("shape_bucket")?;
        let (m, n, k) = (d[0], d[1], d[2]);
        let flops = 2.0 * m * n * k;
        let ai = flops / (2.0 * (m * k + k * n + m * n));
        out.push(Point { label: format!("{bucket} {impl_}"), ai, tflops: flops / (ours_us * 1e-6) / 1e12 });
        if let Some(c) = cublas_us {
            out.push(Point { label: format!("{bucket} cublaslt_best"), ai, tflops: flops / (c * 1e-6) / 1e12 });
        }
    }
    Ok(out)
}

pub fn cmd_roofline(db_path: &Path, arch: i64, out: &Path, arch_file: Option<&Path>) -> Result<()> {
    let con = db::connect(db_path)?;
    let p: ArchParams = params::load(arch_file.unwrap_or(&params::arch_file(arch)))?;
    let (bw_gbps, peak_tflops) = (p.bw_gbps, p.peak_tflops);
    let pts = points(&con, arch)?;
    if pts.is_empty() {
        bail!("no winners for sm_{arch}; run aegis-perf tune first");
    }
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let ridge = p.ridge(); // FLOP/B where the roofs meet
    let (x0, x1) = (0.1f64, 4096.0f64);
    let (y0, y1) = (0.01f64, peak_tflops * 2.0);

    let root = SVGBackend::new(out, (1000, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("AEGIS roofline, sm_{arch} (measured roofs)"), ("sans-serif", 22))
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(80)
        .build_cartesian_2d((x0..x1).log_scale(), (y0..y1).log_scale())?;
    chart
        .configure_mesh()
        .x_desc("arithmetic intensity (FLOP/byte, compulsory traffic)")
        .y_desc("TFLOPS")
        .bold_line_style(RGBColor(200, 200, 200))
        .light_line_style(RGBColor(240, 240, 240))
        .draw()?;

    let roof = |x: f64| (bw_gbps * x / 1e3).min(peak_tflops);
    let xs: Vec<f64> = (0..=200).map(|i| x0 * (x1 / x0).powf(i as f64 / 200.0)).collect();
    chart
        .draw_series(LineSeries::new(xs.iter().map(|&x| (x, roof(x))), BLACK.stroke_width(2)))?
        .label(format!("roof: {bw_gbps:.1} GB/s measured, {peak_tflops:.1} TFLOPS mma.sync measured, ridge {ridge:.0} FLOP/B"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(2)));

    let ours: Vec<&Point> = pts.iter().filter(|p| !p.label.ends_with(" cublaslt_best")).collect();
    let cub: Vec<&Point> = pts.iter().filter(|p| p.label.ends_with(" cublaslt_best")).collect();
    chart
        .draw_series(cub.iter().map(|p| Cross::new((p.ai, p.tflops), 5, RED.stroke_width(2))))?
        .label("cublaslt_best")
        .legend(|(x, y)| Cross::new((x + 10, y), 5, RED.stroke_width(2)));
    chart
        .draw_series(ours.iter().map(|p| Circle::new((p.ai, p.tflops), 4, BLUE.filled())))?
        .label("AEGIS winner")
        .legend(|(x, y)| Circle::new((x + 10, y), 4, BLUE.filled()));
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .draw()?;
    root.present()?;

    println!("wrote {} ({} points, ridge {ridge:.0} FLOP/B; roofs from {})", out.display(), pts.len(), p.path.display());
    for p in &pts {
        println!("  {:<44} AI {:7.1}  {:7.2} TFLOPS  {:5.1}% of roof", p.label, p.ai, p.tflops, p.tflops / roof(p.ai) * 100.0);
    }
    Ok(())
}
