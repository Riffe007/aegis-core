//! Stated diagnoses as data (core-request #9): `docs/analysis/diagnoses.json`
//! (schema `aegis-diagnoses/1`) lists per-shape diagnoses Core states, e.g.
//! "below floor: launch-overhead bound" on the 896x896 decode shapes, each with
//! the doc section it comes from. Readers show a diagnosis only while it is in
//! that file, so revising or withdrawing one is an edit here, not in a viewer.
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// The file, relative to the repo root.
pub const DIAGNOSES_FILE: &str = "docs/analysis/diagnoses.json";
pub const DIAGNOSES_SCHEMA: &str = "aegis-diagnoses/1";

#[derive(Clone, Debug, PartialEq)]
pub struct Diagnosis {
    pub arch: i64,
    pub shape: (i64, i64, i64),
    /// Short label, e.g. "below floor: launch-overhead bound".
    pub label: String,
    /// The statement, as written in the source.
    pub statement: String,
    /// The doc it comes from (repo-relative) and the section heading in it.
    pub source_doc: String,
    pub source_heading: String,
}

/// Every diagnosis in `path` (a diagnoses.json).
pub fn load(path: &Path) -> Result<Vec<Diagnosis>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))?;
    if v["schema"] != DIAGNOSES_SCHEMA {
        bail!("{}: schema {}, expected {DIAGNOSES_SCHEMA}", path.display(), v["schema"]);
    }
    let mut out = Vec::new();
    for d in v["diagnoses"].as_array().context("diagnoses: not a list")? {
        let s = d["shape"].as_array().filter(|s| s.len() == 3).context("diagnosis shape: [M, N, K]")?;
        let dim = |i: usize| s[i].as_i64().context("diagnosis shape: integers");
        let text = |k: &str| d[k].as_str().map(str::to_string).with_context(|| format!("diagnosis {k}"));
        out.push(Diagnosis {
            arch: d["arch"].as_i64().context("diagnosis arch")?,
            shape: (dim(0)?, dim(1)?, dim(2)?),
            label: text("label")?,
            statement: text("statement")?,
            source_doc: d["source"]["doc"].as_str().context("diagnosis source.doc")?.to_string(),
            source_heading: d["source"]["heading"].as_str().context("diagnosis source.heading")?.to_string(),
        });
    }
    Ok(out)
}

/// The diagnoses of one arch and shape.
pub fn of(all: &[Diagnosis], arch: i64, shape: (i64, i64, i64)) -> Vec<&Diagnosis> {
    all.iter().filter(|d| d.arch == arch && d.shape == shape).collect()
}

/// `root`/docs/analysis/diagnoses.json.
pub fn diagnoses_path(root: &Path) -> PathBuf {
    root.join(DIAGNOSES_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed file parses, and every diagnosis cites a section that
    /// exists in its doc and contains its statement (whitespace-normalized).
    #[test]
    fn committed_diagnoses_cite_existing_sections() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let all = load(&diagnoses_path(&root)).unwrap();
        assert_eq!(of(&all, 121, (1, 896, 896)).len(), 1);
        assert_eq!(of(&all, 121, (16, 896, 896))[0].label, "below floor: launch-overhead bound");
        assert!(of(&all, 121, (1, 4864, 896)).is_empty());
        let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ").replace("**", "");
        for d in &all {
            let doc = std::fs::read_to_string(root.join(&d.source_doc)).unwrap();
            let start = doc.find(&format!("## {}", d.source_heading)).unwrap_or_else(|| panic!("{}: no section {}", d.source_doc, d.source_heading));
            let rest = &doc[start + 3..];
            let section = &rest[..rest.find("\n## ").unwrap_or(rest.len())];
            assert!(norm(section).contains(&norm(&d.statement)), "{:?}: statement not in its section", d.shape);
        }
    }
}
