//! aegis-perf as a library: the perf DB schema, ingest, winners, reports and
//! roofline math, shared by the `aegis-perf` CLI and by viewers (AEGIS Custos),
//! so no viewer duplicates the schema. Viewers open the DB with
//! [`db::connect_ro`] and read it through [`view`].
pub mod ab;
pub mod arch;
pub mod bars;
pub mod db;
pub mod gpujob;
pub mod params;
pub mod pyjson;
pub mod roofline;
pub mod sweep;
pub mod tune;
pub mod view;
pub mod watch;
pub mod regress;
pub mod calibrate;

/// The rusqlite this crate is built against, so viewers name the same `Connection`.
pub use rusqlite;
