//! Python `json.dumps(value, sort_keys=True)` formatting, byte for byte.
//!
//! The perf DB derives `kernel_id` from `json.dumps(config, sort_keys=True)`
//! (tools/perfdb.py), so rows ingested from Rust and Python must serialize
//! configs identically or the same kernel gets two ids.
use serde_json::Value;

pub fn dumps_sorted(v: &Value) -> String {
    let mut out = String::new();
    write(v, &mut out);
    out
}

fn write(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&py_float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            // serde_json's default Map is a BTreeMap: keys already in byte
            // order, which equals Python's code-point order for these keys.
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(k, out);
                out.push_str(": ");
                write(&m[k.as_str()], out);
            }
            out.push('}');
        }
    }
}

/// `ensure_ascii=True` string escaping, as Python's json does.
fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr(float)`: shortest round-trip digits, scientific notation
/// outside 1e-4 <= |x| < 1e16, exponent with sign and at least two digits.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let a = x.abs();
    if (1e-4..1e16).contains(&a) {
        let s = format!("{x}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        let s = format!("{x:e}"); // e.g. "1e-5", "1.5e16"
        let (mant, exp) = s.split_once('e').unwrap();
        let exp: i32 = exp.parse().unwrap();
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_python_json_dumps() {
        // Expected strings produced by: json.dumps(v, sort_keys=True)
        assert_eq!(dumps_sorted(&json!({"tile": 32, "block": [32, 32]})), r#"{"block": [32, 32], "tile": 32}"#);
        assert_eq!(
            dumps_sorted(&json!({"BM": 128, "smem": "pad8", "STAGES": 3})),
            r#"{"BM": 128, "STAGES": 3, "smem": "pad8"}"#
        );
        assert_eq!(dumps_sorted(&json!({"algo": "CUBLAS_GEMM_DEFAULT"})), r#"{"algo": "CUBLAS_GEMM_DEFAULT"}"#);
        assert_eq!(dumps_sorted(&json!({"x": 1.5, "y": null, "z": true})), r#"{"x": 1.5, "y": null, "z": true}"#);
    }

    #[test]
    fn float_repr_matches_python() {
        for (x, want) in [(1.0, "1.0"), (0.1, "0.1"), (1e-5, "1e-05"), (1.5e16, "1.5e+16"),
                          (123456.789, "123456.789"), (0.0001, "0.0001"), (1e16, "1e+16")] {
            assert_eq!(py_float_repr(x), want, "{x}");
        }
    }
}
