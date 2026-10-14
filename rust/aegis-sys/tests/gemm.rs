//! End to end through the safe wrapper: manifest, choice, and a GEMM checked
//! against a CPU float64 reference with the bench tolerance.
use aegis_sys::safe::{abi_version, device_manifest, Context, DeviceBuffer};

fn f16_bits(x: f32) -> u16 {
    // Round-to-nearest-even via f64 -> f16 is overkill here; test data are
    // multiples of 1/64 in [-1, 1], exactly representable in f16.
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if x == 0.0 {
        return sign;
    }
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    sign | ((exp as u16) << 10) | mant as u16
}

fn f16_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f64;
    sign * match exp {
        0 => mant * 2f64.powi(-24),
        31 => f64::INFINITY,
        e => (1024.0 + mant) * 2f64.powi(e - 25),
    }
}

#[test]
fn abi_and_manifest() {
    assert_eq!(abi_version(), 1000);
    let json = device_manifest(0).expect("manifest");
    assert!(json.contains("\"measured_d2d_bw_gbps\""), "{json}");
    assert!(device_manifest(-1).is_err());
}

#[test]
fn gemm_matches_reference() {
    let ctx = Context::new(0).expect("context");
    for &(m, n, k) in &[(127usize, 129usize, 65usize), (16, 896, 896), (64, 64, 64)] {
        let mut seed = 7u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((seed >> 16) % 129) as f32 / 64.0 - 1.0 // multiples of 1/64 in [-1, 1]
        };
        let a: Vec<u16> = (0..m * k).map(|_| f16_bits(next())).collect();
        let b: Vec<u16> = (0..k * n).map(|_| f16_bits(next())).collect();
        let da = DeviceBuffer::from_host(&ctx, &a).unwrap();
        let db = DeviceBuffer::from_host(&ctx, &b).unwrap();
        let mut dc = DeviceBuffer::zeroed(&ctx, m * n).unwrap();
        ctx.gemm_f16(&da, &db, &mut dc, m, n, k).unwrap();
        let c = dc.to_host().unwrap();

        let (mut err, mut refmax) = (0f64, 0f64);
        for i in 0..m {
            for j in 0..n {
                let r: f64 = (0..k).map(|p| f16_value(a[i * k + p]) * f16_value(b[p * n + j])).sum();
                err = err.max((f16_value(c[i * n + j]) - r).abs());
                refmax = refmax.max(r.abs());
            }
        }
        let rel = err / refmax.max(1e-12);
        let choice = ctx.gemm_choice(m as i32, n as i32, k as i32).unwrap();
        println!("{m}x{n}x{k} {choice} rel_err={rel:.3e}");
        assert!(rel <= 4e-3, "{m}x{n}x{k} via {choice}: rel_err {rel}");
    }
}

#[test]
fn rejects_bad_arguments() {
    let ctx = Context::new(0).unwrap();
    let a = DeviceBuffer::zeroed(&ctx, 4).unwrap();
    let b = DeviceBuffer::zeroed(&ctx, 4).unwrap();
    let mut c = DeviceBuffer::zeroed(&ctx, 4).unwrap();
    assert!(ctx.gemm_f16(&a, &b, &mut c, 4, 4, 4).is_err(), "buffers too small");
    assert!(ctx.gemm_f16(&a, &b, &mut c, 0, 2, 2).is_err(), "zero dimension");
    assert!(Context::new(9999).is_err());
}
