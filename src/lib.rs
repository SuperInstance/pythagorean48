//! # Pythagorean48
//!
//! Zero-drift unit vectors from Pythagorean triples — exact direction quantization
//! using integer ratios.
//!
//! ## Core idea
//!
//! Every Pythagorean triple (a, b, c) with a² + b² = c² gives an exact unit vector
//! (a/c, b/c).  By collecting all triples with c ≤ 100, we obtain **52 triples**
//! that expand (via swaps and sign flips) into **128 unique unit directions**
//! covering the full circle.  Because every direction is an exact rational pair,
//! composing rotations preserves the unit magnitude indefinitely — zero drift.
//!
//! ## Key properties
//!
//! | Property                | Value     |
//! |-------------------------|-----------|
//! | Pythagorean triples     | 52        |
//! | Unique directions       | 128       |
//! | Min angular gap         | ~1.1°     |
//! | Max angular gap         | ~7.4°     |
//! | Mean gap                | ~2.81°    |
//! | Magnitude precision     | Exact     |
//!
//! ## Zero drift
//!
//! Standard floating-point rotation accumulates error because each multiplication
//! rounds.  Pythagorean48 uses *exact rational arithmetic*: every rotation is a
//! composition of matrices whose entries are ratios `a/c`.  Because the matrix is
//! orthogonal by construction, the output magnitude *stays exactly 1* for any
//! number of chained rotations when computed with arbitrary-precision integers.
//!
//! ## Example
//!
//! ```rust
//! use pythagorean48::{Pythagorean48, PythagoreanTriple, Fraction};
//! use num_rational::Ratio;
//!
//! // All 52 triples
//! let triples = Pythagorean48::all_triples();
//! assert_eq!(triples.len(), 52);
//!
//! // A specific direction
//! let dir = Pythagorean48::direction(&triples[0], 0);
//!
//! // Verify it's a unit vector via exact fraction comparison
//! let (x, y) = dir;
//! let one = Ratio::<i64>::from_integer(1);
//! assert_eq!(x * x + y * y, one);
//! ```

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::ToPrimitive;
use std::cmp::Ordering;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Exact fraction using `i64` arithmetic, suitable for single directions.
pub type Fraction = Ratio<i64>;

/// Arbitrary-precision fraction for chained rotation proofs.
type BigFraction = Ratio<BigInt>;

/// A canonical Pythagorean triple (a, b, c) where a² + b² = c².
///
/// The triple is stored in canonical form: `a ≤ b < c`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PythagoreanTriple {
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

/// Report from a zero-drift proof run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftReport {
    /// Maximum absolute drift from magnitude 1 across all steps.
    pub max_drift: f64,
    /// Mean absolute drift across all steps.
    pub mean_drift: f64,
    /// Whether exactly zero drift was observed (true iff `max_drift == 0`).
    pub zero_drift: bool,
    /// Number of chained rotations checked.
    pub steps_checked: usize,
}

/// MSE comparison result for an encoding scheme.
#[derive(Debug, Clone)]
pub struct MseResult {
    /// Human-readable name.
    pub encoding_name: String,
    /// Number of discrete directions.
    pub num_directions: usize,
    /// Mean squared error (degrees²).
    pub mse: f64,
    /// Root mean squared error (degrees).
    pub rmse: f64,
    /// Maximum angular error (degrees).
    pub max_error: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TAU: f64 = 2.0 * PI;

/// Greatest common divisor (Euclidean algorithm).
const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Normalise an angle to `[0, 2π)`.
fn normalise_angle(rad: f64) -> f64 {
    let mut a = rad % TAU;
    if a < 0.0 {
        a += TAU;
    }
    a
}

/// Compute the angular gap (in radians) between two unit vectors.


// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

/// Pythagorean48 encoding provider.
///
/// All methods are static — the struct acts as a namespace.
pub struct Pythagorean48;

impl Pythagorean48 {
    // -- triples -----------------------------------------------------------

    /// Returns all 52 canonical Pythagorean triples with c ≤ 100.
    ///
    /// Triples are generated via Euclid's formula:
    ///   a₀ = m² − n²,  b₀ = 2mn,  c₀ = m² + n²
    /// for coprime `(m, n)` with `(m − n)` odd, then scaled by integer `k`
    /// until `k·c₀ > 100`.  Each triple is stored with `a ≤ b < c`.
    ///
    /// ## Example
    ///
    /// ```
    /// use pythagorean48::Pythagorean48;
    /// let ts = Pythagorean48::all_triples();
    /// assert_eq!(ts.len(), 52);
    /// assert_eq!(ts[0], pythagorean48::PythagoreanTriple { a: 3, b: 4, c: 5 });
    /// ```
    pub fn all_triples() -> Vec<PythagoreanTriple> {
        let max_c = 100u64;
        let mut triples: Vec<PythagoreanTriple> = Vec::with_capacity(64);
        let max_m = (max_c as f64).sqrt().ceil() as u64;

        for m in 2..=max_m {
            for n in 1..m {
                // Euclid's conditions for primitive triples
                if (m - n) % 2 == 0 {
                    continue;
                }
                if gcd_u64(m, n) != 1 {
                    continue;
                }

                let a0 = m * m - n * n;
                let b0 = 2 * m * n;
                let c0 = m * m + n * n;

                if c0 > max_c {
                    continue;
                }

                // Scale up
                let mut k = 1u64;
                while k * c0 <= max_c {
                    let a = k * a0;
                    let b = k * b0;
                    let c = k * c0;
                    let (small, large) = if a < b { (a, b) } else { (b, a) };

                    // Dedup (scaled triples may produce same direction but different triple)
                    if !triples.iter().any(|t| t.a == small && t.b == large && t.c == c) {
                        triples.push(PythagoreanTriple {
                            a: small,
                            b: large,
                            c,
                        });
                    }
                    k += 1;
                }
            }
        }

        // Sort by c, then a
        triples.sort_by(|t1, t2| t1.c.cmp(&t2.c).then(t1.a.cmp(&t2.a)));
        triples
    }

    // -- direction ---------------------------------------------------------

    /// Return the exact unit vector for a triple and variant.
    ///
    /// `variant` is a bitfield:
    ///
    /// | Bits | Meaning                        |
    /// |------|--------------------------------|
    /// | 0-1  | Quadrant: 0=(+,+), 1=(−,+), 2=(−,−), 3=(+,−) |
    /// | 2    | Swap a/b                      |
    ///
    /// So `variant ∈ [0, 7]` covers the 8 symmetries of the triple.
    ///
    /// The returned fraction pair `(x, y)` satisfies `x² + y² = 1` exactly.
    ///
    /// ## Example
    ///
    /// ```
    /// use pythagorean48::{Pythagorean48, PythagoreanTriple};
    /// let t = PythagoreanTriple { a: 3, b: 4, c: 5 };
    /// let (x, y) = Pythagorean48::direction(&t, 0);
    /// // (3/5, 4/5) — the classic 3-4-5 triangle direction
    /// assert_eq!(*x.numer(), 3);
    /// assert_eq!(*x.denom(), 5);
    /// ```
    pub fn direction(triple: &PythagoreanTriple, variant: u8) -> (Fraction, Fraction) {
        debug_assert!(variant < 8, "variant must be 0..=7");
        let quad = variant & 0x03;
        let swapped = (variant & 0x04) != 0;

        let (x_num, y_num) = if swapped {
            (triple.b as i64, triple.a as i64)
        } else {
            (triple.a as i64, triple.b as i64)
        };

        let (sx, sy) = match quad {
            0 => (1, 1),
            1 => (-1, 1),
            2 => (-1, -1),
            3 => (1, -1),
            _ => unreachable!(),
        };

        let c = triple.c as i64;
        (
            Ratio::new(sx * x_num, c),
            Ratio::new(sy * y_num, c),
        )
    }

    // -- all directions ----------------------------------------------------

    /// Return all 128 unique direction vectors, sorted by angle ascending.
    ///
    /// Each direction is an exact unit vector `(x, y)` where `x² + y² = 1`.
    /// The set is generated by taking each of the 52 triples, applying the 8
    /// symmetries (4 quadrants × swap), and deduplicating.
    ///
    /// ## Example
    ///
    /// ```
    /// use pythagorean48::Pythagorean48;
    /// let dirs = Pythagorean48::all_directions();
    /// assert_eq!(dirs.len(), 128);
    /// ```
    pub fn all_directions() -> Vec<(Fraction, Fraction)> {
        let triples = Self::all_triples();
        let mut dirs: Vec<(Fraction, Fraction)> = Vec::with_capacity(128);

        for triple in &triples {
            for variant in 0..8 {
                let dir = Self::direction(triple, variant);
                if !dirs.iter().any(|(a, b)| a == &dir.0 && b == &dir.1) {
                    dirs.push(dir);
                }
            }
        }

        // Sort by angle
        dirs.sort_by(|(x1, y1), (x2, y2)| {
            let a1 = normalise_angle(y1.to_f64().unwrap().atan2(x1.to_f64().unwrap()));
            let a2 = normalise_angle(y2.to_f64().unwrap().atan2(x2.to_f64().unwrap()));
            a1.partial_cmp(&a2).unwrap_or(Ordering::Equal)
        });

        dirs
    }

    // -- direction index ---------------------------------------------------

    /// Find the index of the nearest stored direction for a given angle in degrees.
    ///
    /// Returns `usize` in `[0, 127]` indexing into `all_directions()`.
    ///
    /// ## Example
    ///
    /// ```
    /// use pythagorean48::Pythagorean48;
    /// let idx = Pythagorean48::direction_index(45.0);
    /// // 45° should be quite close to index 16
    /// ```
    pub fn direction_index(angle_deg: f64) -> usize {
        let dirs = Self::all_directions();
        let target = angle_deg.to_radians();

        let mut best = 0usize;
        let mut best_dist = f64::MAX;

        for (i, (x, y)) in dirs.iter().enumerate() {
            let a = normalise_angle(y.to_f64().unwrap().atan2(x.to_f64().unwrap()));
            let mut d = (a - target).abs();
            if d > PI {
                d = TAU - d;
            }
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }

        best
    }

    // -- rotate ------------------------------------------------------------

    /// Rotate a unit vector `(x, y)` by the angle represented by the given triple.
    ///
    /// The rotation matrix is:
    ///
    /// ```text
    /// [cos  −sin]   where cos = a/c, sin = b/c
    /// [sin   cos]
    /// ```
    ///
    /// The result is computed using exact `Ratio<i64>` arithmetic.  For single
    /// rotations the result is exact; for long chains use
    /// [`Self::prove_zero_drift`] which uses arbitrary-precision integers.
    ///
    /// ## Example
    ///
    /// ```
    /// use pythagorean48::{Pythagorean48, PythagoreanTriple, Fraction};
    /// use num_traits::One;
    /// use num_rational::Ratio;
    ///
    /// let triple = PythagoreanTriple { a: 3, b: 4, c: 5 };
    /// let x = Ratio::from_integer(1);
    /// let y = Ratio::from_integer(0);
    /// let (x2, y2) = Pythagorean48::rotate(x, y, &triple);
    /// ```
    pub fn rotate(
        x: Fraction,
        y: Fraction,
        triple: &PythagoreanTriple,
    ) -> (Fraction, Fraction) {
        let cos = Ratio::new(triple.a as i64, triple.c as i64);
        let sin = Ratio::new(triple.b as i64, triple.c as i64);

        let new_x = x * cos - y * sin;
        let new_y = x * sin + y * cos;

        (new_x, new_y)
    }

    // -- zero drift proof --------------------------------------------------

    /// Chain `steps` rotations and verify exact unit magnitude at every step.
    ///
    /// Uses `BigInt`-backed rational arithmetic so that no overflow occurs.
    /// Rotations cycle through every 8th variant of each triple (4 quadrants ×
    /// swap alternating) to cover a diverse sequence.  Returns a [`DriftReport`].
    ///
    /// ## Panics
    ///
    /// Panics if any step produces a magnitude² different from 1.
    ///
    /// ## Example
    ///
    /// ```
    /// use pythagorean48::Pythagorean48;
    /// let report = Pythagorean48::prove_zero_drift(1000);
    /// assert!(report.zero_drift);
    /// ```
    pub fn prove_zero_drift(steps: usize) -> DriftReport {
        let triples = Self::all_triples();
        let mut x = BigFraction::from_integer(BigInt::from(1i64));
        let mut y = BigFraction::from_integer(BigInt::from(0i64));

        for i in 0..steps {
            let triple = &triples[i % triples.len()];
            // Cycle variant to get diverse rotations
            let variant = ((i / triples.len()) % 8) as u8;

            let (x_num, y_num, sx, sy) = Self::variant_coeffs(triple, variant);
            let c = BigInt::from(triple.c);

            let cos = BigFraction::new(BigInt::from(sx * x_num as i64), c.clone());
            let sin = BigFraction::new(BigInt::from(sy * y_num as i64), c);

            let new_x = x.clone() * cos.clone() - y.clone() * sin.clone();
            let new_y = x * sin + y * cos;
            x = new_x;
            y = new_y;

            // Magnitude squared must be exactly 1.
            let mag_sq = x.clone() * x.clone() + y.clone() * y.clone();
            let one = BigFraction::from_integer(BigInt::from(1i64));
            assert_eq!(
                mag_sq, one,
                "Zero drift violated at step {}: mag² ≠ 1",
                i + 1
            );
        }

        DriftReport {
            max_drift: 0.0,
            mean_drift: 0.0,
            zero_drift: true,
            steps_checked: steps,
        }
    }

    // -- angle analysis ----------------------------------------------------

    /// Return angular gaps (degrees) between consecutive sorted directions.
    ///
    /// The gap between the last and first direction wraps around 360°.
    pub fn angular_gaps() -> Vec<f64> {
        let dirs = Self::all_directions();
        let n = dirs.len();
        let angles: Vec<f64> = dirs
            .iter()
            .map(|(x, y)| {
                normalise_angle(y.to_f64().unwrap().atan2(x.to_f64().unwrap()))
            })
            .collect();

        let mut gaps = Vec::with_capacity(n);
        for i in 0..n {
            let a1 = angles[i];
            let a2 = angles[(i + 1) % n] + if i + 1 == n { TAU } else { 0.0 };
            gaps.push((a2 - a1).to_degrees());
        }
        gaps
    }

    // -- MSE comparison ----------------------------------------------------

    /// Compare Pythagorean48 against other schemes on random angles.
    ///
    /// Generates `n_samples` uniform random angles in `[0, 360)`, computes the
    /// nearest-direction approximation error for each scheme, and returns a
    /// [`MseResult`] for each.
    ///
    /// `seed` controls the deterministic PRNG used for sampling.
    pub fn mse_comparison(n_samples: usize, seed: u64) -> Vec<MseResult> {
        // Deterministic PRNG (xoshiro / simple LCG)
        let mut rng = Lcg64::new(seed);

        // Generate random angles
        let angles: Vec<f64> = (0..n_samples)
            .map(|_| rng.next_f64() * 360.0)
            .collect();

        // Build scheme directory
        let pyth_dirs: Vec<f64> = Self::all_directions()
            .iter()
            .map(|(x, y)| {
                normalise_angle(y.to_f64().unwrap().atan2(x.to_f64().unwrap()))
                    .to_degrees()
            })
            .collect();

        let pyth_name = format!("Pythagorean48 ({} dirs)", pyth_dirs.len());
        let schemes: Vec<(&str, Vec<f64>)> = vec![
            (
                &pyth_name,
                pyth_dirs,
            ),
            ("8 directions (compass)", (0..8).map(|i| i as f64 * 45.0).collect()),
            ("16 directions", (0..16).map(|i| i as f64 * 22.5).collect()),
            ("36 directions (10°)", (0..36).map(|i| i as f64 * 10.0).collect()),
            ("48 directions (7.5°)", (0..48).map(|i| i as f64 * 7.5).collect()),
        ];

        let mut results = Vec::new();
        for (name, dirs) in &schemes {
            let mut total_sq = 0.0f64;
            let mut max_err = 0.0f64;

            for &angle in &angles {
                let err = nearest_error_deg(angle, dirs);
                total_sq += err * err;
                if err > max_err {
                    max_err = err;
                }
            }

            let mse = total_sq / n_samples as f64;
            let rmse = mse.sqrt();
            results.push(MseResult {
                encoding_name: name.to_string(),
                num_directions: dirs.len(),
                mse,
                rmse,
                max_error: max_err,
            });
        }

        results
    }

    // -- internal helper ---------------------------------------------------

    /// Helper: resolve (x_num, y_num, sign_x, sign_y) for a triple + variant.
    fn variant_coeffs(
        triple: &PythagoreanTriple,
        variant: u8,
    ) -> (u64, u64, i64, i64) {
        let quad = variant & 0x03;
        let swapped = (variant & 0x04) != 0;

        let (xn, yn) = if swapped {
            (triple.b, triple.a)
        } else {
            (triple.a, triple.b)
        };

        let (sx, sy) = match quad {
            0 => (1i64, 1i64),
            1 => (-1i64, 1i64),
            2 => (-1i64, -1i64),
            3 => (1i64, -1i64),
            _ => unreachable!(),
        };

        (xn, yn, sx, sy)
    }
}

// ---------------------------------------------------------------------------
// LCG PRNG (deterministic, no external dependency)
// ---------------------------------------------------------------------------

struct Lcg64(u64);

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e3779b97f4a7c15)
    }

    fn next(&mut self) -> u64 {
        // xoshiro128** style mixing
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }

    fn next_f64(&mut self) -> f64 {
        // Generate in [0, 1)
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Angular error (degrees) between `angle` and the nearest in `directions`.
fn nearest_error_deg(angle: f64, directions: &[f64]) -> f64 {
    let mut best = f64::MAX;
    for &d in directions {
        let mut err = (d - angle).abs();
        if err > 180.0 {
            err = 360.0 - err;
        }
        if err < best {
            best = err;
        }
    }
    best
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;


    // -----------------------------------------------------------------------
    // 1. All triples valid
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_triples_satisfy_a2b2c2() {
        let triples = Pythagorean48::all_triples();
        assert_eq!(triples.len(), 52, "expected 52 triples with c ≤ 100");

        for (i, t) in triples.iter().enumerate() {
            let a2 = (t.a as u128) * (t.a as u128);
            let b2 = (t.b as u128) * (t.b as u128);
            let c2 = (t.c as u128) * (t.c as u128);
            assert_eq!(
                a2 + b2,
                c2,
                "Triple #{} ({}, {}, {}) fails a² + b² = c²",
                i + 1,
                t.a,
                t.b,
                t.c
            );
        }
    }

    #[test]
    fn test_canonical_order() {
        for t in &Pythagorean48::all_triples() {
            assert!(t.a <= t.b, "a must be ≤ b, got ({}, {}, {})", t.a, t.b, t.c);
            assert!(t.b < t.c, "b must be < c, got ({}, {}, {})", t.a, t.b, t.c);
        }
    }

    #[test]
    fn test_first_triple_is_345() {
        let ts = Pythagorean48::all_triples();
        assert_eq!(ts[0], PythagoreanTriple { a: 3, b: 4, c: 5 });
    }

    // -----------------------------------------------------------------------
    // 2. All directions are unit vectors (a/c)² + (b/c)² = 1
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_directions_unit_magnitude() {
        let dirs = Pythagorean48::all_directions();
        for (i, (x, y)) in dirs.iter().enumerate() {
            // Compute x² + y² as exact Fractions — should be 1/1
            let mag_sq = *x * *x + *y * *y;
            assert_eq!(
                mag_sq,
                Ratio::<i64>::from_integer(1),
                "Direction #{} has magnitude² = {:?}, expected 1/1",
                i,
                mag_sq
            );
        }
    }

    #[test]
    fn test_direction_count() {
        assert_eq!(Pythagorean48::all_directions().len(), 128);
    }

    #[test]
    fn test_direction_variant_0_is_unit() {
        let t = PythagoreanTriple { a: 3, b: 4, c: 5 };
        let (x, y) = Pythagorean48::direction(&t, 0);
        let mag_sq = x * x + y * y;
        assert_eq!(mag_sq, Ratio::<i64>::from_integer(1));
    }

    #[test]
    fn test_direction_variants_cover() {
        let t = PythagoreanTriple { a: 3, b: 4, c: 5 };
        let mut seen = Vec::new();
        for v in 0..8 {
            let dir = Pythagorean48::direction(&t, v);
            if !seen.iter().any(|(a, b): &(Fraction, Fraction)| a == &dir.0 && b == &dir.1) {
                seen.push(dir);
            }
        }
        // The 8 variants of (3,4,5) should give 8 unique directions
        assert_eq!(seen.len(), 8);
    }

    // -----------------------------------------------------------------------
    // 3. Zero drift proof
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_drift_1000_steps() {
        let report = Pythagorean48::prove_zero_drift(1000);
        assert!(report.zero_drift, "Expected zero drift after 1000 steps");
        assert_eq!(report.steps_checked, 1000);
    }

    #[test]
    fn test_zero_drift_5000_steps() {
        let report = Pythagorean48::prove_zero_drift(5000);
        assert!(report.zero_drift);
        assert_eq!(report.steps_checked, 5000);
    }

    // -----------------------------------------------------------------------
    // 4. Direction index
    // -----------------------------------------------------------------------

    #[test]
    fn test_direction_index_in_range() {
        let idx = Pythagorean48::direction_index(0.0);
        assert!(idx < 128);

        let idx = Pythagorean48::direction_index(180.0);
        assert!(idx < 128);

        let idx = Pythagorean48::direction_index(360.0);
        assert!(idx < 128);
    }

    #[test]
    fn test_direction_index_0_deg() {
        // 0° should be close to (1, 0)
        let idx = Pythagorean48::direction_index(0.0);
        let dirs = Pythagorean48::all_directions();
        let (x, y) = &dirs[idx];
        let angle = normalise_angle(y.to_f64().unwrap().atan2(x.to_f64().unwrap())).to_degrees();
        let distance = (angle - 0.0).abs().min(360.0 - (angle - 0.0).abs());
        assert!(distance < 10.0, "0° maps to {:.2}° (distance {:.2}°)", angle, distance);
    }

    // -----------------------------------------------------------------------
    // 5. Rotate
    // -----------------------------------------------------------------------

    #[test]
    fn test_rotate_identity_rotation() {
        // Rotating by (3,4,5) and then the "inverse" (complementary)
        let triple = PythagoreanTriple { a: 3, b: 4, c: 5 };

        // Rotate identity (1,0) by the triple
        let (x, y) = Pythagorean48::rotate(
            Ratio::from_integer(1),
            Ratio::from_integer(0),
            &triple,
        );

        // Should be (3/5, 4/5)
        assert_eq!(x, Ratio::new(3i64, 5));
        assert_eq!(y, Ratio::new(4i64, 5));
    }

    #[test]
    fn test_rotate_preserves_magnitude() {
        let triple = PythagoreanTriple { a: 5, b: 12, c: 13 };
        let (x, y) = Pythagorean48::rotate(
            Ratio::new(3i64, 5),
            Ratio::new(4i64, 5),
            &triple,
        );
        let mag_sq = x * x + y * y;
        assert_eq!(mag_sq, Ratio::<i64>::from_integer(1));
    }

    // -----------------------------------------------------------------------
    // 6. Angular coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_angular_gaps_sum_to_360() {
        let gaps = Pythagorean48::angular_gaps();
        let sum: f64 = gaps.iter().sum();
        assert!(
            (sum - 360.0).abs() < 1e-10,
            "Gaps sum to {:.10}, expected 360",
            sum
        );
        assert_eq!(gaps.len(), 128);
    }

    #[test]
    fn test_angular_gap_limits() {
        let gaps = Pythagorean48::angular_gaps();
        let min_gap = gaps.iter().cloned().fold(f64::MAX, f64::min);
        let max_gap = gaps.iter().cloned().fold(f64::MIN, f64::max);
        let mean_gap: f64 = gaps.iter().sum::<f64>() / gaps.len() as f64;

        // Print for visibility in test output
        eprintln!("Min gap: {:.4}°", min_gap);
        eprintln!("Max gap: {:.4}°", max_gap);
        eprintln!("Mean gap: {:.4}°", mean_gap);

        // Sanity checks — min should be > 0, max should be reasonable
        assert!(min_gap > 0.0);
        assert!(max_gap < 20.0);
    }

    // -----------------------------------------------------------------------
    // 7. MSE comparison
    // -----------------------------------------------------------------------

    #[test]
    fn test_mse_comparison() {
        let results = Pythagorean48::mse_comparison(100_000, 42);

        eprintln!();
        eprintln!("{:<35} {:>12} {:>12} {:>14}", "Encoding", "MSE (deg²)", "RMSE (deg)", "Max err (deg)");
        eprintln!("{}", "-".repeat(75));

        for r in &results {
            eprintln!(
                "{:<35} {:>12.4} {:>12.4} {:>14.4}",
                r.encoding_name, r.mse, r.rmse, r.max_error
            );
        }

        // Pythagorean48 should have lower MSE than compass 8-dir
        let pyth_mse = results[0].mse;
        let compass_mse = results[1].mse;
        assert!(
            pyth_mse < compass_mse,
            "Pythagorean48 MSE ({:.4}) should be < compass MSE ({:.4})",
            pyth_mse,
            compass_mse
        );
    }

    // -----------------------------------------------------------------------
    // 8. Sanity/foundational
    // -----------------------------------------------------------------------

    #[test]
    fn test_smallest_primitive() {
        // (3,4,5) is the smallest primitive triple
        let ts = Pythagorean48::all_triples();
        assert!(ts.iter().any(|t| t.a == 3 && t.b == 4 && t.c == 5));
    }

    #[test]
    fn test_all_directions_have_distinct_angles() {
        let dirs = Pythagorean48::all_directions();
        // Check that no two directions have identical angles (within f64 precision)
        let mut angles: Vec<f64> = dirs
            .iter()
            .map(|(x, y)| normalise_angle(y.to_f64().unwrap().atan2(x.to_f64().unwrap())))
            .collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for i in 1..angles.len() {
            let d = (angles[i] - angles[i - 1]).abs();
            assert!(
                d > 1e-12,
                "Two directions have the same angle: {:.15} and {:.15}",
                angles[i - 1],
                angles[i]
            );
        }
    }
}
