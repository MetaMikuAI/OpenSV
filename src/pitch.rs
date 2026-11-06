//! Score-only pitch curves.
//!
//! A perfectly flat F0 keeps the upper harmonics as steady lines, which is heard as a constant high whine; in a real
//! voice small pitch movements smear them. This adds glides, late vibrato and slow drift to the note pitch. It uses
//! nothing but the score, and it is a stand-in until the automatic F0 model is available.

use crate::random::Mt19937;

const FRAME: f64 = 0.005;
const CENT: f64 = std::f64::consts::LN_2 / 1200.0;
const GLIDE_FRAMES: usize = 15;
const WANDER_FRAMES: usize = 21;
const VIBRATO_DELAY: usize = 40;
const VIBRATO_FADE: usize = 40;
const VIBRATO_MIN_NOTE: usize = 50;
const VIBRATO_CENTS: f64 = 22.0;
const VIBRATO_HZ: f64 = 5.6;
const DRIFT: [(f64, f64); 3] = [(4.0, 0.7), (3.0, 1.9), (2.0, 3.7)];
const WANDER_CENTS: f64 = 9.0;

fn hann(size: usize) -> Vec<f64> {
    let window: Vec<f64> = (0..size)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * (i as f64 + 1.0) / (size as f64 + 1.0)).cos())
        .collect();
    let total: f64 = window.iter().sum();
    window.into_iter().map(|v| v / total).collect()
}

/// Convolution that holds the first and last values, so curves do not sag at phrase edges.
fn smooth(values: &[f64], window: &[f64]) -> Vec<f64> {
    let half = window.len() / 2;
    let last = values.len() as isize - 1;
    (0..values.len() as isize)
        .map(|i| {
            window
                .iter()
                .enumerate()
                .map(|(k, w)| w * values[(i + k as isize - half as isize).clamp(0, last) as usize])
                .sum()
        })
        .collect()
}

fn uniform(rng: &mut Mt19937) -> f64 {
    rng.next_u32() as f64 / 4_294_967_296.0
}

fn normal(rng: &mut Mt19937) -> f64 {
    let radius = (-2.0 * (1.0 - uniform(rng)).ln()).sqrt();
    radius * (std::f64::consts::TAU * uniform(rng)).cos()
}

/// Turn a stepped note contour (natural log of Hz, one value per 5 ms frame) into a natural-sounding one.
pub fn naturalize(steps: &[f32], seed: u32) -> Vec<f32> {
    let frames = steps.len();
    if frames == 0 {
        return Vec::new();
    }
    let x: Vec<f64> = steps.iter().map(|v| *v as f64).collect();
    let glide = hann(GLIDE_FRAMES);
    let contour = smooth(&x, &glide);

    let mut envelope = vec![0.0; frames];
    let mut start = 0;
    for end in 1..=frames {
        if end == frames || (x[end] - x[end - 1]).abs() > 1e-4 {
            if end - start > VIBRATO_MIN_NOTE {
                for (k, value) in envelope[start + VIBRATO_DELAY..end].iter_mut().enumerate() {
                    *value = (k as f64 / VIBRATO_FADE as f64).min(1.0);
                }
            }
            start = end;
        }
    }
    let envelope = smooth(&envelope, &glide);

    let mut rng = Mt19937::new(seed);
    let phases: Vec<f64> = DRIFT.iter().map(|_| uniform(&mut rng)).collect();
    let white: Vec<f64> = (0..frames).map(|_| normal(&mut rng)).collect();
    let wander = smooth(&white, &hann(WANDER_FRAMES));

    let mut vibrato_phase = 0.0;
    (0..frames)
        .map(|f| {
            let time = f as f64 * FRAME;
            let rate = VIBRATO_HZ + 0.3 * (std::f64::consts::TAU * 0.13 * time).sin();
            vibrato_phase += rate * FRAME;
            let vibrato =
                (std::f64::consts::TAU * vibrato_phase).sin() * envelope[f] * VIBRATO_CENTS;
            let drift: f64 = DRIFT
                .iter()
                .zip(&phases)
                .map(|((amount, hz), phase)| {
                    amount * (std::f64::consts::TAU * (hz * time + phase)).sin()
                })
                .sum();
            (contour[f] + (vibrato + drift + wander[f] * WANDER_CENTS) * CENT) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hz(midi: f64) -> f32 {
        (440f64.ln() + (midi - 69.0) * std::f64::consts::LN_2 / 12.0) as f32
    }

    fn cents(a: f32, b: f32) -> f64 {
        (a as f64 - b as f64) / CENT
    }

    #[test]
    fn keeps_length_and_is_repeatable_per_seed() {
        let notes = vec![hz(69.0); 300];
        let first = naturalize(&notes, 7);
        assert_eq!(first.len(), notes.len());
        assert_eq!(first, naturalize(&notes, 7));
        assert_ne!(first, naturalize(&notes, 8));
        assert!(naturalize(&[], 7).is_empty());
    }

    #[test]
    fn note_changes_become_glides() {
        let mut notes = vec![hz(60.0); 200];
        notes.extend(vec![hz(72.0); 200]);
        let curve = naturalize(&notes, 1);
        let largest = curve
            .windows(2)
            .map(|w| cents(w[1], w[0]).abs())
            .fold(0.0, f64::max);
        assert!(
            largest < 200.0,
            "largest step {largest} cents; raw notes jump 1200"
        );
    }

    #[test]
    fn long_notes_get_vibrato_and_stay_centred() {
        let notes = vec![hz(69.0); 400];
        let curve = naturalize(&notes, 3);
        let late = &curve[120..];
        let offsets: Vec<f64> = late.iter().map(|v| cents(*v, notes[0])).collect();
        let high = offsets.iter().cloned().fold(f64::MIN, f64::max);
        let low = offsets.iter().cloned().fold(f64::MAX, f64::min);
        let mean = offsets.iter().sum::<f64>() / offsets.len() as f64;
        assert!(high - low > 30.0, "vibrato range {} cents", high - low);
        assert!(high - low < 90.0, "vibrato range {} cents", high - low);
        assert!(mean.abs() < 12.0, "mean offset {mean} cents");
    }

    #[test]
    fn short_notes_have_no_vibrato() {
        let mut notes = Vec::new();
        for i in 0..20 {
            notes.extend(vec![hz(60.0 + (i % 2) as f64); 30]);
        }
        let curve = naturalize(&notes, 5);
        for (value, note) in curve[15..30 - 8].iter().zip(&notes[15..30 - 8]) {
            assert!(cents(*value, *note).abs() < 25.0);
        }
    }
}
