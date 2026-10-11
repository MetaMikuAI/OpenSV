use rustfft::{FftPlanner, num_complex::Complex};
use std::{
    f64::consts::{PI, TAU},
    sync::OnceLock,
};

fn knot(i: usize) -> [f32; 6] {
    let rd = 0.01 * 600f64.powf(i as f64 / 255.0);
    let ra = if rd < 0.21 {
        0.000001
    } else if rd < 2.7 {
        (4.8 * rd - 1.0) / 100.0
    } else {
        0.323 / rd
    };
    let (rk, rg) = if i < 209 {
        let k = (22.4 + 11.8 * rd) / 100.0;
        (k, 0.25 * k / (0.11 * rd / (0.5 + 1.2 * k) - ra))
    } else {
        let oq = 1.0 - 1.0 / (2.17 * rd);
        let g = 0.0093552 + 5.96 / (7.92 - 2.0 * oq);
        (2.0 * g * oq - 0.9572, g)
    };
    let peak = 0.5 / rg;
    let closing = peak * (1.0 + rk);
    let tail = 1.0 - closing;
    let (mut lo, mut hi) = (0.0, 1.0 / ra);
    for _ in 0..80 {
        let d = (lo + hi) * 0.5;
        if -(-tail * d).exp_m1() / d > ra {
            lo = d
        } else {
            hi = d
        }
    }
    let decay = (lo + hi) * 0.5;
    let area =
        (-(-tail * decay).exp_m1() - tail * decay * (-tail * decay).exp()) / (decay * decay * ra);
    let w = PI / peak;
    let s = (w * closing).sin();
    let c = (w * closing).cos();
    let residual = |g: f64| area * s * (g * g + w * w) + g * s - w * c + w * (-g * closing).exp();
    lo = 0.0;
    hi = 1.0;
    while residual(hi) > 0.0 {
        hi *= 2.0;
    }
    for _ in 0..80 {
        let g = (lo + hi) * 0.5;
        if residual(g) > 0.0 { lo = g } else { hi = g }
    }
    [
        rd as f32,
        closing as f32,
        peak as f32,
        ra as f32,
        (decay / 0.005).ln() as f32,
        ((lo + hi) * 0.5 / 0.005).ln() as f32,
    ]
}
fn parameters(shape: f32) -> [f64; 5] {
    static KNOTS: OnceLock<Vec<[f32; 6]>> = OnceLock::new();
    let knots = KNOTS.get_or_init(|| (0..256).map(knot).collect());
    let upper = knots.partition_point(|k| k[0] <= shape);
    let left = knots[upper.saturating_sub(1)];
    let right = knots[upper.min(255)];
    let t = if left[0] == right[0] {
        0.0
    } else {
        (shape as f64 - left[0] as f64) / (right[0] - left[0]) as f64
    };
    let mut p = [0.0; 5];
    for i in 0..5 {
        p[i] = left[i + 1] as f64 * (1.0 - t) + right[i + 1] as f64 * t;
    }
    p[3] = p[3].exp() * 0.005;
    p[4] = p[4].exp() * 0.005;
    p
}
fn spectrum(p: [f64; 5], h: usize) -> Complex<f64> {
    let [closing, peak, ra, decay, growth] = p;
    let w = TAU * h as f64;
    let sw = PI / peak;
    let angle = sw * closing;
    let sine = angle.sin();
    let s = Complex::new(growth, -w);
    let rotation = Complex::from_polar(1.0, -w * closing);
    let opening = -(rotation * (s * sine - sw * angle.cos()) + sw * (-growth * closing).exp())
        / (sine * (s * s - sw * sw));
    let rate = Complex::new(decay, w);
    let mut returning = -rotation / rate;
    let endpoint = 1.0 - decay * ra;
    if decay * endpoint >= 0.005 {
        returning += rotation * endpoint * (1.0 - Complex::from_polar(1.0, -w * (1.0 - closing)))
            / (Complex::new(0.0, w) * ra * rate);
    }
    opening + returning
}
pub struct Source {
    pub f0: f32,
    pub voiced: bool,
    pub shape: f32,
}
fn add_window(out: &mut [f32], frame: usize, hop: usize, signal: &[f64], gain: f64) {
    let start = if frame == 0 { hop } else { 0 };
    let dest = frame.saturating_sub(1) * hop;
    for i in start..signal.len() {
        let window = gain * (0.5 - 0.5 * (TAU * i as f64 / (2 * hop) as f64).cos());
        out[dest + i - start] = (out[dest + i - start] as f64 + signal[i] * window) as f32;
    }
}
pub fn periodic(frames: &[Source], hop: usize, period: f64, gain: f32) -> (Vec<f32>, Vec<f64>) {
    let rate = hop as f32 / period as f32;
    let len = hop * 2;
    let n = (len * 2 - 1).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(n);
    let inverse = planner.plan_fft_inverse(n);
    let mut output = vec![0.0; frames.len() * hop];
    let mut phases = Vec::new();
    let mut cycles = 0.0;
    for (f, frame) in frames.iter().enumerate() {
        let increment = (frame.f0 * period as f32) as f64;
        cycles += increment;
        phases.push(cycles);
        if !frame.voiced {
            continue;
        }
        let count = ((0.5 * rate / frame.f0) as usize).min(len);
        if count == 0 {
            continue;
        }
        let p = parameters(frame.shape);
        let scale = 0.1 * rate as f64 / (frame.f0 as f64 * count as f64 * spectrum(p, 2).norm());
        let phase = TAU * (cycles % 1.0 - increment);
        let reference = spectrum(p, 1).arg();
        let step = PI * frame.f0 as f64 / rate as f64;
        let chirp: Vec<_> = (0..len)
            .map(|i| Complex::<f64>::from_polar(1.0, step * (i * i) as f64))
            .collect();
        let mut a = vec![Complex::new(0.0, 0.0); n];
        let mut b = a.clone();
        for i in 0..len {
            b[i] = Complex::new(chirp[i].re as f32, -chirp[i].im as f32);
        }
        for h in 1..=count.min(len - 1) {
            let lf = spectrum(p, h);
            let v = Complex::from_polar(lf.norm(), lf.arg() + h as f64 * (phase - reference))
                * chirp[h];
            a[h] = Complex::new(v.re as f32, v.im as f32);
            b[n - h] = b[h];
        }
        forward.process(&mut a);
        forward.process(&mut b);
        for i in 0..n {
            a[i] *= b[i];
        }
        inverse.process(&mut a);
        let signal: Vec<_> = (0..len)
            .map(|i| {
                (Complex::new((a[i].re / n as f32) as f64, (a[i].im / n as f32) as f64) * chirp[i])
                    .re
                    * scale
            })
            .collect();
        add_window(&mut output, f, hop, &signal, gain as f64);
    }
    (output, phases)
}
pub fn modulation(
    frames: &[Source],
    phases: &[f64],
    ratios: &[f32],
    parameters: &[f32],
    hop: usize,
    period: f64,
) -> Vec<f32> {
    let mut output = vec![0.0; frames.len() * hop];
    for (f, frame) in frames.iter().enumerate() {
        let mut signal = vec![0.0; 2 * hop];
        for (r, ratio) in ratios.iter().enumerate() {
            let offset = f * 2 * ratios.len() + r;
            let amplitude = (parameters[offset] as f64).exp();
            let angular = TAU * (*ratio as f64);
            let phase =
                angular * (phases[f] as f32) as f64 + parameters[offset + ratios.len()] as f64;
            let step = angular * frame.f0 as f64 * period / hop as f64;
            for (i, v) in signal.iter_mut().enumerate() {
                *v += amplitude * (phase + (i as f64 - hop as f64) * step).cos();
            }
        }
        add_window(&mut output, f, hop, &signal, 1.0);
    }
    output
}
