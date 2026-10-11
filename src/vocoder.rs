use crate::{
    excitation::{self, Source},
    model::{Model, f32_at, u32_at},
    network::Network,
    normalizer::Normalizer,
    residual::Residual,
    tensor::Tensor,
};
use anyhow::{Result, ensure};
use rustfft::{FftPlanner, num_complex::Complex};

struct Band {
    first: usize,
    width: usize,
    fade_in: usize,
    fade_out: usize,
    network: Network,
}
pub struct Vocoder {
    pub sample_rate: u32,
    hop: usize,
    period: f64,
    gain: f32,
    bias: f32,
    force_voiced: bool,
    normalizer: Normalizer,
    shape: Network,
    condition: Network,
    modulation: Network,
    ratios: Vec<f32>,
    bands: Vec<Band>,
    bins: usize,
    residual: Residual,
}
impl Vocoder {
    pub fn load(m: &Model) -> Result<Self> {
        let root = &m.nodes[0];
        let c = &root.children;
        let p = &root.payload;
        ensure!(
            c.len() == 13 && p.len() == 18 && p[12] == 1,
            "unsupported vocoder root"
        );
        let config = &m.nodes[c[0]];
        let cp = &config.payload;
        ensure!(
            (config.kind == "_vocfv1" && cp.len() == 48)
                || (config.kind == "_vocfv2" && cp.len() == 52),
            "unsupported vocoder configuration"
        );
        ensure!(
            u32_at(cp, 0)? == 72 && config.children.len() >= 4,
            "invalid vocoder features"
        );
        let sample_rate = u32_at(cp, 8)?;
        let period = f32_at(cp, 12)? as f64;
        let hop = (sample_rate as f64 * period).round() as usize;
        ensure!(
            sample_rate > 0 && period.is_finite() && period > 0.0 && (8..=65536).contains(&hop),
            "invalid vocoder timing"
        );
        for &i in &c[6..9] {
            ensure!(
                m.nodes[i].payload.len() == 13 && u32_at(&m.nodes[i].payload, 0)? == 2,
                "unsupported filter"
            );
        }
        let heads = &m.nodes[c[3]];
        let hp = &heads.payload;
        let count = u32_at(hp, 4)? as usize;
        ensure!(
            u32_at(hp, 0)? == 3 && hp.len() == 8 + count * 16 && heads.children.len() == 1,
            "invalid spectral heads"
        );
        let group = &m.nodes[heads.children[0]];
        ensure!(
            group.children.len() == count,
            "spectral band count mismatch"
        );
        let mut bands = Vec::new();
        let mut bins = 0;
        for (i, &node) in group.children.iter().enumerate() {
            let at = 8 + i * 16;
            let first = u32_at(hp, at)? as usize;
            let end = u32_at(hp, at + 4)? as usize;
            let fade_in = u32_at(hp, at + 8)? as usize;
            let fade_out = u32_at(hp, at + 12)? as usize;
            ensure!(
                end > first
                    && end <= 65536
                    && fade_in != 1
                    && fade_out != 1
                    && fade_in + fade_out <= end - first,
                "invalid spectral band"
            );
            bins = bins.max(end);
            bands.push(Band {
                first,
                width: end - first,
                fade_in,
                fade_out,
                network: Network::load(m, node)?,
            });
        }
        Ok(Self {
            sample_rate,
            hop,
            period,
            gain: f32_at(p, 4)?.exp(),
            bias: f32_at(p, 14)?,
            force_voiced: cp.len() == 52 && u32_at(cp, 48)? != 0,
            normalizer: Normalizer::load(m, config.children[0])?,
            shape: Network::load(m, c[1])?,
            condition: Network::load(m, c[2])?,
            modulation: Network::load(m, c[5])?,
            ratios: m.vector(c[4])?,
            bands,
            bins,
            residual: Residual::load(m, hop)?,
        })
    }
    pub fn render(&self, input: &Tensor) -> Result<Vec<f32>> {
        input.validate()?;
        ensure!(
            input.frames > 0 && input.channels == 72 && input.frames <= 32768,
            "invalid acoustic input"
        );
        let mut x = input.clone();
        for row in x.data.chunks_exact_mut(72) {
            ensure!(row[1] > 0.0, "F0 must be positive");
            row[1] = row[1].ln();
            if self.force_voiced {
                row[0] = 1.0;
            }
        }
        let x = self.normalizer.normalize(&x, true)?;
        let shape = self.shape.run(&x, None)?;
        ensure!(shape.channels == 1, "invalid shape output");
        let frames: Vec<_> = (0..input.frames)
            .map(|i| Source {
                f0: input.row(i)[1],
                voiced: self.force_voiced || input.row(i)[0] >= 0.5,
                shape: 2.9 / ((-shape.row(i)[0]).exp() + 1.0) + 0.1,
            })
            .collect();
        let condition = self.condition.run(&x, None)?;
        let mut spectra: [Tensor; 3] =
            std::array::from_fn(|_| Tensor::zeros(input.frames, self.bins));
        for band in &self.bands {
            let y = band.network.run(&condition, None)?;
            ensure!(
                y.frames == input.frames && y.channels == 3 * band.width,
                "spectral output mismatch"
            );
            for f in 0..input.frames {
                for (s, dest) in spectra.iter_mut().enumerate() {
                    for b in 0..band.width {
                        let mut v = y.row(f)[s * band.width + b];
                        if b < band.fade_in {
                            v = v * b as f32 / (band.fade_in - 1) as f32;
                        } else if b >= band.width - band.fade_out {
                            v = v * (band.width - 1 - b) as f32 / (band.fade_out - 1) as f32;
                        }
                        dest.row_mut(f)[band.first + b] += v;
                    }
                }
            }
        }
        let modulation = self.modulation.run(&condition, None)?;
        ensure!(
            modulation.frames == input.frames && modulation.channels == 2 * self.ratios.len(),
            "modulation shape mismatch"
        );
        let (source, phases) = excitation::periodic(&frames, self.hop, self.period, self.gain);
        let mut modulated = excitation::modulation(
            &frames,
            &phases,
            &self.ratios,
            &modulation.data,
            self.hop,
            self.period,
        );
        for (a, b) in modulated.iter_mut().zip(&source) {
            *a = *b * (*a + self.bias);
        }
        let residual = self.residual.render(&x, &source)?;
        let mut output = filter(&spectra[0], &source, self.hop)?;
        let noise = filter(&spectra[1], &residual, self.hop)?;
        let modulation = filter(&spectra[2], &modulated, self.hop)?;
        for i in 0..output.len() {
            output[i] = output[i] + noise[i] + modulation[i];
        }
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "nonfinite audio output"
        );
        Ok(output)
    }
}
fn filter(features: &Tensor, source: &[f32], hop: usize) -> Result<Vec<f32>> {
    let n = 2 * (2 * hop - 1).next_power_of_two();
    let half = features.channels / 2;
    ensure!(
        features.channels >= 4 && features.channels.is_multiple_of(2) && features.channels <= n,
        "invalid spectrum width"
    );
    let window: Vec<_> = (0..2 * hop)
        .map(|i| 0.5 - 0.5 * (i as f32 * std::f32::consts::TAU / (2 * hop) as f32).cos())
        .collect();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(n);
    let inverse = planner.plan_fft_inverse(n);
    let mut output = vec![0.0; source.len()];
    for f in 0..features.frames {
        let values = features.row(f);
        let mut spectrum = vec![Complex::new(0.0, 0.0); n];
        let mut signal = spectrum.clone();
        for i in 0..=half {
            spectrum[i].re = 10.0 * values[i] / (i + 1) as f32;
        }
        for i in 1..half {
            spectrum[n - half + i].re = 10.0 * values[half + i] / (half + 1 - i) as f32;
        }
        forward.process(&mut spectrum);
        for i in 0..2 * hop {
            let at = (f * hop + i) as isize - hop as isize;
            if at >= 0 && (at as usize) < source.len() {
                signal[(n - 2 * hop) / 2 + i].re = source[at as usize] * window[i];
            }
        }
        forward.process(&mut signal);
        for i in 0..=n / 2 {
            signal[i] *= Complex::from_polar(spectrum[i].re.exp(), spectrum[i].im);
        }
        signal[0].im = 0.0;
        signal[n / 2].im = 0.0;
        for i in 1..n / 2 {
            signal[n - i] = signal[i].conj();
        }
        inverse.process(&mut signal);
        for (i, v) in signal.iter().enumerate() {
            let at = (f * hop + i) as isize - (n / 2) as isize;
            if at >= 0 && (at as usize) < output.len() {
                let w = if i < 16 {
                    (i + 1) as f32 / 16.0
                } else if i >= n - 16 {
                    (n - 1 - i) as f32 / 16.0
                } else {
                    1.0
                };
                output[at as usize] += v.re / n as f32 * w;
            }
        }
    }
    Ok(output)
}
