use crate::{model::Model, network::Network, tensor::Tensor};
use anyhow::{Result, ensure};

struct Noise {
    state: u64,
    spare: Option<f32>,
}
impl Noise {
    fn uniform(&mut self) -> f32 {
        self.state = self.state * 48271 % 2147483647;
        (self.state - 1) as f32 * (1.0 / 1073741824.0) - 1.0
    }
    fn next(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        loop {
            let a = self.uniform();
            let b = self.uniform();
            let r = b * b + a * a;
            if r > 1.0 || r == 0.0 {
                continue;
            }
            let s = (-2.0 * r.ln() / r).sqrt();
            self.spare = Some(b * s);
            return a * s;
        }
    }
}
pub struct Residual {
    analysis: Vec<Vec<f32>>,
    synthesis: Vec<Vec<f32>>,
    network: Network,
    projection: Network,
    gate: bool,
    hop: usize,
}
impl Residual {
    pub fn load(m: &Model, hop: usize) -> Result<Self> {
        let root = &m.nodes[0];
        let c = &root.children;
        ensure!(c.len() == 13, "vocoder residual child count");
        let read = |i: usize| -> Result<Vec<Vec<f32>>> {
            let v = m.nodes[i]
                .children
                .iter()
                .map(|&j| m.vector(j))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                !v.is_empty()
                    && v.iter()
                        .all(|x| x.len() == v[0].len() && x.len() >= 3 && x.len() % 2 == 1),
                "invalid FIR filters"
            );
            Ok(v)
        };
        let synthesis = read(c[9])?;
        let analysis = read(c[10])?;
        ensure!(
            analysis.len() == synthesis.len()
                && analysis.len() >= 2
                && hop.is_multiple_of(analysis.len()),
            "invalid subbands"
        );
        Ok(Self {
            analysis,
            synthesis,
            network: Network::load(m, c[11])?,
            projection: Network::load(m, c[12])?,
            gate: root.payload[13] != 0,
            hop,
        })
    }
    pub fn render(&self, features: &Tensor, excitation: &[f32]) -> Result<Vec<f32>> {
        let bands = self.analysis.len();
        let factor = self.hop / bands;
        let frames = features.frames * factor;
        ensure!(
            frames * features.channels <= 64 * 1024 * 1024,
            "residual tensor exceeds limit"
        );
        let len = self.analysis[0].len();
        let first = bands - 1 - (len / 2) % bands;
        let discard = bands / 2 - 1;
        let mut input = Tensor::zeros(frames, bands * 2);
        let mut noise = Noise {
            state: 1,
            spare: None,
        };
        for f in 0..frames {
            let start = (first + (f + discard) * bands) as isize - (len - 1) as isize;
            let begin = (-start).max(0) as usize;
            let end = (excitation.len() as isize - start).clamp(0, len as isize) as usize;
            for b in 0..bands {
                let mut sum = 0.0;
                for k in begin..end {
                    sum += self.analysis[b][k] * excitation[(start + k as isize) as usize];
                }
                input.data[f * bands * 2 + b] = sum;
            }
            for b in 0..bands {
                input.data[f * bands * 2 + bands + b] = noise.next();
            }
        }
        let mut condition = Tensor::zeros(frames, features.channels);
        let leading = factor / 2;
        let offset = if factor.is_multiple_of(2) { 0.5 } else { 0.0 };
        let mut dest = 0;
        for _ in 0..leading {
            condition.row_mut(dest).copy_from_slice(features.row(0));
            dest += 1;
        }
        for f in 1..features.frames {
            for step in 0..factor {
                for c in 0..features.channels {
                    let a = features.row(f - 1)[c];
                    let b = features.row(f)[c];
                    condition.row_mut(dest)[c] =
                        (b - a) * (step as f32 + offset) * (1.0 / factor as f32) + a;
                }
                dest += 1;
            }
        }
        while dest < frames {
            condition
                .row_mut(dest)
                .copy_from_slice(features.row(features.frames - 1));
            dest += 1;
        }
        let hidden = self.network.run(&input, Some(&condition))?;
        let mut sub = self.projection.run(&hidden, None)?;
        ensure!(
            sub.frames == frames && sub.channels == bands,
            "subband shape mismatch"
        );
        if self.gate {
            for f in 0..frames {
                for b in 0..bands {
                    sub.data[f * bands + b] = input.data[f * bands * 2 + bands + b]
                        / (1.0 + (-sub.data[f * bands + b]).exp());
                }
            }
        }
        let len = self.synthesis[0].len();
        let latency = (len - 1) / 2;
        let mut ring = vec![0.0; len.max(512)];
        let mut pos = 0;
        let mut pending = 0;
        let mut output = Vec::with_capacity(excitation.len());
        for f in 0..frames {
            for b in 0..bands {
                let amplitude = sub.row(f)[b];
                for k in 0..len {
                    let i = (pos + k) % ring.len();
                    ring[i] += self.synthesis[b][k] * amplitude;
                }
            }
            for _ in 0..bands {
                if pending < latency {
                    pending += 1;
                } else {
                    output.push(ring[pos]);
                    ring[pos] = 0.0;
                    pos = (pos + 1) % ring.len();
                }
            }
        }
        for _ in 0..pending {
            output.push(ring[pos]);
            ring[pos] = 0.0;
            pos = (pos + 1) % ring.len();
        }
        ensure!(
            output.len() == excitation.len(),
            "residual sample count mismatch"
        );
        Ok(output)
    }
}
