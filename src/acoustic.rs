use crate::{
    model::{Model, f32_at, u32_at},
    network::Network,
    normalizer::Normalizer,
    phones::{PhoneSet, TimedPhoneme, language_index},
    random::Gaussian,
    tensor::{Matrix, Tensor},
};
use anyhow::{Result, ensure};

pub struct AcousticFeatures {
    pub sets: Vec<PhoneSet>,
    pub unified: PhoneSet,
    pub interval: f32,
    phones: Normalizer,
    pitch: Normalizer,
    acoustic: Normalizer,
}
impl AcousticFeatures {
    pub fn load(m: &Model, index: usize) -> Result<Self> {
        let n = &m.nodes[index];
        let c = &n.children;
        ensure!(
            ["_ftmfv2", "_ftmfv3"].contains(&n.kind.as_str())
                && c.len() == 6
                && n.payload.len() == 32,
            "unsupported acoustic features"
        );
        ensure!(
            u32_at(&n.payload, 4)? == 1 && u32_at(&n.payload, 8)? == 71,
            "unsupported feature dimensions"
        );
        let sets = m.nodes[c[0]]
            .children
            .iter()
            .map(|i| PhoneSet::load(m, *i))
            .collect::<Result<Vec<_>>>()?;
        let unified = PhoneSet::load(m, c[1])?;
        let out = Self {
            sets,
            unified,
            interval: f32_at(&n.payload, 24)?,
            phones: Normalizer::load(m, c[3])?,
            pitch: Normalizer::load(m, c[4])?,
            acoustic: Normalizer::load(m, c[5])?,
        };
        ensure!(
            out.phones.channels() == out.unified.symbols.len() + 1
                && out.pitch.channels() == 1
                && out.acoustic.channels() == 71,
            "feature normalization mismatch"
        );
        ensure!(out.interval == 0.005f32, "unsupported frame period");
        Ok(out)
    }
    pub fn encode(&self, phones: &[TimedPhoneme]) -> Result<Tensor> {
        ensure!(!phones.is_empty(), "empty phoneme sequence");
        let mut raw = Tensor::zeros(phones.len(), self.phones.channels());
        for (i, p) in phones.iter().enumerate() {
            ensure!(p.frames > 0, "empty phoneme");
            let set = &self.sets[language_index(&self.sets, &p.language)?];
            let source = set.index(&p.symbol)?;
            let target = self.unified.index(&set.unified[source])?;
            let row = raw.row_mut(i);
            row[target] = 1.0;
            *row.last_mut().unwrap() = (p.frames as f32 * self.interval + 0.05).ln();
        }
        self.phones.normalize(&raw, false)
    }
    pub fn normalize_pitch(&self, f0: &[f32]) -> Result<Tensor> {
        self.pitch
            .normalize(&Tensor::new(f0.len(), 1, f0.to_vec())?, false)
    }
    /// 将老师的连续声学特征转换到底模的归一化范围。
    pub fn target_from(&self, teacher: &Self, x: &Tensor) -> Result<Tensor> {
        self.acoustic
            .normalize(&teacher.acoustic.denormalize(x)?, false)
    }
    pub fn decode(&self, x: &Tensor, f0: &[f32]) -> Result<Tensor> {
        ensure!(x.frames == f0.len(), "pitch frame mismatch");
        let mut decoded = self.acoustic.denormalize(x)?;
        let mut out = Tensor::zeros(x.frames, 72);
        for (f, pitch) in f0.iter().enumerate() {
            let row = decoded.row_mut(f);
            row[0] = if row[0] > 0.5 { 1.0 } else { 0.0 };
            row[1] = row[1].clamp(0.01, 3.0);
            for v in &mut row[66..] {
                *v = v.clamp(0.0, 1.0);
            }
            let target = out.row_mut(f);
            target[0] = row[0];
            target[1] = pitch.exp();
            target[2..].copy_from_slice(&row[1..]);
        }
        out.validate()?;
        Ok(out)
    }
}

pub struct AcousticModel {
    pub features: AcousticFeatures,
    networks: Vec<Network>,
    speaker: Vec<f32>,
    style: Vec<f32>,
    language: Matrix,
}

pub struct AcousticOutput {
    pub context: Tensor,
    pub noise: Tensor,
    pub normalized: Tensor,
    pub decoded: Tensor,
}
impl AcousticModel {
    pub fn load(m: &Model) -> Result<Self> {
        let root = &m.nodes[0];
        ensure!(
            root.kind == "_rldtg0" && root.children.len() == 2,
            "unsupported acoustic root"
        );
        let base = &m.nodes[root.children[0]];
        let styles = &m.nodes[root.children[1]];
        ensure!(
            base.kind == "_rldms0"
                && base.children.len() == 8
                && styles.kind == "_stbkv1"
                && styles.children.len() == 2,
            "unsupported acoustic layout"
        );
        let context = &m.nodes[base.children[2]];
        let latent = &m.nodes[base.children[3]];
        ensure!(
            context.children.len() == 4 && latent.children.len() == 3,
            "unsupported latent/context model"
        );
        let indices: Vec<_> = context.children[..3]
            .iter()
            .chain(latent.children.iter())
            .chain(base.children[4..].iter())
            .copied()
            .collect();
        let networks = indices
            .iter()
            .map(|i| Network::load(m, *i))
            .collect::<Result<Vec<_>>>()?;
        let speaker = m.vector(base.children[1])?;
        let style = m.vector(m.nodes[styles.children[0]].children[0])?;
        let language = m.matrix(m.nodes[context.children[3]].children[0])?;
        ensure!(
            speaker.len() == 32 && style.len() == 32 && language.rows == 8,
            "unsupported embeddings"
        );
        Ok(Self {
            features: AcousticFeatures::load(m, base.children[0])?,
            networks,
            speaker,
            style,
            language,
        })
    }
    pub fn run(
        &self,
        phones: &[TimedPhoneme],
        log_f0: &[f32],
        seed: u32,
    ) -> Result<AcousticOutput> {
        let frames: usize = phones.iter().map(|p| p.frames).sum();
        ensure!(
            frames > 0 && frames == log_f0.len() && frames <= 32768,
            "acoustic frame limit/mismatch"
        );
        let pitch = self.features.normalize_pitch(log_f0)?;
        let encoded = self.networks[0].run(&self.features.encode(phones)?, None)?;
        ensure!(
            encoded.channels == 64 && encoded.frames == phones.len(),
            "phone encoder shape mismatch"
        );
        let mut context = Tensor::zeros(frames, 144);
        let mut frame = 0;
        let phases = [
            std::f32::consts::FRAC_PI_2,
            0.0,
            -std::f32::consts::FRAC_PI_2,
            -std::f32::consts::PI,
        ];
        for (i, p) in phones.iter().enumerate() {
            let language = language_index(&self.features.sets, &p.language)?;
            ensure!(language < self.language.cols, "language embedding missing");
            for j in 0..p.frames {
                let row = context.row_mut(frame);
                row[..64].copy_from_slice(encoded.row(i));
                let position = if p.frames == 1 {
                    0.5
                } else {
                    j as f32 / (p.frames - 1) as f32
                };
                for (s, value) in [position, pitch.data[frame]].iter().enumerate() {
                    let angle = value * 6.283_185_5_f32;
                    for k in 0..4 {
                        row[64 + s * 4 + k] = ((angle + phases[k]).cos() as f64 * 0.5 + 0.5) as f32;
                    }
                }
                row[72..104].copy_from_slice(&self.speaker);
                for k in 0..8 {
                    row[104 + k] = self.language.data[k * self.language.cols + language];
                }
                row[112..].copy_from_slice(&self.style);
                frame += 1;
            }
        }
        let n = &self.networks;
        let hidden = n[2].run(&context, None)?;
        let projected = n[1].run(&context, None)?;
        let latent_input = n[3].run(&projected, None)?;
        let mut controls = [0.0; 128];
        for i in 0..32 {
            let frequency = (i as f32 * (-26.575_424_f32 / 64.0)).exp2();
            let a = 64.0 * 0.995 * frequency;
            let b = 64.0 * frequency;
            controls[i] = a.sin();
            controls[i + 32] = a.cos();
            controls[i + 64] = b.sin();
            controls[i + 96] = b.cos();
        }
        let mut condition = Tensor::zeros(frames, 388);
        let mut noise = Tensor::zeros(frames, 4);
        let mut rng = Gaussian::new(seed);
        ensure!(hidden.channels == 256, "context output shape mismatch");
        for f in 0..frames {
            let row = condition.row_mut(f);
            row[..256].copy_from_slice(hidden.row(f));
            row[256..384].copy_from_slice(&controls);
            for c in 0..4 {
                let v = rng.sample();
                noise.data[f * 4 + c] = v;
                row[384 + c] = v * 0.5;
            }
        }
        let latent = n[5].run(&n[4].run(&latent_input, Some(&condition))?, None)?;
        let spectrum = n[7].run(&n[6].run(&hidden, Some(&latent))?, None)?;
        let aux = n[9].run(&n[8].run(&hidden, Some(&spectrum))?, None)?;
        ensure!(
            spectrum.channels == 64 && aux.channels == 7,
            "acoustic head shape mismatch"
        );
        let mut normalized = Tensor::zeros(frames, 71);
        for f in 0..frames {
            let row = normalized.row_mut(f);
            row[..2].copy_from_slice(&aux.row(f)[..2]);
            row[2..66].copy_from_slice(spectrum.row(f));
            row[66..].copy_from_slice(&aux.row(f)[2..]);
        }
        let decoded = self.features.decode(&normalized, log_f0)?;
        Ok(AcousticOutput {
            context,
            noise,
            normalized,
            decoded,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(lower: f32, upper: f32) -> AcousticFeatures {
        let normalizer = Normalizer {
            lower: vec![lower; 71],
            upper: vec![upper; 71],
            target: [0.0, 1.0],
        };
        AcousticFeatures {
            sets: vec![],
            unified: PhoneSet {
                name: String::new(),
                symbols: vec![],
                categories: vec![],
                unified: vec![],
                classes: vec![],
            },
            interval: 0.005,
            phones: normalizer.clone(),
            pitch: normalizer.clone(),
            acoustic: normalizer,
        }
    }

    #[test]
    fn teacher_target_uses_student_scale_without_threshold_or_clamp() {
        let teacher = features(10.0, 20.0);
        let student = features(0.0, 10.0);
        let input = Tensor::new(1, 71, vec![0.25; 71]).unwrap();
        let target = student.target_from(&teacher, &input).unwrap();
        assert!(target.data.iter().all(|&v| v == 1.25));
        let restored = teacher.target_from(&student, &target).unwrap();
        assert_eq!(restored.data, input.data);
    }
}
