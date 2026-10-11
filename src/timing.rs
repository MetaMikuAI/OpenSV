use crate::{
    model::{Model, u32_at},
    network::Network,
    normalizer::Normalizer,
    phones::{PhoneSet, TimedPhoneme, language_index},
    tensor::{Matrix, Tensor},
};
use anyhow::{Result, ensure};

pub struct Syllable {
    pub language: String,
    pub phones: Vec<String>,
    pub seconds: f64,
    pub pitch: i32,
}
#[derive(Clone)]
struct Phone {
    syllable: usize,
    position: usize,
    language: usize,
    unified: usize,
    nucleus: bool,
}
struct Interval {
    phones: Vec<Phone>,
    seconds: f64,
    pitch: i32,
}
pub struct Timing {
    sets: Vec<PhoneSet>,
    unified: PhoneSet,
    input: Normalizer,
    output: Normalizer,
    language: Matrix,
    phone: Matrix,
    position: Matrix,
    voice: Vec<f32>,
    network: Network,
    min: i32,
    max: i32,
}
impl Timing {
    pub fn load(m: &Model) -> Result<Self> {
        let c = &m.nodes[0].children;
        ensure!(c.len() == 6, "unsupported duration model");
        let front = &m.nodes[c[0]];
        let f = &front.children;
        ensure!(f.len() == 4, "duration frontend mismatch");
        let embed = |i: usize| -> Result<Matrix> {
            let n = &m.nodes[i];
            ensure!(
                n.kind == "modl4" && n.children.len() == 1,
                "invalid embedding"
            );
            m.matrix(n.children[0])
        };
        let out = Self {
            sets: m.nodes[f[0]]
                .children
                .iter()
                .map(|&i| PhoneSet::load(m, i))
                .collect::<Result<_>>()?,
            unified: PhoneSet::load(m, f[1])?,
            input: Normalizer::load(m, f[2])?,
            output: Normalizer::load(m, f[3])?,
            language: embed(c[2])?,
            phone: embed(c[3])?,
            position: embed(c[4])?,
            voice: m.vector(c[5])?,
            network: Network::load(m, c[1])?,
            min: u32_at(&front.payload, 0)? as i32,
            max: u32_at(&front.payload, 4)? as i32,
        };
        ensure!(
            out.voice.len() == 128
                && out.language.rows == 32
                && out.phone.rows == 32
                && out.position.rows == 16
                && out.position.cols == 8
                && out.min < out.max,
            "invalid timing embedding dimensions"
        );
        Ok(out)
    }
    pub fn predict(&self, syllables: &[Syllable]) -> Result<Vec<TimedPhoneme>> {
        ensure!(
            !syllables.is_empty() && syllables.len() <= 4096,
            "invalid syllable count"
        );
        let mut intervals = Vec::new();
        for (si, s) in syllables.iter().enumerate() {
            ensure!(
                s.seconds.is_finite()
                    && s.seconds > 0.0
                    && (0..=127).contains(&s.pitch)
                    && !s.phones.is_empty(),
                "invalid syllable"
            );
            let language = language_index(&self.sets, &s.language)?;
            let set = &self.sets[language];
            let mut phones = Vec::new();
            for (position, symbol) in s.phones.iter().enumerate() {
                let i = set.index(symbol)?;
                phones.push(Phone {
                    syllable: si,
                    position,
                    language,
                    unified: self.unified.index(&set.unified[i])?,
                    nucleus: matches!(set.categories[i].as_str(), "vowel" | "diphthong"),
                });
            }
            intervals.push(Interval {
                phones,
                seconds: s.seconds,
                pitch: s.pitch,
            });
        }
        for i in 1..intervals.len() {
            if let Some(n) = intervals[i].phones.iter().position(|p| p.nucleus) {
                let onset: Vec<_> = intervals[i].phones.drain(..n).collect();
                intervals[i - 1].phones.extend(onset);
            }
        }
        let mut aligned = Vec::new();
        for interval in intervals {
            let count = interval.phones.iter().filter(|p| p.nucleus).count();
            if count <= 1 {
                aligned.push(interval);
                continue;
            }
            let mut begin = 0;
            let mut found = false;
            for i in 0..interval.phones.len() {
                if interval.phones[i].nucleus {
                    if found {
                        aligned.push(Interval {
                            phones: interval.phones[begin..i].to_vec(),
                            seconds: interval.seconds / count as f64,
                            pitch: interval.pitch,
                        });
                        begin = i;
                    }
                    found = true;
                }
            }
            aligned.push(Interval {
                phones: interval.phones[begin..].to_vec(),
                seconds: interval.seconds / count as f64,
                pitch: interval.pitch,
            });
        }
        let durations = Tensor::new(
            aligned.len(),
            1,
            aligned
                .iter()
                .map(|i| (i.seconds + 0.01f32 as f64).ln() as f32)
                .collect(),
        )?;
        let normalized = self.input.normalize(&durations, false)?;
        let count = aligned.iter().map(|i| i.phones.len()).sum();
        ensure!(count <= 4096, "too many phonemes");
        let mut features = Tensor::zeros(count, 272);
        let mut frame = 0;
        for (i, interval) in aligned.iter().enumerate() {
            for (pos, p) in interval.phones.iter().enumerate() {
                let row = features.row_mut(frame);
                row[..32].copy_from_slice(&self.phone.column(p.unified));
                row[32..64].copy_from_slice(&self.language.column(p.language));
                row[64..80].copy_from_slice(&self.position.column(pos.min(7)));
                row[80..96].copy_from_slice(
                    &self
                        .position
                        .column((interval.phones.len() - pos - 1).min(7)),
                );
                encode(normalized.data[i], &mut row[96..128]);
                encode(
                    (interval.pitch - self.min) as f32 / (self.max - self.min) as f32,
                    &mut row[128..144],
                );
                row[144..].copy_from_slice(&self.voice);
                frame += 1;
            }
        }
        let predicted = self
            .output
            .denormalize(&self.network.run(&features, None)?)?;
        ensure!(
            predicted.channels == 1 && predicted.frames == count,
            "timing output mismatch"
        );
        let mut result = Vec::new();
        let mut first = 0;
        let mut seconds = 0.0;
        let mut boundary = 0;
        for interval in aligned {
            let values: Vec<_> = predicted.data[first..first + interval.phones.len()]
                .iter()
                .map(|v| v.exp())
                .collect();
            let reciprocal = 1.0 / values.iter().sum::<f32>();
            for (p, v) in interval.phones.iter().zip(values) {
                let duration = ((v * reciprocal) as f64 * interval.seconds) as f32;
                ensure!(
                    duration.is_finite() && duration > 0.0,
                    "invalid predicted duration"
                );
                seconds += duration as f64;
                let next = ((seconds / (0.005f32 as f64)) as usize).max(boundary + 1);
                let s = &syllables[p.syllable];
                result.push(TimedPhoneme {
                    language: s.language.clone(),
                    symbol: s.phones[p.position].clone(),
                    frames: next - boundary,
                });
                boundary = next;
            }
            first += interval.phones.len();
        }
        Ok(result)
    }
}
fn encode(value: f32, out: &mut [f32]) {
    let half = out.len() / 2;
    for i in 0..half {
        let angle = value * (i as f32 * (-26.575424f32) / out.len() as f32).exp2();
        out[i] = angle.sin();
        out[i + half] = angle.cos();
    }
}
