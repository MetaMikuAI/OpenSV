use crate::{project, voice::Voice};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::{fs::OpenOptions, path::PathBuf, time::Instant};

#[derive(Clone)]
pub struct RenderRequest {
    pub voice: PathBuf,
    pub project: PathBuf,
    pub output: PathBuf,
    pub language: String,
    pub dictionary: Option<PathBuf>,
    pub seed: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct RenderReport {
    pub output: PathBuf,
    pub samples: usize,
    pub sample_rate: u32,
    pub elapsed_ms: u128,
}

pub fn render(request: &RenderRequest) -> Result<RenderReport> {
    ensure!(!request.output.exists(), "output already exists");
    let started = Instant::now();
    let voice = Voice::load(&request.voice)?;
    let samples = project::render(
        &request.project,
        &voice,
        &request.language,
        request.dictionary.as_deref(),
        request.seed,
    )?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: voice.vocoder.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&request.output)?;
    let mut writer = hound::WavWriter::new(file, spec)?;
    for &sample in &samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(RenderReport {
        output: request.output.clone(),
        samples: samples.len(),
        sample_rate: spec.sample_rate,
        elapsed_ms: started.elapsed().as_millis(),
    })
}
