use anyhow::{Result, ensure};
use opensv::{
    phones::language_index,
    project,
    tensor::Tensor,
    voice::{Voice, read_nofs},
};
use serde_json::json;
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

fn tensor(directory: &Path, name: &str, value: &Tensor) -> Result<()> {
    value.validate()?;
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(format!("{name}.f32")))?;
    let mut file = BufWriter::new(file);
    for value in &value.data {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

pub struct Request<'a> {
    pub voice: &'a Path,
    pub teacher: &'a Path,
    pub projects: [&'a Path; 2],
    pub output: &'a Path,
    pub language: &'a str,
    pub dictionary: Option<&'a Path>,
    pub seed: u32,
    pub audio: bool,
}

pub fn run(request: &Request<'_>) -> Result<()> {
    ensure!(!request.output.exists(), "output directory already exists");
    ensure!(
        fs::read(request.projects[0])? != fs::read(request.projects[1])?,
        "training and validation projects must differ"
    );
    let source = Voice::load(request.voice)?;
    let teacher = Voice::load(request.teacher)?;
    let prepared = request
        .projects
        .iter()
        .map(|path| project::prepare(path, &source, request.language, request.dictionary))
        .collect::<Result<Vec<_>>>()?;
    let model = if request.voice.is_dir() {
        fs::read(request.voice.join("base_model.dnni"))?
    } else {
        read_nofs(&fs::read(request.voice)?)?
            .remove(b"acoustic".as_slice())
            .ok_or_else(|| anyhow::anyhow!("missing acoustic model"))?
    };
    fs::create_dir_all(request.output.parent().unwrap_or(Path::new(".")))?;
    fs::create_dir(request.output)?;
    fs::write(request.output.join("base_model.dnni"), model)?;
    if request.audio {
        ensure!(
            source.vocoder.sample_rate == teacher.vocoder.sample_rate,
            "sample rates differ"
        );
        let model = if request.voice.is_dir() {
            fs::read(request.voice.join("pitch_model.dnni"))?
        } else {
            read_nofs(&fs::read(request.voice)?)?
                .remove(b"vocoder".as_slice())
                .ok_or_else(|| anyhow::anyhow!("missing vocoder"))?
        };
        fs::write(request.output.join("pitch_model.dnni"), model)?;
    }
    for (split, phrases) in ["train", "validation"].iter().zip(prepared) {
        for (index, phrase) in phrases.iter().enumerate() {
            let directory = request.output.join(format!("{split}-{index:04}"));
            fs::create_dir(&directory)?;
            let features = &source.acoustic.features;
            let baseline = source
                .acoustic
                .run(&phrase.phones, &phrase.log_f0, request.seed)?;
            let prediction = teacher
                .acoustic
                .run(&phrase.phones, &phrase.log_f0, request.seed)?;
            ensure!(
                baseline.noise.data == prediction.noise.data,
                "teacher noise mismatch"
            );
            if request.audio {
                let samples = teacher.vocoder.render(&prediction.decoded)?;
                let spec = hound::WavSpec {
                    channels: 1,
                    sample_rate: teacher.vocoder.sample_rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                };
                let mut writer = hound::WavWriter::create(directory.join("target.wav"), spec)?;
                for sample in samples {
                    writer.write_sample(sample)?;
                }
                writer.finalize()?;
            } else {
                let target =
                    features.target_from(&teacher.acoustic.features, &prediction.normalized)?;
                tensor(&directory, "target", &target)?;
            }
            let phones = features.encode(&phrase.phones)?;
            let pitch = features.normalize_pitch(&phrase.log_f0)?;
            for (name, value) in [
                ("phones", &phones),
                ("pitch", &pitch),
                ("noise", &baseline.noise),
                ("baseline", &baseline.normalized),
            ] {
                tensor(&directory, name, value)?;
            }
            let languages = phrase
                .phones
                .iter()
                .map(|p| language_index(&features.sets, &p.language))
                .collect::<Result<Vec<_>>>()?;
            let metadata = json!({"version": 1, "split": split, "frames": pitch.frames,
                "phone_channels": phones.channels, "seed": request.seed,
                "counts": phrase.phones.iter().map(|p| p.frames).collect::<Vec<_>>(),
                "languages": languages,
                "symbols": phrase.phones.iter().map(|p| &p.symbol).collect::<Vec<_>>(),
                "log_f0": phrase.log_f0, "supervision": if request.audio { "audio" } else { "features" }});
            fs::write(
                directory.join("sample.json"),
                serde_json::to_vec_pretty(&metadata)?,
            )?;
            eprintln!("captured {split}-{index:04}: {} frames", pitch.frames);
        }
    }
    Ok(())
}
