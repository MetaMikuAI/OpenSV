use anyhow::{Result, ensure};
use clap::Args as ClapArgs;
use opensv::{
    phones::language_index,
    project,
    random::Gaussian,
    voice::{Voice, read_nofs},
};
use serde_json::json;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    voice: PathBuf,
    #[arg(long)]
    project: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "mandarin")]
    language: String,
    #[arg(long)]
    dictionary: Option<PathBuf>,
    #[arg(long, default_value_t = 8.0)]
    phrase_seconds: f64,
}

fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(())
}

pub fn run(args: Args) -> Result<()> {
    ensure!(
        args.phrase_seconds.is_finite() && (1.0..=30.0).contains(&args.phrase_seconds),
        "invalid phrase limit"
    );
    let voice = Voice::load(&args.voice)?;
    let phrases = project::prepare_with_limit(
        &args.project,
        &voice,
        &args.language,
        args.dictionary.as_deref(),
        args.phrase_seconds,
    )?;
    let mut output = Vec::new();
    for phrase in phrases {
        let features = &voice.acoustic.features;
        let phones = features.encode(&phrase.phones)?;
        let mut rng = Gaussian::new(5489);
        let noise: Vec<f32> = (0..phrase.log_f0.len() * 4).map(|_| rng.sample()).collect();
        output.push(json!({"start":phrase.start,"phones":phones.data,
            "counts":phrase.phones.iter().map(|p|p.frames).collect::<Vec<_>>(),
            "languages":phrase.phones.iter().map(|p|language_index(&features.sets,&p.language)).collect::<Result<Vec<_>>>()?,
            "noise":noise,"log_f0":phrase.log_f0}));
    }
    fs::create_dir_all(&args.output)?;
    let container = if args.voice.is_file() {
        Some(read_nofs(&fs::read(&args.voice)?)?)
    } else {
        None
    };
    for (filename, key) in [
        ("base_model.dnni", b"acoustic".as_slice()),
        ("pitch_model.dnni", b"vocoder".as_slice()),
    ] {
        let bytes = if let Some(ref container) = container {
            container
                .get(key)
                .ok_or_else(|| anyhow::anyhow!("missing {filename}"))?
                .clone()
        } else {
            fs::read(args.voice.join(filename))?
        };
        write(&args.output.join(filename), &bytes)?;
    }
    write(
        &args.output.join("conditioning.json"),
        &serde_json::to_vec(&json!({
        "sample_rate":voice.vocoder.sample_rate,"phrases":output}))?,
    )?;
    Ok(())
}
