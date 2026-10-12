mod capture;
mod conditioning;
mod flat;
mod training;

use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use opensv::{
    acoustic::AcousticModel, model::Model, network::Network, phones::TimedPhoneme, tensor::Tensor,
};
use serde::Deserialize;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(name = "opensv", version, about = "OpenSV singing synthesis")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Assemble a Flat voice archive from a trained voice and a matching template.
    VoicePack(flat::Args),
    TrainComponents(training::ComponentsArgs),
    /// Package a trained acoustic model with a decoded voice template.
    VoiceExport {
        #[arg(long)]
        voice: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Train(training::TrainArgs),
    Dataset {
        #[command(subcommand)]
        command: DatasetCommand,
    },
    Render {
        #[arg(long)]
        voice: PathBuf,
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        language: String,
        #[arg(long)]
        dictionary: Option<PathBuf>,
        #[arg(long, default_value_t = 5489)]
        seed: u32,
        #[arg(long,required=true,value_parser=["notes"])]
        pitch_mode: String,
    },
    Vocoder {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        frames: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
    DnniRun {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 0)]
        root: usize,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        frames: usize,
        #[arg(long)]
        channels: usize,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        condition: Option<PathBuf>,
        #[arg(long)]
        condition_channels: Option<usize>,
        #[arg(long)]
        condition_frames: Option<usize>,
    },
    AcousticFeatures {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "japanese")]
        language: String,
        #[arg(long, default_value_t = 5489)]
        seed: u32,
    },
}
#[derive(Subcommand)]
enum DatasetCommand {
    Condition(conditioning::Args),
    Capture(training::CaptureArgs),
    Check {
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        audio: PathBuf,
        #[arg(long)]
        json: bool,
    },
}
fn read_tensor(path: &Path, frames: usize, channels: usize) -> Result<Tensor> {
    ensure!(
        frames > 0
            && channels > 0
            && frames
                .checked_mul(channels)
                .is_some_and(|n| n <= 64 * 1024 * 1024),
        "tensor dimensions exceed limit"
    );
    let bytes = fs::read(path)?;
    ensure!(
        bytes.len() == frames * channels * 4,
        "tensor file size mismatch"
    );
    Tensor::new(
        frames,
        channels,
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
}
fn write_tensor(path: &Path, t: &Tensor) -> Result<()> {
    t.validate()?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let bytes: Vec<u8> = t.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    file.write_all(&bytes)?;
    Ok(())
}
#[derive(Deserialize)]
struct Request {
    phonemes: Vec<Phone>,
}
#[derive(Deserialize)]
struct Phone {
    symbol: String,
    frames: usize,
    midi: f64,
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Dataset {
            command: DatasetCommand::Condition(args),
        } => conditioning::run(args)?,
        Command::TrainComponents(args) => training::components(args)?,
        Command::Dataset {
            command: DatasetCommand::Capture(args),
        } => training::capture(args)?,
        Command::VoicePack(args) => flat::pack(args)?,
        Command::VoiceExport {
            voice,
            model,
            output,
        } => {
            opensv::voice::export(&voice, &model, &output)?;
            println!("{}", output.display());
        }
        Command::Train(args) => training::run(args)?,
        Command::Dataset {
            command:
                DatasetCommand::Check {
                    project,
                    audio,
                    json,
                },
        } => {
            let report = opensv::dataset::check(&project, &audio)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "{}: {} tracks, {} notes; WAV: {:.2}s, {} Hz, {} channels",
                    report.project_format,
                    report.tracks,
                    report.notes,
                    report.audio_seconds,
                    report.sample_rate,
                    report.channels
                );
                for check in report.checks {
                    println!("{check}");
                }
            }
        }
        Command::Render {
            voice,
            project,
            output,
            language,
            dictionary,
            seed,
            pitch_mode: _,
        } => {
            let report = opensv::workflow::render(&opensv::workflow::RenderRequest {
                voice,
                project,
                output,
                language,
                dictionary,
                seed,
            })?;
            println!(
                "samples={} sample_rate={} elapsed_ms={}",
                report.samples, report.sample_rate, report.elapsed_ms
            );
        }
        Command::Vocoder {
            model,
            input,
            frames,
            output,
        } => {
            ensure!(!output.exists(), "output already exists");
            let vocoder = opensv::vocoder::Vocoder::load(&Model::read(model)?)?;
            let x = read_tensor(&input, frames, 72)?;
            let started = std::time::Instant::now();
            let samples = vocoder.render(&x)?;
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: vocoder.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(output)?,
                spec,
            )?;
            for &sample in &samples {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
            println!(
                "samples={} elapsed_ms={}",
                samples.len(),
                started.elapsed().as_millis()
            );
        }
        Command::Inspect { model } => {
            let m = Model::read(model)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"version":m.version,"nodes":m.nodes.len(),"root":m.nodes[0].kind})
                )?
            );
        }
        Command::DnniRun {
            model,
            root,
            input,
            frames,
            channels,
            output,
            condition,
            condition_channels,
            condition_frames,
        } => {
            ensure!(!output.exists(), "output already exists");
            let m = Model::read(model)?;
            let network = Network::load(&m, root)?;
            let x = read_tensor(&input, frames, channels)?;
            let c = if let Some(path) = condition {
                Some(read_tensor(
                    &path,
                    condition_frames.unwrap_or(frames),
                    condition_channels
                        .ok_or_else(|| anyhow::anyhow!("condition channels required"))?,
                )?)
            } else {
                None
            };
            let y = network.run(&x, c.as_ref())?;
            write_tensor(&output, &y)?;
            println!("frames={} channels={}", y.frames, y.channels);
        }
        Command::AcousticFeatures {
            model,
            input,
            output,
            language,
            seed,
        } => {
            ensure!(!output.exists(), "output directory already exists");
            let request: Request = serde_json::from_slice(&fs::read(input)?)?;
            let mut phones = Vec::new();
            let mut pitch = Vec::new();
            for p in request.phonemes {
                ensure!(
                    p.frames > 0
                        && p.frames <= 4000
                        && p.midi.is_finite()
                        && (0.0..=127.0).contains(&p.midi),
                    "invalid phoneme input"
                );
                pitch.extend(std::iter::repeat_n(
                    (440f64.ln() + (p.midi - 69.0) * 2f64.ln() / 12.0) as f32,
                    p.frames,
                ));
                phones.push(TimedPhoneme {
                    language: language.clone(),
                    symbol: p.symbol,
                    frames: p.frames,
                });
            }
            let m = Model::read(model)?;
            let acoustic = AcousticModel::load(&m)?;
            let started = std::time::Instant::now();
            let y = acoustic.run(&phones, &pitch, seed)?;
            fs::create_dir(&output)?;
            for (name, t) in [
                ("context.f32", &y.context),
                ("noise.f32", &y.noise),
                ("target.f32", &y.normalized),
                ("acoustic.f32", &y.decoded),
            ] {
                write_tensor(&output.join(name), t)?;
            }
            println!(
                "frames={} elapsed_ms={}",
                y.normalized.frames,
                started.elapsed().as_millis()
            );
        }
    }
    Ok(())
}
