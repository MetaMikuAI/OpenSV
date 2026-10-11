use anyhow::{Context, Result, ensure};
use clap::Args;
use std::{path::PathBuf, process::Command};

/// Train the acoustic core using the PyTorch backend.
#[derive(Args)]
pub struct TrainArgs {
    #[arg(long, default_value = "python")]
    python: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "acoustic", value_parser = ["acoustic", "embeddings"])]
    scope: String,
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Total target steps, including steps restored from a checkpoint.
    #[arg(long, default_value_t = 100)]
    steps: u32,
    #[arg(long, default_value_t = 1800)]
    seconds: u32,
    #[arg(long, default_value_t = 4)]
    threads: u32,
    #[arg(long, default_value_t = 25)]
    save_every: u32,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long, default_value_t = 5489)]
    seed: u32,
    #[arg(long)]
    resume: bool,
}

pub fn run(args: TrainArgs) -> Result<()> {
    ensure!(args.model.is_file(), "model file does not exist");
    ensure!(args.dataset.is_file(), "dataset manifest does not exist");
    ensure!(
        args.steps > 0 && args.seconds > 0 && args.threads > 0 && args.save_every > 0,
        "training limits must be positive"
    );
    ensure!(
        args.lr.is_none_or(|lr| lr > 0.0 && lr < 1.0),
        "learning rate must be between 0 and 1"
    );
    let mut command = Command::new(args.python);
    command
        .env("PYTHONUTF8", "1")
        .args(["-u", "-m", "metasv_train"])
        .arg("--model")
        .arg(args.model)
        .arg("--scope")
        .arg(args.scope)
        .arg("--dataset")
        .arg(args.dataset)
        .arg("--output")
        .arg(args.output)
        .arg("--steps")
        .arg(args.steps.to_string())
        .arg("--seconds")
        .arg(args.seconds.to_string())
        .arg("--threads")
        .arg(args.threads.to_string())
        .arg("--save-every")
        .arg(args.save_every.to_string())
        .arg("--seed")
        .arg(args.seed.to_string());
    if let Some(lr) = args.lr {
        command.arg("--lr").arg(lr.to_string());
    }
    if args.resume {
        command.arg("--resume");
    }
    let status = command.status().context(
        "cannot start Python; install the training package with python -m pip install -e training",
    )?;
    ensure!(status.success(), "training backend exited with {status}");
    Ok(())
}

/// Distill duration, pitch and vocoder networks from renderer traces.
#[derive(Args)]
pub struct ComponentsArgs {
    #[arg(long, default_value = "python")]
    python: PathBuf,
    #[arg(long)]
    voice: PathBuf,
    #[arg(long)]
    teacher: PathBuf,
    #[arg(long)]
    traces: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    acoustic_model: Option<PathBuf>,
    #[arg(long, default_value_t = 2)]
    steps: u32,
    #[arg(long, default_value_t = 1800)]
    seconds: u32,
    #[arg(long, default_value_t = 4)]
    threads: u32,
    #[arg(long, default_value_t = 0.00001)]
    lr: f64,
    #[arg(long)]
    resume: bool,
}

pub fn components(args: ComponentsArgs) -> Result<()> {
    let mut command = Command::new(args.python);
    command
        .env("PYTHONUTF8", "1")
        .args(["-u", "-m", "metasv_train.components"])
        .arg("--voice")
        .arg(args.voice)
        .arg("--teacher")
        .arg(args.teacher)
        .arg("--traces")
        .arg(args.traces)
        .arg("--output")
        .arg(args.output)
        .arg("--steps")
        .arg(args.steps.to_string())
        .arg("--seconds")
        .arg(args.seconds.to_string())
        .arg("--threads")
        .arg(args.threads.to_string())
        .arg("--lr")
        .arg(args.lr.to_string());
    if let Some(model) = args.acoustic_model {
        command.arg("--acoustic-model").arg(model);
    }
    if args.resume {
        command.arg("--resume");
    }
    let status = command
        .status()
        .context("cannot start component training backend")?;
    ensure!(status.success(), "component training exited with {status}");
    Ok(())
}

#[derive(Args)]
pub struct CaptureArgs {
    #[arg(long, default_value = "python")]
    python: PathBuf,
    #[arg(long)]
    teacher: PathBuf,
    #[arg(long)]
    voice: PathBuf,
    #[arg(long)]
    train_project: PathBuf,
    #[arg(long)]
    validation_project: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "mandarin")]
    language: String,
    #[arg(long)]
    dictionary: Option<PathBuf>,
    #[arg(long, default_value_t = 5489)]
    seed: u32,
}

pub fn capture(args: CaptureArgs) -> Result<()> {
    crate::capture::run(&crate::capture::Request {
        voice: &args.voice,
        teacher: &args.teacher,
        projects: [&args.train_project, &args.validation_project],
        output: &args.output,
        language: &args.language,
        dictionary: args.dictionary.as_deref(),
        seed: args.seed,
    })?;
    let mut command = Command::new(args.python);
    command
        .env("PYTHONUTF8", "1")
        .args(["-u", "-m", "metasv_train.capture"])
        .arg("--output")
        .arg(args.output);
    let status = command
        .status()
        .context("cannot start trace capture backend")?;
    ensure!(status.success(), "trace capture exited with {status}");
    Ok(())
}
