# OpenSV

[中文](README.md) | [English](README.en.md)

Singing voice model training and rendering tools, with a CLI and a local training interface. Rust handles model loading and rendering; Python/PyTorch handles training.

## Install

Install [Rust](https://www.rust-lang.org/tools/install) and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```powershell
cargo build --release
uv sync --project training --locked
```

uv manages the Python environment in `training/.venv`, using Python 3.13 and CPU PyTorch by default. Dependency versions are recorded in `training/uv.lock`. If you already have a compiled engine, skip the Rust build and enter its path in the interface.

## Training interface

```powershell
uv run --project training --locked opensv-gui
```

For NVIDIA CUDA, use this command every time you launch the interface, then select CPU or CUDA there. The first run downloads CUDA PyTorch; omitting these options restores the default CPU dependencies.

```powershell
uv run --project training --locked --no-group cpu --extra cuda opensv-gui
```

The interface runs in your local browser and supports Chinese and English. Interface language and material language are separate settings.

1. Enter a base voice and the compiled `opensv` engine, then click **Check engine**. The base voice accepts `.svpk`, `voice.nofs`, or a decoded model directory. Supported architectures are `dds` acoustic, `gen2a` duration, and `nhv` vocoder; legacy networks and encrypted SVDB are unsupported.
2. Add one or more pairs of dry WAV audio and matching SVP/VSQX projects, then choose a new work directory. All recordings should use the same target voice. No handwritten dataset manifest is needed.
3. Select the material language (Mandarin or Japanese) and provide the base voice's pronunciation dictionary when needed. Click **Check and split**, review the messages, and listen to the segments.
4. Confirm that the materials match, set the total target epochs, and start training. Advanced settings let you switch to total steps. Each epoch is converted to steps using the number of training clips, with batch size fixed at 1; the interface shows epochs and the resulting total steps.

The training-loss metric shows the raw current-step value. The chart retains the raw curve and overlays a trailing 20-step mean (using available consecutive records during warm-up). Smoothing uses full per-step records, survives resume, and is independent of chart sampling. GUI training history is stored in `training-history.jsonl` and reloaded on resume. Long curves are sampled for display; the file retains all raw values. Validation loss is measured on a fixed held-out set and updates at checkpoints, showing performance on material not used for training.

Export each WAV from the project's zero point, keeping any leading silence. Use a single dry vocal without backing music, reverb, or layered voices. Each project must contain only one active vocal track; merge referenced groups in SVP first. Each project can be up to 10 minutes long. **Audio offset is normally 0**: enter `-0.5` if you trimmed 0.5 seconds from the beginning, or `0.5` if you added 0.5 seconds of silence.

Acoustic training from dry WAV audio supports CPU, NVIDIA CUDA, or Apple MPS, with CPU as the default. Data preparation and voice export still run on CPU. A GPU is not necessarily faster; measure performance on your dataset. Duration, automatic F0, and vocoder weights remain fixed. Training targets come from dry WAV audio; no teacher voicebank or teacher model's internal features are required. Pitch is extracted from the WAV and phoneme boundaries are estimated by the base model. This does not replace phoneme-level forced alignment, and passing the checks does not guarantee correct lyrics or alignment for every phoneme.

**Save and stop** saves after the current step. To resume, enter the original work directory, set a higher total target in epochs or steps (not an additional amount), and click **Resume training**. You can switch between CPU, CUDA, and MPS when resuming. Outputs include `run/latest.pt` (checkpoint), `run/trained-base.dnni` (acoustic model), and `trained-*.nofs` (voicebank). Providing a preview project also generates a WAV. Closing the browser does not stop training.

## CLI

**Pronunciation dictionary** and `--dictionary` accept a `clf-data` directory or a custom JSON file, for example:

```json
{"base":"clf-data","languages":{"japanese":{"人":["h i","t o"]}}}
```

`base` is optional and must point to a directory; relative paths are resolved from the JSON file's directory. Custom entries override the base dictionary. Each array item is one syllable with space-separated phonemes. Specify kanji readings yourself. Rendering supports `+` for syllable continuation, `-`/`ー` for sustained notes, and `br` for breaths.

The interface accepts a separate **Preview dictionary**; leave it blank to use the material dictionary. This does not change prepared training materials.

Below, `opensv` refers to the compiled engine, located at `target/release/opensv.exe` on Windows. Replace it with the program's path or add it to PATH.

```powershell
uv run --project training --locked opensv-prepare --engine target/release/opensv.exe --voice base.svpk --project song.vsqx --audio song.wav --output dataset --dictionary path/to/clf-data
uv run --project training --locked opensv train --model dataset/base_model.dnni --vocoder dataset/pitch_model.dnni --dataset dataset/dataset.json --output runs/demo --steps 100
opensv voice-export --voice base.svpk --model runs/demo/trained-base.dnni --output trained.nofs
opensv render --voice trained.nofs --project song.svp --output song.wav --language mandarin --dictionary path/to/clf-data --pitch-mode notes
```

To use CUDA for the WAV acoustic training command above, add `--no-group cpu --extra cuda` to the `uv run` options and `--device cuda` to the `opensv train` options.

Material import supports SVP and VSQX 3/4. Direct rendering accepts SVP, supports Mandarin, Japanese, and explicit phonemes, and outputs mono WAV. For parameters, batch import, and dataset formats, see the [training guide (Chinese)](training/README.md) and `opensv --help`.

Rendering uses note pitch; automatic F0, parameter curves, referenced groups, and encrypted voicebanks are not supported. Exported voices are not guaranteed to work with official hosts. Provide your own models, voicebanks, audio, and pronunciation dictionaries. Compiled engines and distribution ZIPs are not tracked in Git.

## Layout

- `src/`: Rust CLI, model loading, and inference.
- `training/`: Python training package and tests.

## Reference projects

- [RynthV](https://github.com/cubeww/rynthv)
- [TrainSV](https://github.com/Quadimension-CL/TrainSV)

## License

[GPL-3.0](LICENSE)

## macOS / Apple MPS

On an MPS-capable Mac, use the default dependencies (no CUDA extra). The GUI detects and enables **Apple GPU (MPS)**; CPU remains the default. Add `--device mps` to `opensv train`. On macOS the engine path is `./target/release/opensv`, without `.exe`.

The acoustic network runs on MPS. For WAV supervision, the fixed vocoder and spectral loss stay on CPU to preserve the existing float64 and complex FFT algorithms; differentiable device transfers carry gradients back to MPS. Feature-supervised prediction and loss run on MPS. Preparation, export and `train-components` remain CPU-only. Checkpoints can resume across CPU, CUDA and MPS, with possible floating-point differences.

Benchmark on your dataset: this hybrid path is not guaranteed to outperform CPU. Unavailable MPS and out-of-memory failures produce explicit errors, not silent device switches. `PYTORCH_ENABLE_MPS_FALLBACK` is not required; do not disable MPS memory safety limits.

History compatibility: older versions did not persist curves. Resuming in the same GUI service migrates any retained in-memory history for that work directory. If the old service has exited and no `training-history.jsonl` exists, past per-step losses cannot be recovered; recording starts with subsequent training.

Failed-job recovery: reuse the same work directory. Retry **Check and split** after preparation failure; use **Start training** if no `run/latest.pt` exists, or **Resume training** if a checkpoint exists. Retries preserve the old configuration and partial training output (plus the old dataset when preparing again) under `retry-backups/`. Unrelated directories, symlink outputs and jobs with `run.lock` are not automatically reused. Confirm the original process has exited before handling a stale lock.
