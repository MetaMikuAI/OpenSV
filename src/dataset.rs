use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize)]
pub struct DatasetReport {
    pub project: PathBuf,
    pub audio: PathBuf,
    pub project_format: String,
    pub tracks: usize,
    pub notes: usize,
    pub sample_rate: u32,
    pub channels: u16,
    pub audio_seconds: f64,
    pub peak: f64,
    pub training_ready: bool,
    pub checks: Vec<String>,
}

fn project_info(path: &Path) -> Result<(String, usize, usize)> {
    ensure!(
        fs::metadata(path)?.len() <= 32 * 1024 * 1024,
        "project exceeds 32 MiB"
    );
    let text = fs::read_to_string(path).context("cannot read project text")?;
    let text = text.trim_start_matches('\u{feff}').trim_end_matches('\0');
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match extension.as_str() {
        "svp" => {
            let project: Value = serde_json::from_str(text).context("invalid SVP JSON")?;
            let tracks = project["tracks"]
                .as_array()
                .context("SVP has no tracks array")?;
            let library = project["library"].as_array();
            let mut notes = 0;
            let mut vocal_tracks = 0;
            for track in tracks {
                let mut count = track["mainGroup"]["notes"].as_array().map_or(0, Vec::len);
                if let Some(groups) = track["groups"].as_array() {
                    for reference in groups {
                        let id = reference["groupID"].as_str().context("missing groupID")?;
                        let group = library
                            .and_then(|items| items.iter().find(|g| g["uuid"].as_str() == Some(id)))
                            .with_context(|| format!("missing referenced group {id}"))?;
                        count += group["notes"].as_array().map_or(0, Vec::len);
                    }
                }
                notes += count;
                vocal_tracks += usize::from(count > 0);
            }
            ensure!(notes > 0, "SVP contains no notes");
            Ok(("SVP".into(), vocal_tracks, notes))
        }
        "vsqx" => {
            let document = roxmltree::Document::parse(text).context("invalid VSQX XML")?;
            ensure!(
                matches!(document.root_element().tag_name().name(), "vsq3" | "vsq4"),
                "unsupported VSQX root"
            );
            let tracks: Vec<_> = document
                .descendants()
                .filter(|n| n.has_tag_name("vsTrack"))
                .collect();
            let notes = tracks
                .iter()
                .map(|t| t.descendants().filter(|n| n.has_tag_name("note")).count())
                .sum();
            ensure!(notes > 0, "VSQX contains no notes");
            Ok(("VSQX".into(), tracks.len(), notes))
        }
        _ => bail!("dataset project must be .svp or .vsqx"),
    }
}

/// 检查工程和音频的基础内容，不执行时间对齐或特征提取。
pub fn check(project: &Path, audio: &Path) -> Result<DatasetReport> {
    let (project_format, tracks, notes) = project_info(project)?;
    let mut wav = hound::WavReader::open(audio).context("cannot read WAV")?;
    let spec = wav.spec();
    ensure!(
        spec.sample_rate > 0 && spec.channels > 0 && wav.duration() > 0,
        "empty or invalid WAV"
    );
    let audio_seconds = wav.duration() as f64 / spec.sample_rate as f64;
    ensure!(audio_seconds <= 3600.0, "WAV exceeds one hour");
    let mut peak = 0f64;
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in wav.samples::<f32>() {
                let sample = sample?;
                ensure!(sample.is_finite(), "WAV contains nonfinite samples");
                peak = peak.max((sample as f64).abs());
            }
        }
        hound::SampleFormat::Int => {
            let scale = 2f64.powi(spec.bits_per_sample as i32 - 1);
            for sample in wav.samples::<i32>() {
                peak = peak.max((sample? as f64 / scale).abs());
            }
        }
    }
    ensure!(peak > 1e-8, "WAV is silent");
    let mut checks = vec!["工程与 WAV 可读取，音频包含有效采样。".into()];
    if tracks > 1 {
        checks.push("工程包含多个声轨，需要指定 WAV 对应的轨道。".into());
    }
    if spec.channels > 1 {
        checks.push("音频为多声道，预处理时需确认单声道转换方式。".into());
    }
    if peak >= 0.999 {
        checks.push("峰值接近或超过满幅，请检查削波。".into());
    }
    checks.push("尚未验证工程与录音是否匹配、是否含伴奏或混响。".into());
    checks.push("音素对齐、实际音高及声学特征提取尚未接入；当前不可直接训练。".into());
    Ok(DatasetReport {
        project: project.into(),
        audio: audio.into(),
        project_format,
        tracks,
        notes,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        audio_seconds,
        peak,
        training_ready: false,
        checks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_files_are_not_marked_training_ready() {
        let dir = std::env::temp_dir().join(format!("metasv-dataset-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let project = dir.join("song.svp");
        fs::write(
            &project,
            r#"{"tracks":[{"mainGroup":{"notes":[{"pitch":60}]}}]}"#,
        )
        .unwrap();
        let audio = dir.join("voice.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&audio, spec).unwrap();
        writer.write_sample(0.25f32).unwrap();
        writer.finalize().unwrap();
        let report = check(&project, &audio).unwrap();
        assert_eq!(report.notes, 1);
        assert!(!report.training_ready);
        assert_eq!(report.peak, 0.25);
        let vsqx = dir.join("song.vsqx");
        fs::write(&vsqx, r#"<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/"><vsTrack><vsPart><note><n>60</n></note></vsPart></vsTrack></vsq4>"#).unwrap();
        assert_eq!(check(&vsqx, &audio).unwrap().notes, 1);
        fs::write(&vsqx, "<vsq4>").unwrap();
        assert!(check(&vsqx, &audio).is_err());
        fs::remove_file(vsqx).unwrap();
        fs::remove_file(project).unwrap();
        fs::remove_file(audio).unwrap();
        fs::remove_dir(dir).unwrap();
    }
}
