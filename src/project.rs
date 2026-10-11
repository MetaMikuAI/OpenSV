use crate::{phones::TimedPhoneme, timing::Syllable, voice::Voice};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

const QUARTER: f64 = 705600000.0;
#[derive(Deserialize)]
struct Tempo {
    position: i64,
    bpm: f64,
}
#[derive(Deserialize)]
struct Time {
    tempo: Vec<Tempo>,
}
#[derive(Deserialize)]
struct Project {
    time: Time,
    tracks: Vec<Track>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Track {
    main_group: Group,
    #[serde(default)]
    groups: Vec<serde_json::Value>,
    #[serde(default)]
    main_ref: Reference,
    #[serde(default)]
    mixer: Mixer,
}
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Reference {
    #[serde(default)]
    blick_offset: i64,
    #[serde(default)]
    pitch_offset: i32,
}
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Mixer {
    #[serde(default)]
    gain_decibel: f32,
    #[serde(default)]
    pan: f32,
    #[serde(default)]
    mute: bool,
    #[serde(default)]
    solo: bool,
}
#[derive(Deserialize)]
struct Group {
    notes: Vec<Note>,
}
#[derive(Deserialize)]
struct Note {
    onset: i64,
    duration: i64,
    pitch: i32,
    #[serde(default)]
    lyrics: String,
    #[serde(default)]
    phonemes: String,
}
struct RenderNote {
    start: f64,
    end: f64,
    pitch: i32,
    phones: Vec<String>,
}
fn seconds(tempos: &[Tempo], position: i64) -> f64 {
    let mut elapsed = 0.0;
    let mut previous = 0;
    let mut bpm = tempos[0].bpm;
    for t in &tempos[1..] {
        if t.position > position {
            break;
        }
        elapsed += (t.position - previous) as f64 / QUARTER * 60.0 / bpm;
        previous = t.position;
        bpm = t.bpm;
    }
    elapsed + (position - previous) as f64 / QUARTER * 60.0 / bpm
}
fn pinyin(s: &str) -> String {
    s.to_lowercase()
        .replace("u:", "v")
        .replace('ü', "v")
        .trim_end_matches(['1', '2', '3', '4', '5'])
        .to_string()
}
fn dictionary(path: &Path, language: &str) -> Result<HashMap<String, Vec<String>>> {
    let file = match language {
        "mandarin" => "mandarin-xsampa-dict.txt",
        "japanese" => "japanese-romaji-dict.txt",
        _ => anyhow::bail!(
            "dictionary currently supports mandarin and japanese; use explicit phonemes for other languages"
        ),
    };
    let mut entries = HashMap::new();
    for line in fs::read_to_string(path.join(file))?.lines() {
        let words: Vec<_> = line.split_whitespace().collect();
        if words.len() > 1 && !words[0].starts_with('#') {
            entries.insert(
                words[0].to_string(),
                words[1..].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            );
        }
    }
    if language == "mandarin" {
        for line in fs::read_to_string(path.join("cedict.txt"))?.lines() {
            if line.starts_with('#') {
                continue;
            }
            let Some((words, reading)) = line.split_once('[') else {
                continue;
            };
            let Some((reading, _)) = reading.split_once(']') else {
                continue;
            };
            if reading.split_whitespace().count() != 1 {
                continue;
            }
            if let Some(phones) = entries.get(&pinyin(reading)).cloned() {
                for word in words.split_whitespace() {
                    entries
                        .entry(word.to_string())
                        .or_insert_with(|| phones.clone());
                }
            }
        }
    }
    Ok(entries)
}
pub struct Phrase {
    pub phones: Vec<TimedPhoneme>,
    pub log_f0: Vec<f32>,
    pub start: f64,
    pub gain: f32,
}

/// 共享渲染和训练采集的音素时长与音符音高。
pub fn prepare(
    path: &Path,
    voice: &Voice,
    language: &str,
    dict: Option<&Path>,
) -> Result<Vec<Phrase>> {
    let bytes = fs::read(path)?;
    let bytes = bytes.strip_suffix(&[0]).unwrap_or(&bytes);
    let project: Project = serde_json::from_slice(bytes)?;
    let tempo = &project.time.tempo;
    ensure!(
        !tempo.is_empty()
            && tempo[0].position == 0
            && tempo.iter().all(|t| t.bpm.is_finite() && t.bpm > 0.0)
            && tempo.windows(2).all(|t| t[0].position < t[1].position),
        "invalid tempo map"
    );
    let dict = if let Some(path) = dict {
        dictionary(path, language)?
    } else {
        HashMap::new()
    };
    let solo = project.tracks.iter().any(|t| t.mixer.solo);
    let mut output = Vec::new();
    for track in &project.tracks {
        if track.mixer.mute || (solo && !track.mixer.solo) {
            continue;
        }
        ensure!(
            track.groups.is_empty(),
            "referenced groups are not supported yet; flatten them before rendering"
        );
        ensure!(
            track.mixer.pan == 0.0,
            "nonzero pan is not supported by the mono renderer"
        );
        ensure!(track.mixer.gain_decibel.is_finite(), "invalid mixer gain");
        let mut notes = Vec::new();
        for n in &track.main_group.notes {
            let onset = n
                .onset
                .checked_add(track.main_ref.blick_offset)
                .context("onset overflow")?;
            let end = onset.checked_add(n.duration).context("duration overflow")?;
            let pitch = n
                .pitch
                .checked_add(track.main_ref.pitch_offset)
                .context("pitch overflow")?;
            ensure!(
                onset >= 0 && n.duration > 0 && (0..=127).contains(&pitch),
                "invalid note"
            );
            let phones = if !n.phonemes.trim().is_empty() {
                n.phonemes.split_whitespace().map(str::to_string).collect()
            } else {
                let key = if language == "mandarin" {
                    pinyin(&n.lyrics)
                } else {
                    n.lyrics.clone()
                };
                dict.get(&key).cloned().with_context(|| {
                    format!(
                        "no pronunciation for {:?}; supply --dictionary or explicit phonemes",
                        n.lyrics
                    )
                })?
            };
            notes.push(RenderNote {
                start: seconds(tempo, onset),
                end: seconds(tempo, end),
                pitch,
                phones,
            });
        }
        notes.sort_by(|a, b| a.start.total_cmp(&b.start));
        ensure!(
            notes.windows(2).all(|n| n[0].end <= n[1].start + 1e-9),
            "overlapping notes in a track"
        );
        let mut first = 0;
        while first < notes.len() {
            let mut end = first + 1;
            while end < notes.len()
                && notes[end].start - notes[end - 1].end < 0.2
                && notes[end].end - notes[first].start < 30.0
            {
                end += 1;
            }
            let phrase = &notes[first..end];
            let start = phrase[0].start - 0.1;
            let finish = phrase.last().unwrap().end + 0.1;
            ensure!(
                finish - start <= 30.2 && finish <= 600.0,
                "render exceeds current 30-second phrase / 10-minute project limits"
            );
            let silence = |seconds: f64, pitch: i32| Syllable {
                language: language.to_string(),
                phones: vec!["sil".to_string()],
                seconds,
                pitch,
            };
            let mut syllables = vec![silence(0.1, phrase[0].pitch)];
            let mut previous = phrase[0].start;
            for n in phrase {
                if n.start - previous > 1e-9 {
                    syllables.push(silence(n.start - previous, n.pitch));
                }
                syllables.push(Syllable {
                    language: language.to_string(),
                    phones: n.phones.clone(),
                    seconds: n.end - n.start,
                    pitch: n.pitch,
                });
                previous = n.end;
            }
            syllables.push(silence(0.1, phrase.last().unwrap().pitch));
            let phones = voice.timing.predict(&syllables)?;
            let frames = phones.iter().map(|p| p.frames).sum::<usize>();
            let mut pitch = Vec::with_capacity(frames);
            let mut ni = 0;
            for f in 0..frames {
                let time = start + f as f64 * 0.005;
                while ni + 1 < phrase.len() && time >= phrase[ni + 1].start {
                    ni += 1;
                }
                pitch.push(
                    (440f64.ln() + (phrase[ni].pitch - 69) as f64 * std::f64::consts::LN_2 / 12.0)
                        as f32,
                );
            }
            let gain = 10f32.powf(track.mixer.gain_decibel / 20.0);
            output.push(Phrase {
                phones,
                log_f0: pitch,
                start,
                gain,
            });
            first = end;
        }
    }
    ensure!(!output.is_empty(), "empty project");
    Ok(output)
}

/// 按音符音高渲染主音符组，输出单声道采样。
pub fn render(
    path: &Path,
    voice: &Voice,
    language: &str,
    dict: Option<&Path>,
    seed: u32,
) -> Result<Vec<f32>> {
    let phrases = prepare(path, voice, language, dict)?;
    let mut output = Vec::<f32>::new();
    let rate = voice.vocoder.sample_rate as f64;
    for phrase in phrases {
        let acoustic = voice.acoustic.run(&phrase.phones, &phrase.log_f0, seed)?;
        let samples = voice.vocoder.render(&acoustic.decoded)?;
        let offset = (phrase.start * rate).round() as i64;
        let length = (offset + samples.len() as i64).max(0) as usize;
        output.resize(output.len().max(length), 0.0);
        for (i, sample) in samples.iter().enumerate() {
            let at = offset + i as i64;
            if at >= 0 {
                output[at as usize] += sample * phrase.gain;
            }
        }
    }
    ensure!(output.iter().all(|v| v.is_finite()), "nonfinite audio");
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tempo_change_integrates_both_segments() {
        let map = vec![
            Tempo {
                position: 0,
                bpm: 120.0,
            },
            Tempo {
                position: 705600000,
                bpm: 60.0,
            },
        ];
        assert_eq!(seconds(&map, 1411200000), 1.5);
    }
    #[test]
    fn mandarin_normalization_preserves_hanzi() {
        assert_eq!(pinyin("lü4"), "lv");
        assert_eq!(pinyin("闪"), "闪");
    }
}
