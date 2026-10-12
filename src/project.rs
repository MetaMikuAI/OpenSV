use crate::{phones::TimedPhoneme, timing::Syllable, voice::Voice};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::{Path, PathBuf},
};

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
    syllables: Syllables,
}
type Syllables = Vec<Vec<String>>;
type Dictionary = HashMap<String, Syllables>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CustomDictionary {
    base: Option<PathBuf>,
    languages: HashMap<String, HashMap<String, Vec<String>>>,
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
fn dictionary(path: &Path, language: &str) -> Result<Dictionary> {
    if path.is_dir() {
        return Ok(directory_dictionary(path, language)?
            .into_iter()
            .map(|(word, phones)| (word, vec![phones]))
            .collect());
    }
    let custom: CustomDictionary = serde_json::from_slice(&fs::read(path)?)?;
    let mut entries = if let Some(base) = custom.base {
        let base = path.parent().unwrap_or(Path::new(".")).join(base);
        ensure!(base.is_dir(), "dictionary base must be a directory");
        dictionary(&base, language)?
    } else {
        HashMap::new()
    };
    if let Some(words) = custom.languages.get(language) {
        for (word, readings) in words {
            let word = lyric_key(word, language);
            ensure!(
                !word.is_empty() && !readings.is_empty(),
                "empty dictionary entry"
            );
            let syllables: Syllables = readings
                .iter()
                .map(|reading| reading.split_whitespace().map(str::to_owned).collect())
                .collect();
            ensure!(
                syllables.iter().all(|s| !s.is_empty()),
                "empty dictionary syllable"
            );
            entries.insert(word, syllables);
        }
    }
    Ok(entries)
}

fn lyric_key(lyrics: &str, language: &str) -> String {
    if language == "mandarin" {
        pinyin(lyrics.trim())
    } else {
        lyrics.trim().to_owned()
    }
}

fn directory_dictionary(path: &Path, language: &str) -> Result<HashMap<String, Vec<String>>> {
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
    } else if language == "japanese" {
        let mut aliases = Vec::new();
        for file in [
            "japanese-hira2romaji-dict.txt",
            "japanese-kata2romaji-dict.txt",
        ] {
            let text = match fs::read_to_string(path.join(file)) {
                Ok(text) => text,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error).with_context(|| format!("cannot read {file}")),
            };
            for line in text.lines() {
                let words: Vec<_> = line.split_whitespace().collect();
                if words.len() == 2
                    && !words[0].starts_with('#')
                    && let Some(phones) = entries.get(words[1]).cloned()
                {
                    aliases.push((words[0].to_string(), phones));
                }
            }
        }
        for (word, phones) in aliases {
            entries.entry(word).or_insert(phones);
        }
    }
    Ok(entries)
}

fn sustainable(phone: &str, language: &str) -> bool {
    matches!(phone, "a" | "i" | "u" | "e" | "o") || language == "japanese" && phone == "N"
}

fn explicit_syllables(phonemes: &str, language: &str) -> Syllables {
    let mut syllables = Vec::new();
    let mut phones = Vec::new();
    for phone in phonemes.split_whitespace() {
        phones.push(phone.to_owned());
        if sustainable(phone, language) {
            syllables.push(std::mem::take(&mut phones));
        }
    }
    if !phones.is_empty() {
        if let Some(last) = syllables.last_mut() {
            last.extend(phones);
        } else {
            syllables.push(phones);
        }
    }
    syllables
}

fn marker(note: &Note) -> Option<bool> {
    if !note.phonemes.trim().is_empty() {
        return None;
    }
    match note.lyrics.trim() {
        "+" | "＋" => Some(true),
        "-" | "－" | "ー" => Some(false),
        _ => None,
    }
}

fn prepare_notes(
    track: &Track,
    tempo: &[Tempo],
    language: &str,
    dictionary: &Dictionary,
) -> Result<Vec<RenderNote>> {
    struct PlacedNote<'a> {
        note: &'a Note,
        start: f64,
        end: f64,
        pitch: i32,
    }
    let mut placed = Vec::new();
    for note in &track.main_group.notes {
        let onset = note
            .onset
            .checked_add(track.main_ref.blick_offset)
            .context("onset overflow")?;
        let end = onset
            .checked_add(note.duration)
            .context("duration overflow")?;
        let pitch = note
            .pitch
            .checked_add(track.main_ref.pitch_offset)
            .context("pitch overflow")?;
        ensure!(
            onset >= 0 && note.duration > 0 && (0..=127).contains(&pitch),
            "invalid note"
        );
        placed.push(PlacedNote {
            note,
            start: seconds(tempo, onset),
            end: seconds(tempo, end),
            pitch,
        });
    }
    placed.sort_by(|a, b| a.start.total_cmp(&b.start));
    ensure!(
        placed.windows(2).all(|n| n[0].end <= n[1].start + 1e-9),
        "overlapping notes in a track"
    );
    let mut output: Vec<RenderNote> = Vec::new();
    let mut pending = VecDeque::new();
    let mut extensions = 0;
    for (index, current) in placed.iter().enumerate() {
        let connected = output
            .last()
            .is_some_and(|last| current.start - last.end < 0.2);
        if !connected {
            pending.clear();
            extensions = 0;
        }
        let syllables = match marker(current.note) {
            Some(true) => {
                ensure!(
                    connected && extensions > 0 && !pending.is_empty(),
                    "syllable continuation has no remaining syllable"
                );
                extensions -= 1;
                if extensions == 0 {
                    pending.drain(..).collect()
                } else {
                    vec![pending.pop_front().unwrap()]
                }
            }
            Some(false) => {
                let phone = output
                    .last()
                    .filter(|_| connected)
                    .and_then(|last| last.syllables.last())
                    .and_then(|phones| {
                        phones
                            .iter()
                            .rev()
                            .find(|phone| sustainable(phone, language))
                    })
                    .context("sustain marker has no preceding sustainable phoneme")?;
                vec![vec![phone.clone()]]
            }
            None => {
                let syllables = if !current.note.phonemes.trim().is_empty() {
                    explicit_syllables(&current.note.phonemes, language)
                } else {
                    let key = lyric_key(&current.note.lyrics, language);
                    if let Some(syllables) = dictionary.get(&key) {
                        syllables.clone()
                    } else if key == "br" {
                        vec![vec!["br".to_owned()]]
                    } else {
                        anyhow::bail!(
                            "no pronunciation for {:?}; supply --dictionary or explicit phonemes",
                            current.note.lyrics
                        );
                    }
                };
                extensions = 0;
                let mut previous_end = current.end;
                for next in &placed[index + 1..] {
                    let Some(consume) = marker(next.note) else {
                        break;
                    };
                    if next.start - previous_end >= 0.2 {
                        break;
                    }
                    extensions += usize::from(consume);
                    previous_end = next.end;
                }
                ensure!(
                    extensions < syllables.len(),
                    "too many syllable continuation markers for {:?}",
                    current.note.lyrics
                );
                pending = syllables.into();
                if extensions == 0 {
                    pending.drain(..).collect()
                } else {
                    vec![pending.pop_front().unwrap()]
                }
            }
        };
        output.push(RenderNote {
            start: current.start,
            end: current.end,
            pitch: current.pitch,
            syllables,
        });
    }
    Ok(output)
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
    prepare_with_limit(path, voice, language, dict, 30.0)
}

pub fn prepare_with_limit(
    path: &Path,
    voice: &Voice,
    language: &str,
    dict: Option<&Path>,
    phrase_seconds: f64,
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
        let notes = prepare_notes(track, tempo, language, &dict)?;
        let mut first = 0;
        while first < notes.len() {
            let mut end = first + 1;
            while end < notes.len()
                && notes[end].start - notes[end - 1].end < 0.2
                && notes[end].end - notes[first].start < phrase_seconds
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
                for phones in &n.syllables {
                    syllables.push(Syllable {
                        language: language.to_string(),
                        phones: phones.clone(),
                        seconds: (n.end - n.start) / n.syllables.len() as f64,
                        pitch: n.pitch,
                    });
                }
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

    fn test_directory() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "opensv-pronunciation-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        dir
    }

    fn track(lyrics: &[&str]) -> Track {
        Track {
            main_group: Group {
                notes: lyrics
                    .iter()
                    .enumerate()
                    .map(|(index, lyrics)| Note {
                        onset: index as i64 * QUARTER as i64,
                        duration: QUARTER as i64,
                        pitch: 60 + index as i32,
                        lyrics: lyrics.to_string(),
                        phonemes: String::new(),
                    })
                    .collect(),
            },
            groups: Vec::new(),
            main_ref: Reference::default(),
            mixer: Mixer::default(),
        }
    }

    fn resolve(track: &Track) -> Result<Vec<RenderNote>> {
        let dictionary = HashMap::from([(
            "word".to_owned(),
            vec![
                vec!["k".to_owned(), "i".to_owned()],
                vec!["r".to_owned(), "a".to_owned()],
                vec!["s".to_owned(), "u".to_owned()],
            ],
        )]);
        prepare_notes(
            track,
            &[Tempo {
                position: 0,
                bpm: 120.0,
            }],
            "japanese",
            &dictionary,
        )
    }

    fn phones(note: &RenderNote) -> Vec<&str> {
        note.syllables
            .iter()
            .flatten()
            .map(String::as_str)
            .collect()
    }

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

    #[test]
    fn japanese_kana_uses_optional_romaji_mappings() {
        let dir = test_directory();
        let romaji = dir.join("japanese-romaji-dict.txt");
        let hiragana = dir.join("japanese-hira2romaji-dict.txt");
        let katakana = dir.join("japanese-kata2romaji-dict.txt");
        fs::write(&romaji, "ki k i\nshi sh i\n").unwrap();
        let entries = directory_dictionary(&dir, "japanese").unwrap();
        assert_eq!(entries["ki"], ["k", "i"]);
        assert!(!entries.contains_key("き"));

        fs::write(&hiragana, "# kana mappings\nき ki\nし shi\nん unknown\n").unwrap();
        fs::write(&katakana, "キ ki\nシ shi\n").unwrap();
        let entries = directory_dictionary(&dir, "japanese").unwrap();
        assert_eq!(entries["き"], entries["キ"]);
        assert_eq!(entries["き"], entries["ki"]);
        assert_eq!(entries["し"], entries["シ"]);
        assert_eq!(entries["し"], entries["shi"]);
        assert!(!entries.contains_key("ん"));
        assert!(!entries.contains_key("漢字"));
        assert!(!entries.contains_key("ー"));

        fs::write(&hiragana, [0xff]).unwrap();
        assert!(directory_dictionary(&dir, "japanese").is_err());
        for file in [romaji, hiragana, katakana] {
            fs::remove_file(file).unwrap();
        }
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn custom_dictionary_overrides_relative_base_and_keeps_languages_separate() {
        let dir = test_directory();
        let base = dir.join("base");
        fs::create_dir(&base).unwrap();
        fs::write(base.join("japanese-romaji-dict.txt"), "ki k i\nn N\n").unwrap();
        fs::write(base.join("mandarin-xsampa-dict.txt"), "a a\n").unwrap();
        fs::write(base.join("cedict.txt"), "").unwrap();
        let file = dir.join("dictionary.json");
        fs::write(&file, r#"{"base":"base","languages":{"japanese":{"ki":["s u"],"word":["k i","r a"]},"mandarin":{"word":["a"]}}}"#).unwrap();
        let japanese = dictionary(&file, "japanese").unwrap();
        assert_eq!(japanese["ki"], [vec!["s", "u"]]);
        assert_eq!(japanese["n"], [vec!["N"]]);
        assert_eq!(japanese["word"], [vec!["k", "i"], vec!["r", "a"]]);
        assert_eq!(dictionary(&file, "mandarin").unwrap()["word"], [vec!["a"]]);

        fs::write(&file, r#"{"languages":{"mandarin":{"word":["a"]}}}"#).unwrap();
        assert!(dictionary(&file, "japanese").unwrap().is_empty());
        for value in [
            r#"{"languages":{"japanese":{"word":[]}}}"#,
            r#"{"languages":{"japanese":{"word":[" "]}}}"#,
            r#"{"languages":{"japanese":{" ":["a"]}}}"#,
            r#"{"languages":{"japanese":{"word":"a"}}}"#,
            r#"{"base":"dictionary.json","languages":{}}"#,
        ] {
            fs::write(&file, value).unwrap();
            assert!(dictionary(&file, "japanese").is_err(), "{value}");
        }
        for name in [
            "japanese-romaji-dict.txt",
            "mandarin-xsampa-dict.txt",
            "cedict.txt",
        ] {
            fs::remove_file(base.join(name)).unwrap();
        }
        fs::remove_file(file).unwrap();
        fs::remove_dir(base).unwrap();
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn continuation_keeps_all_syllables_and_original_note_timing() {
        let whole = resolve(&track(&["word"])).unwrap();
        assert_eq!(phones(&whole[0]), ["k", "i", "r", "a", "s", "u"]);
        assert_eq!(whole[0].syllables.len(), 3);
        let notes = resolve(&track(&["word", " + "])).unwrap();
        assert_eq!(phones(&notes[0]), ["k", "i"]);
        assert_eq!(phones(&notes[1]), ["r", "a", "s", "u"]);
        assert_eq!(notes[1].syllables.len(), 2);
        assert_eq!(
            (notes[0].start, notes[0].end, notes[0].pitch),
            (0.0, 0.5, 60)
        );
        assert_eq!(
            (notes[1].start, notes[1].end, notes[1].pitch),
            (0.5, 1.0, 61)
        );

        let mut source = track(&["word", "-", "＋", "ー", "+", "－", "br"]);
        source.main_group.notes.reverse();
        let notes = resolve(&source).unwrap();
        let expected = [
            vec!["k", "i"],
            vec!["i"],
            vec!["r", "a"],
            vec!["a"],
            vec!["s", "u"],
            vec!["u"],
            vec!["br"],
        ];
        for (note, expected) in notes.iter().zip(expected) {
            assert_eq!(phones(note), expected);
        }
    }

    #[test]
    fn markers_reject_missing_context_and_stop_at_track_or_rest_boundaries() {
        for lyrics in [
            vec!["+"],
            vec!["-"],
            vec!["ー"],
            vec!["word", "+", "+", "+"],
            vec!["br", "-"],
        ] {
            assert!(resolve(&track(&lyrics)).is_err(), "{lyrics:?}");
        }
        for continuation in ["+", "-"] {
            let mut source = track(&["word", continuation]);
            source.main_group.notes[1].onset += QUARTER as i64;
            assert!(resolve(&source).is_err());
        }
        resolve(&track(&["word"])).unwrap();
        assert!(resolve(&track(&["-"])).is_err());
    }

    #[test]
    fn explicit_phones_override_lyrics_and_support_syllable_markers() {
        let mut source = track(&["unlisted", "+", "ー"]);
        source.main_group.notes[0].phonemes = "h i t o".to_owned();
        let notes = resolve(&source).unwrap();
        assert_eq!(phones(&notes[0]), ["h", "i"]);
        assert_eq!(phones(&notes[1]), ["t", "o"]);
        assert_eq!(phones(&notes[2]), ["o"]);
        source.main_group.notes[1].phonemes = "N".to_owned();
        let notes = resolve(&source).unwrap();
        assert_eq!(phones(&notes[0]), ["h", "i", "t", "o"]);
        assert_eq!(phones(&notes[1]), ["N"]);
        assert_eq!(phones(&notes[2]), ["N"]);
    }
}
