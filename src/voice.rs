use crate::{
    acoustic::AcousticModel,
    model::{Model, u32_at},
    timing::Timing,
    vocoder::Vocoder,
};
use anyhow::{Context, Result, ensure};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::Write,
    path::Path,
};

/// 用训练后的声学模型生成新的已解码声库容器。
pub fn export(template: &Path, model: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "output already exists");
    let acoustic = fs::read(model)?;
    AcousticModel::load(&Model::parse(&acoustic)?)?;
    let mut entries: BTreeMap<Vec<u8>, Vec<u8>> = if template.is_dir() {
        let mut entries = BTreeMap::new();
        for (key, file) in [
            ("duration", "timing_model.dnni"),
            ("vocoder", "pitch_model.dnni"),
        ] {
            entries.insert(key.as_bytes().to_vec(), fs::read(template.join(file))?);
        }
        let f0 = template.join("f0_model.dnni");
        if f0.exists() {
            entries.insert(b"f0model-dds".to_vec(), fs::read(f0)?);
        }
        for (key, value) in [
            (".name", "MetaSV"),
            (".language", "japanese"),
            (".phoneset", "romaji"),
            (".version", "201"),
        ] {
            entries.insert(key.as_bytes().to_vec(), value.as_bytes().to_vec());
        }
        let fields = [
            ("model_duration", "duration"),
            ("model_duration_arch", "gen2a"),
            ("model_timbre_pred", "acoustic"),
            ("model_timbre_arch", "dds"),
            ("model_vocoder", "vocoder"),
            ("model_vocoder_arch", "nhv"),
        ];
        let mut config = Vec::new();
        config.extend(0xfeffu32.to_le_bytes());
        config.extend((fields.len() as u32).to_le_bytes());
        for (ordinal, (key, value)) in fields.iter().enumerate() {
            config.extend(0u32.to_le_bytes());
            config.extend((ordinal as u32).to_le_bytes());
            config.extend((key.len() as u16).to_le_bytes());
            let mut state = 0xbefcb5e0u32;
            for byte in key.bytes() {
                config.push(byte ^ state as u8);
                state = state.wrapping_mul(0x41c64e6d).wrapping_add(0x3039);
            }
            config.extend(1u32.to_le_bytes());
            config.extend((value.len() as u16).to_le_bytes());
            config.extend(value.as_bytes());
        }
        entries.insert(vec![0x91], config);
        entries
    } else {
        read_nofs(&fs::read(template)?)?.into_iter().collect()
    };
    entries.insert(b"acoustic".to_vec(), acoustic);
    Timing::load(&Model::parse(
        entries
            .get(b"duration".as_slice())
            .context("missing duration model")?,
    )?)?;
    Vocoder::load(&Model::parse(
        entries
            .get(b"vocoder".as_slice())
            .context("missing vocoder model")?,
    )?)?;
    let mut bytes = Vec::new();
    bytes.extend(0xf580u32.to_le_bytes());
    bytes.extend(10u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(12u32.to_le_bytes());
    bytes.extend(0x1000u16.to_le_bytes());
    bytes.extend(0u16.to_le_bytes());
    bytes.extend(12u32.to_le_bytes());
    for (key, value) in entries {
        let size = u32::try_from(16 + key.len() + value.len())?;
        bytes.extend(size.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(u16::try_from(key.len())?.to_le_bytes());
        bytes.extend(key);
        bytes.extend(u32::try_from(value.len())?.to_le_bytes());
        bytes.extend(value);
        bytes.extend(size.to_le_bytes());
    }
    let size = bytes.len() as u64;
    bytes[8..16].copy_from_slice(&size.to_le_bytes());
    read_nofs(&bytes)?;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?
        .write_all(&bytes)?;
    Ok(())
}

pub struct Voice {
    pub acoustic: AcousticModel,
    pub timing: Timing,
    pub vocoder: Vocoder,
}
impl Voice {
    pub fn load(path: &Path) -> Result<Self> {
        let models = if path.is_dir() {
            ["base_model.dnni", "timing_model.dnni", "pitch_model.dnni"]
                .map(|name| fs::read(path.join(name)).with_context(|| format!("read {name}")))
        } else {
            let entries = read_nofs(&fs::read(path)?)?;
            [b"acoustic".as_slice(), b"duration", b"vocoder"].map(|name| {
                entries.get(name).cloned().with_context(|| {
                    format!(
                        "missing decoded model {}; encrypted SVDB requires prior preparation",
                        String::from_utf8_lossy(name)
                    )
                })
            })
        };
        let [a, t, v] = models;
        Ok(Self {
            acoustic: AcousticModel::load(&Model::parse(&a?)?)?,
            timing: Timing::load(&Model::parse(&t?)?)?,
            vocoder: Vocoder::load(&Model::parse(&v?)?)?,
        })
    }
}
pub fn read_nofs(bytes: &[u8]) -> Result<HashMap<Vec<u8>, Vec<u8>>> {
    ensure!(
        bytes.len() >= 28
            && bytes.len() <= 512 * 1024 * 1024
            && u32_at(bytes, 0)? == 0xf580
            && u32_at(bytes, 4)? == 10,
        "unsupported NOFS container"
    );
    let declared = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    ensure!(
        declared == 0 || declared == bytes.len() as u64,
        "NOFS length mismatch"
    );
    let mut entries = HashMap::new();
    let mut at = 16;
    while at < bytes.len() {
        let size = u32_at(bytes, at)? as usize;
        ensure!(
            size >= 12 && size <= bytes.len() - at,
            "invalid NOFS record length"
        );
        let block = &bytes[at..at + size];
        ensure!(
            u32_at(block, size - 4)? as usize == size,
            "NOFS trailer mismatch"
        );
        let kind = u16::from_le_bytes(block[4..6].try_into().unwrap());
        if kind == 1 {
            ensure!(size >= 16, "NOFS entry is shorter than its header");
            let key_len = u16::from_le_bytes(block[6..8].try_into().unwrap()) as usize;
            ensure!(key_len <= size - 16, "invalid NOFS key");
            let len = u32_at(block, 8 + key_len)? as usize;
            ensure!(len == size - 16 - key_len, "invalid NOFS value");
            ensure!(
                entries
                    .insert(
                        block[8..8 + key_len].to_vec(),
                        block[12 + key_len..size - 4].to_vec()
                    )
                    .is_none(),
                "duplicate NOFS entry"
            );
        } else {
            ensure!(kind == 0x1000, "unsupported NOFS record type");
        }
        at += size;
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn short_entry_returns_error() {
        let mut bytes = Vec::new();
        bytes.extend(0xf580u32.to_le_bytes());
        bytes.extend(10u32.to_le_bytes());
        bytes.extend(28u64.to_le_bytes());
        bytes.extend(12u32.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(0u16.to_le_bytes());
        bytes.extend(12u32.to_le_bytes());
        assert!(read_nofs(&bytes).is_err());
    }
}
