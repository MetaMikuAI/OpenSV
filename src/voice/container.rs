use super::read_nofs;
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

const LIMIT: u64 = 512 * 1024 * 1024;
pub type Entries = HashMap<Vec<u8>, Vec<u8>>;

pub fn read(path: &Path) -> Result<Entries> {
    let mut file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= LIMIT,
        "voice file exceeds 512 MiB"
    );
    let mut magic = [0; 4];
    file.read_exact(&mut magic)?;
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    if &magic == b"SVPK" {
        let mut archive = zip::ZipArchive::new(file).context("invalid SVPK archive")?;
        ensure!(archive.len() <= 1024, "too many SVPK entries");
        let mut voice = archive
            .by_name("voice.nofs")
            .context("SVPK has no voice.nofs")?;
        ensure!(voice.size() <= LIMIT, "voice.nofs exceeds 512 MiB");
        voice.by_ref().take(LIMIT + 1).read_to_end(&mut bytes)?;
    } else {
        file.take(LIMIT + 1).read_to_end(&mut bytes)?;
    }
    let mut entries = read_nofs(&bytes)?;
    if !entries.contains_key(b"\x91".as_slice())
        && [b"acoustic".as_slice(), b"duration", b"vocoder"]
            .iter()
            .all(|key| entries.contains_key(*key))
    {
        return Ok(entries);
    }
    let refs = references(&entries)?;
    for (alias, key) in refs {
        let value = entries
            .get(&key)
            .with_context(|| format!("missing {alias} model"))?
            .clone();
        entries.insert(alias.as_bytes().to_vec(), value);
    }
    Ok(entries)
}

pub fn references(entries: &Entries) -> Result<Vec<(&'static str, Vec<u8>)>> {
    let data = entries
        .get(b"\x91".as_slice())
        .context("missing voice configuration; encrypted SVDB is unsupported")?;
    let fields = configuration(data)?;
    let mut result = Vec::new();
    for (alias, field, arch_field, expected) in [
        ("acoustic", "model_timbre_pred", "model_timbre_arch", "dds"),
        ("duration", "model_duration", "model_duration_arch", "gen2a"),
        ("vocoder", "model_vocoder", "model_vocoder_arch", "nhv"),
    ] {
        let single = |name: &str| -> Result<&Vec<u8>> {
            let values = fields
                .get(name)
                .with_context(|| format!("missing {name}"))?;
            ensure!(values.len() == 1 && !values[0].is_empty(), "invalid {name}");
            Ok(&values[0])
        };
        let arch = single(arch_field)?;
        ensure!(
            arch == expected.as_bytes(),
            "unsupported {alias} architecture: {} (expected {expected})",
            String::from_utf8_lossy(arch)
        );
        result.push((alias, single(field)?.clone()));
    }
    Ok(result)
}

fn configuration(data: &[u8]) -> Result<HashMap<String, Vec<Vec<u8>>>> {
    let mut reader = Reader(data);
    ensure!(reader.u32()? == 0xfeff, "invalid voice configuration");
    let count = reader.u32()? as usize;
    ensure!(
        count <= 4096 && data.len() <= 1024 * 1024,
        "voice configuration limit exceeded"
    );
    let mut fields = HashMap::new();
    for _ in 0..count {
        let kind = reader.u32()?;
        reader.u32()?;
        let mut state = 0xbefcb5e0u32;
        let name: Vec<u8> = reader
            .string()?
            .iter()
            .map(|byte| {
                let decoded = byte ^ state as u8;
                state = state.wrapping_mul(0x41c64e6d).wrapping_add(0x3039);
                decoded
            })
            .collect();
        let name = String::from_utf8(name)?;
        let count = reader.u32()? as usize;
        ensure!(count <= data.len(), "invalid configuration value count");
        let values = match kind {
            0 => (0..count)
                .map(|_| reader.string().map(Vec::from))
                .collect::<Result<Vec<_>>>()?,
            1 | 2 => {
                reader.take(count * if kind == 1 { 8 } else { 4 })?;
                Vec::new()
            }
            _ => bail!("unsupported configuration value type {kind}"),
        };
        if name.starts_with("model_") {
            ensure!(
                fields.insert(name, values).is_none(),
                "duplicate model configuration field"
            );
        }
    }
    ensure!(reader.0.is_empty(), "trailing voice configuration bytes");
    Ok(fields)
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, size: usize) -> Result<&'a [u8]> {
        ensure!(size <= self.0.len(), "truncated voice configuration");
        let (value, rest) = self.0.split_at(size);
        self.0 = rest;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn string(&mut self) -> Result<&'a [u8]> {
        let size = u16::from_le_bytes(self.take(2)?.try_into()?) as usize;
        self.take(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(name: &str, value: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend(0u32.to_le_bytes());
        data.extend(0u32.to_le_bytes());
        data.extend((name.len() as u16).to_le_bytes());
        let mut state = 0xbefcb5e0u32;
        for byte in name.bytes() {
            data.push(byte ^ state as u8);
            state = state.wrapping_mul(0x41c64e6d).wrapping_add(0x3039);
        }
        data.extend(1u32.to_le_bytes());
        data.extend((value.len() as u16).to_le_bytes());
        data.extend(value);
        data
    }

    #[test]
    fn binary_references_and_truncation() {
        let mut data = Vec::from(0xfeffu32.to_le_bytes());
        data.extend(1u32.to_le_bytes());
        data.extend(field("model_timbre_pred", &[0xff, 0, 0x80]));
        assert_eq!(
            configuration(&data).unwrap()["model_timbre_pred"][0],
            [0xff, 0, 0x80]
        );
        for len in 0..data.len() {
            assert!(configuration(&data[..len]).is_err());
        }
        data[4..8].copy_from_slice(&2u32.to_le_bytes());
        data.extend(field("model_timbre_pred", b"duplicate"));
        assert!(configuration(&data).is_err());
    }
}
