use aes::cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit, block_padding::Pkcs7};
use anyhow::{Context, Result, ensure};
use opensv::{
    model::{Model, u32_at},
    voice::{read_container, read_nofs},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use zip::{ZipWriter, write::SimpleFileOptions};

#[derive(clap::Args)]
pub struct Args {
    #[arg(long)]
    voice: PathBuf,
    #[arg(long)]
    template: PathBuf,
    #[arg(long)]
    models: PathBuf,
    #[arg(long)]
    runtime: PathBuf,
    #[arg(long)]
    name: String,
    #[arg(long)]
    vendor: String,
    #[arg(long, default_value = "201")]
    version: String,
    #[arg(long)]
    profile: Option<PathBuf>,
    #[arg(long)]
    background: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|x| format!("{x:02x}")).collect()
}

fn file_component(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.len() <= 120
            && !value.ends_with(['.', ' '])
            && !value
                .chars()
                .any(|c| c.is_control() || "<>:\"/\\|?*".contains(c))
            && value != "."
            && value != "..",
        "invalid file name component"
    );
    let stem = value.split('.').next().unwrap().to_ascii_uppercase();
    ensure!(
        !["CON", "PRN", "AUX", "NUL"].contains(&stem.as_str())
            && !(stem.len() == 4
                && (stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.as_bytes()[3].is_ascii_digit()),
        "reserved file name"
    );
    Ok(())
}

fn runtime_key(path: &Path) -> Result<[u8; 32]> {
    let data = fs::read(path)?;
    ensure!(
        hex(&Sha256::digest(&data))
            == "b5e7b304c76e45e8a26c673b906e65bb442990269fa309f0bbaf37d09722fb5a",
        "unsupported Flat runtime build"
    );
    let pe = u32_at(&data, 0x3c)? as usize;
    let sections = u16::from_le_bytes(data[pe + 6..pe + 8].try_into()?);
    let optional = u16::from_le_bytes(data[pe + 20..pe + 22].try_into()?);
    let rva = 0x3860c + 0x431f8b;
    for i in 0..sections as usize {
        let at = pe + 24 + optional as usize + i * 40;
        let start = u32_at(&data, at + 12)?;
        let size = u32_at(&data, at + 16)?;
        if rva >= start && rva + 32 <= start + size {
            let offset = (u32_at(&data, at + 20)? + rva - start) as usize;
            return Ok(data
                .get(offset..offset + 32)
                .context("invalid runtime section")?
                .try_into()?);
        }
    }
    anyhow::bail!("runtime material not found")
}

fn envelope(data: &[u8]) -> Result<usize> {
    if data.starts_with(b"SVDB") {
        return Ok(4);
    }
    ensure!(
        data.starts_with(&[0xff, 0, 0xca, 0x7f, 3, 0, 0, 0]),
        "unsupported Flat envelope"
    );
    Ok(8)
}

fn decode(data: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
    let offset = envelope(data)?;
    let iv = data.get(offset..offset + 16).context("truncated IV")?;
    let mut body = data[offset + 16..].to_vec();
    let plain = cbc::Decryptor::<aes::Aes256>::new_from_slices(key, iv)?
        .decrypt_padded_mut::<Pkcs7>(&mut body)
        .map_err(|_| anyhow::anyhow!("invalid Flat payload"))?;
    Ok(plain.to_vec())
}

fn encode(plain: &[u8], template: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
    let offset = envelope(template)?;
    let iv = template.get(offset..offset + 16).context("truncated IV")?;
    let mut body = plain.to_vec();
    body.resize(plain.len() + 16, 0);
    let encrypted = cbc::Encryptor::<aes::Aes256>::new_from_slices(key, iv)?
        .encrypt_padded_mut::<Pkcs7>(&mut body, plain.len())
        .map_err(|_| anyhow::anyhow!("invalid padding"))?;
    let mut result = template[..offset + 16].to_vec();
    result.extend(encrypted);
    ensure!(
        decode(&result, key)? == plain,
        "Flat payload roundtrip mismatch"
    );
    Ok(result)
}

fn compatible_model(data: &[u8], template: &[u8]) -> Result<Vec<u8>> {
    let original = Model::parse(template)?;
    let mut model = Model::parse(data)?;
    ensure!(
        original.version == model.version && original.nodes.len() == model.nodes.len(),
        "model layout mismatch"
    );
    for i in 0..model.nodes.len() {
        let base = &original.nodes[i];
        ensure!(
            base.children == model.nodes[i].children,
            "model topology mismatch at {i}"
        );
        if base.kind == "prim4" && model.nodes[i].kind == "prim2" {
            let matrix = model.matrix(i)?;
            let shape = original.matrix(i)?;
            ensure!(
                (shape.rows, shape.cols) == (matrix.rows, matrix.cols),
                "matrix shape mismatch"
            );
            let bits = u32_at(&base.payload, 0)?;
            let limit = ((1u32 << (bits - 1)) - 1) as f32;
            let scales: Vec<f32> = matrix
                .data
                .chunks_exact(matrix.cols)
                .map(|row| {
                    (row.iter().fold(0.0f32, |m, v| m.max(v.abs())) * (limit + 1.0) / limit)
                        .max(f32::MIN_POSITIVE)
                })
                .collect();
            let mut payload = base.payload[..16].to_vec();
            for scale in &scales {
                payload.extend(scale.to_le_bytes());
            }
            payload.extend((matrix.rows as u32).to_le_bytes());
            payload.extend((matrix.cols as u32).to_le_bytes());
            for col in 0..matrix.cols {
                for (row, scale) in scales.iter().enumerate() {
                    let n = (matrix.data[row * matrix.cols + col] / scale * (limit + 1.0))
                        .round_ties_even()
                        .clamp(-limit - 1.0, limit) as i16;
                    if bits == 8 {
                        payload.push(n as i8 as u8);
                    } else {
                        payload.extend(n.to_le_bytes());
                    }
                }
            }
            model.nodes[i].payload = payload;
            model.nodes[i].kind = base.kind.clone();
        }
        ensure!(
            base.kind == model.nodes[i].kind,
            "unsupported node change at {i}"
        );
        if base.kind == "prim1" {
            ensure!(
                original.vector(i)?.len() == model.vector(i)?.len(),
                "vector shape mismatch"
            );
        } else if ["prim0", "prim2", "prim3", "prim4", "prim5"].contains(&base.kind.as_str()) {
            let a = original.matrix(i)?;
            let b = model.matrix(i)?;
            ensure!(
                (a.rows, a.cols) == (b.rows, b.cols),
                "matrix shape mismatch"
            );
        } else {
            ensure!(
                base.payload == model.nodes[i].payload,
                "non-weight model change at {i}"
            );
        }
    }
    let mut output = template[..8].to_vec();
    let mut position = 8;
    for node in &model.nodes {
        let header = template
            .get(position..position + 20)
            .context("truncated model")?;
        let old_size = u32_at(header, 16)? as usize;
        output.extend(&header[..16]);
        output.extend((node.payload.len() as u32).to_le_bytes());
        output.extend(&node.payload);
        position += 20 + old_size;
    }
    ensure!(position == template.len(), "model serialization mismatch");
    Model::parse(&output)?;
    Ok(output)
}

fn vector(model: &Model, path: &[usize], count: usize) -> Result<String> {
    let mut index = 0;
    for &child in path {
        index = *model.nodes[index]
            .children
            .get(child)
            .context("unsupported embedding layout")?;
    }
    let values = model.vector(index)?;
    ensure!(values.len() == count, "embedding size mismatch");
    Ok(hex(&values
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>())
    .to_uppercase())
}

fn replace_config(template: &[u8], config: &[u8]) -> Result<Vec<u8>> {
    read_nofs(template)?;
    let end = 16 + u32_at(template, 16)? as usize;
    ensure!(
        end >= 36 && (end - 36).is_multiple_of(12),
        "unsupported NOFS index layout"
    );
    let mut result = template[..end].to_vec();
    let mut offsets = HashMap::new();
    let mut pos = end;
    let mut found = false;
    while pos < template.len() {
        offsets.insert(pos as u64, result.len() as u64);
        let size = u32_at(template, pos)? as usize;
        let key_size = u16::from_le_bytes(template[pos + 6..pos + 8].try_into()?) as usize;
        let key = &template[pos + 8..pos + 8 + key_size];
        if key == [0x7e, 0x7f, 0x7f, 0x7f] {
            let size = (16 + key.len() + config.len()) as u32;
            result.extend(size.to_le_bytes());
            result.extend(1u16.to_le_bytes());
            result.extend((key.len() as u16).to_le_bytes());
            result.extend(key);
            result.extend((config.len() as u32).to_le_bytes());
            result.extend(config);
            result.extend(size.to_le_bytes());
            found = true;
        } else {
            result.extend(&template[pos..pos + size]);
        }
        pos += size;
    }
    ensure!(found, "missing SVDB configuration");
    for at in (22..end - 14).step_by(12) {
        let old = u64::from_le_bytes(template[at + 4..at + 12].try_into()?);
        result[at + 4..at + 12].copy_from_slice(
            &offsets
                .get(&old)
                .context("invalid NOFS index offset")?
                .to_le_bytes(),
        );
    }
    if template[8..16] != [0; 8] {
        let len = result.len() as u64;
        result[8..16].copy_from_slice(&len.to_le_bytes());
    }
    read_nofs(&result)?;
    Ok(result)
}

pub fn pack(args: Args) -> Result<()> {
    file_component(&args.name)?;
    ensure!(!args.vendor.trim().is_empty(), "vendor cannot be empty");
    ensure!(
        !args.version.is_empty() && args.version.bytes().all(|x| x.is_ascii_digit()),
        "version must be numeric"
    );
    ensure!(!args.output.exists(), "output already exists");
    let key = runtime_key(&args.runtime)?;
    let template = fs::read(&args.template)?;
    let entries = read_nofs(&template)?;
    let config_blob = entries
        .get([0x7e, 0x7f, 0x7f, 0x7f].as_slice())
        .context("template is not a Flat voice")?;
    let mut config: Value = serde_json::from_slice(&decode(config_blob, &key)?)?;
    let voice = read_container(&args.voice)?;
    let mut files = Vec::new();
    let mut models = HashMap::new();
    for (field, alias) in [
        ("base_model", "acoustic"),
        ("pitch_model", "vocoder"),
        ("timing_model", "duration"),
        ("f0_model", "f0model-dds"),
    ] {
        let reference = config[field].as_str().context("missing model reference")?;
        file_component(reference)?;
        let wrapped = fs::read(args.models.join(format!("{reference}.dnni")))?;
        let data = compatible_model(
            voice
                .get(alias.as_bytes())
                .with_context(|| format!("missing {alias} model"))?,
            &decode(&wrapped, &key)?,
        )?;
        models.insert(alias, Model::parse(&data)?);
        let name = hex(&Sha256::digest(&data));
        let name = &name[..32];
        files.push((
            format!("models/{name}.dnni"),
            encode(&data, &wrapped, &key)?,
        ));
        config[field] = json!(name);
    }
    config["name"] = json!(args.name);
    config["vendor"] = json!(args.vendor);
    config["version"] = json!(args.version);
    config["styles"] = json!([
        {"name":"(base)", "data":vector(&models["acoustic"], &[0, 1], 32)?},
        {"name":"(default)", "data":vector(&models["acoustic"], &[1, 0, 0], 32)?}
    ]);
    config["timing"] = json!(vector(&models["duration"], &[5], 128)?);
    config["pitch"] = json!(vector(&models["f0model-dds"], &[0, 1], 32)?);
    let blob = encode(&serde_json::to_vec(&config)?, config_blob, &key)?;
    files.push((
        format!("databases/{}/voice.{}.nofs", args.name, args.version),
        replace_config(&template, &blob)?,
    ));
    for (path, name) in [
        (args.profile, "profile.png"),
        (args.background, "background.png"),
    ] {
        if let Some(path) = path {
            let data = fs::read(path)?;
            ensure!(
                data.starts_with(b"\x89PNG\r\n\x1a\n"),
                "artwork must be PNG"
            );
            files.push((format!("databases/{}/{name}", args.name), data));
        }
    }
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let mut zip = ZipWriter::new(file);
    for (name, data) in files {
        zip.start_file(
            name,
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated),
        )?;
        zip.write_all(&data)?;
    }
    zip.finish()?;
    println!("{}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(kind: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut bytes = 0x7fca00ffu32.to_le_bytes().to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(0x7fca41ffu32.to_le_bytes());
        let mut tag = [0; 8];
        tag[..kind.len()].copy_from_slice(kind);
        bytes.extend(tag);
        bytes.extend(0u32.to_le_bytes());
        bytes.extend((payload.len() as u32).to_le_bytes());
        bytes.extend(payload);
        bytes
    }

    #[test]
    fn restores_quantized_nodes_without_clipping() -> Result<()> {
        let mut payload = Vec::new();
        for n in [16u32, 0, 0, 2] {
            payload.extend(n.to_le_bytes());
        }
        for v in [1.0f32, 1.0] {
            payload.extend(v.to_le_bytes());
        }
        for n in [2u32, 2] {
            payload.extend(n.to_le_bytes());
        }
        payload.extend([0; 8]);
        let template = model(b"prim4", &payload);
        let values = [-3.5f32, 0.25, 8.0, -9.0];
        let mut payload = Vec::new();
        for n in [2u32, 2] {
            payload.extend(n.to_le_bytes());
        }
        for v in values {
            payload.extend(v.to_le_bytes());
        }
        let result = Model::parse(&compatible_model(&model(b"prim2", &payload), &template)?)?;
        assert_eq!(result.nodes[0].kind, "prim4");
        for (actual, expected) in result.matrix(0)?.data.iter().zip(values) {
            assert!((actual - expected).abs() < 0.001);
        }
        Ok(())
    }

    #[test]
    fn updates_index_when_configuration_grows() -> Result<()> {
        let mut bytes = Vec::new();
        for n in [0xf580u32, 10, 0, 0, 44] {
            bytes.extend(n.to_le_bytes());
        }
        bytes.extend(0x1000u16.to_le_bytes());
        for (hash, offset) in [(1u32, 60u64), (2, 81)] {
            bytes.extend(hash.to_le_bytes());
            bytes.extend(offset.to_le_bytes());
        }
        bytes.extend([0; 10]);
        bytes.extend(44u32.to_le_bytes());
        for (key, value) in [
            (vec![0x7e, 0x7f, 0x7f, 0x7f], vec![1]),
            (b"tail".to_vec(), vec![2]),
        ] {
            bytes.extend(21u32.to_le_bytes());
            bytes.extend(1u16.to_le_bytes());
            bytes.extend(4u16.to_le_bytes());
            bytes.extend(key);
            bytes.extend(1u32.to_le_bytes());
            bytes.extend(value);
            bytes.extend(21u32.to_le_bytes());
        }
        let output = replace_config(&bytes, &[4; 20])?;
        assert_eq!(u64::from_le_bytes(output[38..46].try_into()?), 100);
        assert_eq!(read_nofs(&output)?[b"tail".as_slice()], [2]);
        bytes[26..34].copy_from_slice(&999u64.to_le_bytes());
        assert!(replace_config(&bytes, &[4; 20]).is_err());
        Ok(())
    }

    #[test]
    fn envelope_roundtrip_and_validation() -> Result<()> {
        let template = [b"SVDB".as_slice(), &[3; 16]].concat();
        let wrapped = encode(b"configuration", &template, &[7; 32])?;
        assert_eq!(decode(&wrapped, &[7; 32])?, b"configuration");
        assert!(decode(&wrapped[..21], &[7; 32]).is_err());
        for name in ["../voice", "CON", "voice.", "a/b", "a\\b", ""] {
            assert!(file_component(name).is_err());
        }
        file_component("Example Voice")?;
        Ok(())
    }
}
