use crate::tensor::Matrix;
use anyhow::{Result, bail, ensure};
use std::{collections::HashMap, path::Path};

pub fn u32_at(data: &[u8], offset: usize) -> Result<u32> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow::anyhow!("truncated u32 at {offset}"))?;
    Ok(u32::from_le_bytes(bytes.try_into()?))
}
pub fn f32_at(data: &[u8], offset: usize) -> Result<f32> {
    Ok(f32::from_bits(u32_at(data, offset)?))
}

#[derive(Clone, Debug)]
pub struct Node {
    pub kind: String,
    pub type_id: u64,
    pub payload: Vec<u8>,
    pub children: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct Model {
    pub version: u32,
    pub nodes: Vec<Node>,
}

const SEEDS: [u64; 12] = [
    0x0DCD59189D5A0F24,
    0x59346D79970CA21E,
    0xCF3519B773B767BF,
    0x562C8E41FB7FBEE2,
    0xBD6D25457E1ED24E,
    0x0123456789ABCDEF,
    0x76543210FEDCBA98,
    0x02468ACEECA86420,
    0xECA864202468ACEE,
    0x70556F5965766947,
    0x5A4D5A4D5A4D5A4D,
    0x000000626D6F6379,
];

impl Model {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&std::fs::read(path)?)
    }
    pub fn parse(data: &[u8]) -> Result<Self> {
        ensure!(
            data.len() <= 512 * 1024 * 1024 && u32_at(data, 0)? == 0x7fca00ff,
            "invalid DNNI header"
        );
        let version = u32_at(data, 4)?;
        ensure!(
            version == 1 || version == 2,
            "unsupported DNNI version {version}; unwrap v3 first"
        );
        let names = "prim0 prim1 prim2 prim3 prim4 prim5 modm0 modl0 modl1 modl3 modl4 modl6 moda0 moda1 moda2 moda3 moda4 moda5 moda7 _gnc1v0 _ncwnv0 cmpg1 cmpu0 cmpu1 _vocfv1 _vocfv2 _ppusv0 _ppdsv0 _psv2 _rldtg0 _rldms0 _stbkv1 _vqctx1 _didsv0 _ftmfv2 _ftmfv3 _dctov0";
        let mut types = HashMap::new();
        for name in names.split_whitespace() {
            for seed in SEEDS {
                let hash = name
                    .bytes()
                    .fold(seed, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
                types.insert(hash, name.to_string());
            }
        }
        let mut model = Self {
            version,
            nodes: Vec::new(),
        };
        let mut position = 8;
        model.parse_node(data, &mut position, 0, &types)?;
        ensure!(position == data.len(), "trailing DNNI bytes");
        Ok(model)
    }
    fn parse_node(
        &mut self,
        data: &[u8],
        pos: &mut usize,
        depth: usize,
        types: &HashMap<u64, String>,
    ) -> Result<usize> {
        ensure!(
            depth <= 256 && self.nodes.len() < 1000000,
            "DNNI tree limit"
        );
        let marker = u32_at(data, *pos)?;
        ensure!(
            marker == 0x7fca40ff || marker == 0x7fca41ff,
            "invalid DNNI node marker"
        );
        let tag = data
            .get(*pos + 4..*pos + 12)
            .ok_or_else(|| anyhow::anyhow!("truncated tag"))?;
        let type_id = u64::from_le_bytes(tag.try_into()?);
        let mut count = u32_at(data, *pos + 12)?;
        let size = u32_at(data, *pos + 16)? as usize;
        let kind = if self.version == 1 {
            let len = tag.iter().position(|b| *b == 0).unwrap_or(8);
            std::str::from_utf8(&tag[..len])?.to_owned()
        } else {
            count ^= type_id as u32 ^ (type_id >> 32) as u32 ^ 0xac5e7bd5;
            ensure!(count % 3 == 0, "invalid child count");
            count /= 3;
            types
                .get(&type_id)
                .cloned()
                .unwrap_or_else(|| format!("unknown:{type_id:016x}"))
        };
        *pos += 20;
        let payload = data
            .get(*pos..*pos + size)
            .ok_or_else(|| anyhow::anyhow!("truncated payload"))?
            .to_vec();
        *pos += size;
        ensure!(
            count as usize <= (data.len() - *pos) / 20,
            "child count out of bounds"
        );
        let index = self.nodes.len();
        self.nodes.push(Node {
            kind,
            type_id,
            payload,
            children: Vec::new(),
        });
        for _ in 0..count {
            let child = self.parse_node(data, pos, depth + 1, types)?;
            self.nodes[index].children.push(child);
        }
        Ok(index)
    }
    pub fn vector(&self, index: usize) -> Result<Vec<f32>> {
        let n = &self.nodes[index];
        ensure!(n.kind == "prim1", "expected vector, got {}", n.kind);
        let count = u32_at(&n.payload, 0)? as usize;
        ensure!(n.payload.len() == 4 + count * 4, "vector length mismatch");
        let result: Vec<f32> = n.payload[4..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(result.iter().all(|v| v.is_finite()), "non-finite vector");
        Ok(result)
    }
    pub fn set_vector(&mut self, index: usize, values: &[f32]) -> Result<()> {
        ensure!(
            self.vector(index)?.len() == values.len() && values.iter().all(|x| x.is_finite()),
            "vector shape/value mismatch"
        );
        let mut payload = (values.len() as u32).to_le_bytes().to_vec();
        for v in values {
            payload.extend(v.to_le_bytes());
        }
        self.nodes[index].payload = payload;
        Ok(())
    }
    pub fn matrix(&self, index: usize) -> Result<Matrix> {
        let node = &self.nodes[index];
        ensure!(
            ["prim0", "prim2", "prim3", "prim4", "prim5"].contains(&node.kind.as_str()),
            "expected matrix: {}",
            node.kind
        );
        let p = &node.payload;
        let quant = ["prim4", "prim5"].contains(&node.kind.as_str());
        let sparse = ["prim3", "prim5"].contains(&node.kind.as_str());
        let mut bits = 32;
        let mut scales = Vec::new();
        let mut offset = 0;
        if quant {
            bits = u32_at(p, 0)?;
            ensure!(
                (bits == 8 || bits == 16) && u32_at(p, 8)? == 0,
                "unsupported quantization"
            );
            let count = u32_at(p, 12)? as usize;
            ensure!(count <= p.len() / 4, "scale count exceeds payload");
            for i in 0..count {
                scales.push(f32_at(p, 16 + i * 4)?);
            }
            offset = 16 + count * 4;
        }
        let rows = u32_at(p, offset)? as usize;
        let cols = u32_at(p, offset + 4)? as usize;
        ensure!(
            rows > 0
                && cols > 0
                && rows
                    .checked_mul(cols)
                    .is_some_and(|n| n <= 128 * 1024 * 1024),
            "matrix dimensions out of range"
        );
        ensure!(
            !quant || scales.len() == rows,
            "quantization scale count mismatch"
        );
        let size = (bits / 8) as usize;
        let mut out = vec![0.0f32; rows * cols];
        let value = |pos: usize| -> Result<f32> {
            let b = p
                .get(pos..pos + size)
                .ok_or_else(|| anyhow::anyhow!("truncated matrix"))?;
            Ok(match bits {
                8 => (b[0] as i8) as f32,
                16 => i16::from_le_bytes(b.try_into()?) as f32,
                32 => f32::from_le_bytes(b.try_into()?),
                _ => unreachable!(),
            })
        };
        if !sparse {
            let start = offset + 8;
            ensure!(
                p.len() == start + rows * cols * size,
                "matrix byte length mismatch"
            );
            for r in 0..rows {
                for c in 0..cols {
                    let k = if quant { c * rows + r } else { r * cols + c };
                    out[r * cols + c] = value(start + k * size)?;
                }
            }
        } else {
            let br = u32_at(p, offset + 8)? as usize;
            let bc = u32_at(p, offset + 12)? as usize;
            let count = u32_at(p, offset + 16)? as usize;
            ensure!(
                br > 0 && bc > 0 && rows.is_multiple_of(br) && cols.is_multiple_of(bc),
                "unsupported sparse blocks"
            );
            let indices = offset + 20;
            let pointers = indices + count * 2;
            let start = pointers + (rows / br + 1) * 4;
            ensure!(
                count <= p.len() / size / br / bc && p.len() == start + count * br * bc * size,
                "sparse payload size mismatch"
            );
            ensure!(
                u32_at(p, pointers)? == 0 && u32_at(p, pointers + rows / br * 4)? as usize == count,
                "invalid sparse pointers"
            );
            for rb in 0..rows / br {
                let a = u32_at(p, pointers + rb * 4)? as usize;
                let b = u32_at(p, pointers + (rb + 1) * 4)? as usize;
                ensure!(a <= b && b <= count, "invalid sparse range");
                for block in a..b {
                    let cb = u16::from_le_bytes(
                        p[indices + block * 2..indices + block * 2 + 2].try_into()?,
                    ) as usize;
                    ensure!(cb < cols / bc, "sparse column out of range");
                    for c in 0..bc {
                        for r in 0..br {
                            out[(rb * br + r) * cols + cb * bc + c] +=
                                value(start + (block * br * bc + c * br + r) * size)?;
                        }
                    }
                }
            }
        }
        if quant {
            let divisor = (1u32 << (bits - 1)) as f32;
            for r in 0..rows {
                for c in 0..cols {
                    out[r * cols + c] = out[r * cols + c] / divisor * scales[r];
                }
            }
        }
        if !out.iter().all(|v| v.is_finite()) {
            bail!("non-finite matrix");
        }
        Ok(Matrix {
            rows,
            cols,
            data: out,
        })
    }
}
