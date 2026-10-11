use crate::model::{Model, u32_at};
use anyhow::{Result, ensure};

#[derive(Clone, Debug)]
pub struct PhoneSet {
    pub name: String,
    pub symbols: Vec<String>,
    pub categories: Vec<String>,
    pub unified: Vec<String>,
    pub classes: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct TimedPhoneme {
    pub language: String,
    pub symbol: String,
    pub frames: usize,
}

impl PhoneSet {
    pub fn load(model: &Model, index: usize) -> Result<Self> {
        let n = &model.nodes[index];
        ensure!(n.kind == "_psv2" && n.children.is_empty(), "expected _psv2");
        let mut p = 0;
        fn string(b: &[u8], p: &mut usize) -> Result<String> {
            let len = u32_at(b, *p)? as usize;
            *p += 4;
            ensure!(len <= 4096 && *p + len <= b.len(), "invalid phone string");
            let value = std::str::from_utf8(&b[*p..*p + len])?.to_string();
            *p += len;
            ensure!(!value.contains('\0'), "null in phone string");
            Ok(value)
        }
        fn list(b: &[u8], p: &mut usize) -> Result<Vec<String>> {
            let n = u32_at(b, *p)? as usize;
            *p += 4;
            ensure!(n <= 4096, "phone list limit");
            (0..n).map(|_| string(b, p)).collect()
        }
        let b = &n.payload;
        let out = Self {
            name: string(b, &mut p)?,
            symbols: list(b, &mut p)?,
            categories: list(b, &mut p)?,
            unified: list(b, &mut p)?,
            classes: list(b, &mut p)?,
        };
        ensure!(
            p == b.len()
                && !out.symbols.is_empty()
                && out.symbols.len() == out.categories.len()
                && out.unified.len() >= out.symbols.len(),
            "invalid phone table"
        );
        Ok(out)
    }
    pub fn index(&self, symbol: &str) -> Result<usize> {
        self.symbols
            .iter()
            .position(|s| s == symbol)
            .ok_or_else(|| anyhow::anyhow!("unknown phoneme {symbol} in {}", self.name))
    }
}
pub fn language_index(sets: &[PhoneSet], language: &str) -> Result<usize> {
    ensure!(!language.is_empty(), "language required");
    let matches: Vec<_> = sets
        .iter()
        .enumerate()
        .filter(|(_, s)| s.name.contains(language))
        .map(|(i, _)| i)
        .collect();
    ensure!(
        matches.len() == 1,
        "unknown or ambiguous language {language}"
    );
    Ok(matches[0])
}
