use crate::{model::Model, tensor::Tensor};
use anyhow::{Result, ensure};

#[derive(Clone, Debug)]
pub struct Normalizer {
    pub lower: Vec<f32>,
    pub upper: Vec<f32>,
    pub target: [f32; 2],
}
impl Normalizer {
    pub fn load(model: &Model, index: usize) -> Result<Self> {
        let n = &model.nodes[index];
        ensure!(
            n.kind == "cmpu0" && n.children.len() == 3 && n.payload.is_empty(),
            "expected cmpu0"
        );
        let lower = model.vector(n.children[0])?;
        let upper = model.vector(n.children[1])?;
        let range = model.vector(n.children[2])?;
        ensure!(
            !lower.is_empty()
                && lower.len() == upper.len()
                && range.len() == 2
                && range[0] < range[1],
            "invalid normalizer"
        );
        ensure!(
            lower
                .iter()
                .zip(&upper)
                .all(|(a, b)| a <= b && (b - a).is_finite()),
            "invalid normalization bounds"
        );
        Ok(Self {
            lower,
            upper,
            target: [range[0], range[1]],
        })
    }
    pub fn channels(&self) -> usize {
        self.lower.len()
    }
    pub fn normalize(&self, x: &Tensor, clamp: bool) -> Result<Tensor> {
        self.transform(x, false, clamp)
    }
    pub fn denormalize(&self, x: &Tensor) -> Result<Tensor> {
        self.transform(x, true, false)
    }
    fn transform(&self, x: &Tensor, inverse: bool, clamp: bool) -> Result<Tensor> {
        ensure!(x.channels == self.channels(), "normalizer channel mismatch");
        let mut y = x.clone();
        let range = self.target[1] - self.target[0];
        for (i, v) in y.data.iter_mut().enumerate() {
            let c = i % x.channels;
            let source = self.upper[c] - self.lower[c];
            *v = if inverse {
                self.lower[c] + source * (*v - self.target[0]) / range
            } else {
                self.target[0] + range * (*v - self.lower[c]) / source.max(1e-5)
            };
            if clamp {
                *v = v.clamp(self.target[0], self.target[1]);
            }
        }
        y.validate()?;
        Ok(y)
    }
}
