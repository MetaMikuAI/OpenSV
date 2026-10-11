use crate::{
    model::{Model, f32_at, u32_at},
    tensor::{Matrix, Tensor},
};
use anyhow::{Result, bail, ensure};

#[derive(Clone, Debug)]
struct Packed {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}
impl Packed {
    fn new(m: Matrix) -> Self {
        let mut data = vec![0.0; m.data.len()];
        for c in 0..m.cols {
            for r in 0..m.rows {
                data[c * m.rows + r] = m.data[r * m.cols + c];
            }
        }
        Self {
            rows: m.rows,
            cols: m.cols,
            data,
        }
    }
    fn add(&self, x: &[f32], y: &mut [f32]) {
        for (column, &v) in x.iter().enumerate() {
            let w = &self.data[column * self.rows..(column + 1) * self.rows];
            for (out, weight) in y.iter_mut().zip(w) {
                *out += *weight * v;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Network {
    op: Op,
}
#[derive(Clone, Debug)]
enum Op {
    Sequence(Vec<Network>),
    Linear {
        kernels: Vec<Packed>,
        bias: Vec<f32>,
        stride: usize,
        padding: usize,
        dilation: usize,
    },
    Gru {
        weights: Vec<Packed>,
        biases: Vec<Vec<f32>>,
    },
    BiGru(Box<Network>, Box<Network>),
    Gate {
        input: usize,
        hidden: usize,
        condition: usize,
        parts: Vec<Network>,
    },
    Residual {
        input: usize,
        hidden: usize,
        condition: usize,
        stages: Vec<(Network, Network, Network)>,
        projection: Option<Box<Network>>,
    },
    Activation(String, f32),
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Network {
    pub fn load(model: &Model, index: usize) -> Result<Self> {
        let node = model
            .nodes
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("network root out of range"))?;
        let p = &node.payload;
        let c = &node.children;
        let op =
            match node.kind.as_str() {
                "modm0" => Op::Sequence(
                    c.iter()
                        .map(|i| Self::load(model, *i))
                        .collect::<Result<_>>()?,
                ),
                "modl0" | "modl1" => {
                    let (kernel, stride, padding, dilation) = if node.kind == "modl1" {
                        ensure!(
                            p.len() == 20 && u32_at(p, 16)? == 1,
                            "unsupported convolution"
                        );
                        (
                            u32_at(p, 0)? as usize,
                            u32_at(p, 4)? as usize,
                            u32_at(p, 8)? as usize,
                            u32_at(p, 12)? as usize,
                        )
                    } else {
                        ensure!(p.is_empty(), "dense payload unexpected");
                        (1, 1, 0, 1)
                    };
                    ensure!(
                        kernel > 0 && stride > 0 && dilation > 0 && padding < 1000000,
                        "invalid convolution dimensions"
                    );
                    let mut kernels = Vec::new();
                    let mut bias = Vec::new();
                    for &child in c {
                        if model.nodes[child].kind == "prim1" {
                            ensure!(bias.is_empty(), "duplicate bias");
                            bias = model.vector(child)?;
                        } else {
                            kernels.push(Packed::new(model.matrix(child)?));
                        }
                    }
                    ensure!(
                        kernels.len() == kernel
                            && kernels
                                .iter()
                                .all(|m| m.rows == kernels[0].rows && m.cols == kernels[0].cols),
                        "kernel count/shape mismatch"
                    );
                    ensure!(
                        bias.is_empty() || bias.len() == kernels[0].rows,
                        "bias size mismatch"
                    );
                    if bias.is_empty() {
                        bias.resize(kernels[0].rows, 0.0);
                    }
                    Op::Linear {
                        kernels,
                        bias,
                        stride,
                        padding,
                        dilation,
                    }
                }
                "modl3" => {
                    ensure!(p.is_empty() && c.len() == 12, "unsupported GRU");
                    let mut weights = Vec::new();
                    let mut biases = Vec::new();
                    for pair in c.chunks_exact(2) {
                        weights.push(Packed::new(model.matrix(pair[0])?));
                        biases.push(model.vector(pair[1])?);
                    }
                    let hidden = weights[0].rows;
                    let input = weights[0].cols;
                    ensure!(
                        weights.iter().enumerate().all(|(i, w)| w.rows == hidden
                            && w.cols == if i < 3 { input } else { hidden })
                            && biases.iter().all(|b| b.len() == hidden),
                        "GRU shape mismatch"
                    );
                    Op::Gru { weights, biases }
                }
                "modl6" => {
                    ensure!(
                        p.is_empty()
                            && c.len() == 2
                            && c.iter().all(|i| model.nodes[*i].kind == "modl3"),
                        "invalid BiGRU"
                    );
                    Op::BiGru(
                        Box::new(Self::load(model, c[0])?),
                        Box::new(Self::load(model, c[1])?),
                    )
                }
                "_gnc1v0" | "_ncwnv0" => {
                    ensure!(p.len() == 12, "invalid conditional network");
                    let input = u32_at(p, 0)? as usize;
                    let hidden = u32_at(p, 4)? as usize;
                    let condition = u32_at(p, 8)? as usize;
                    ensure!(input > 0 && hidden > 0, "invalid channels");
                    if node.kind == "_gnc1v0" {
                        ensure!(
                            c.len() == if condition > 0 { 2 } else { 1 },
                            "gate child count"
                        );
                        Op::Gate {
                            input,
                            hidden,
                            condition,
                            parts: c
                                .iter()
                                .map(|i| Self::load(model, *i))
                                .collect::<Result<_>>()?,
                        }
                    } else {
                        ensure!(
                            c.len() == if condition > 0 { 4 } else { 3 },
                            "residual child count"
                        );
                        let groups: Vec<_> = c[..3].iter().map(|i| &model.nodes[*i]).collect();
                        let count = groups[0].children.len();
                        ensure!(
                            count > 0
                                && groups
                                    .iter()
                                    .all(|g| g.kind == "cmpg1" && g.children.len() == count),
                            "invalid residual groups"
                        );
                        let mut stages = Vec::new();
                        for i in 0..count {
                            stages.push((
                                Self::load(model, groups[0].children[i])?,
                                Self::load(model, groups[1].children[i])?,
                                Self::load(model, groups[2].children[i])?,
                            ));
                        }
                        let projection = if condition > 0 {
                            Some(Box::new(Self::load(model, c[3])?))
                        } else {
                            None
                        };
                        Op::Residual {
                            input,
                            hidden,
                            condition,
                            stages,
                            projection,
                        }
                    }
                }
                "moda0" | "moda1" | "moda2" | "moda3" | "moda4" | "moda5" | "moda7" => {
                    ensure!(c.is_empty(), "activation children unexpected");
                    let alpha = if node.kind == "moda3" || node.kind == "moda4" {
                        ensure!(p.len() == 4, "activation parameter");
                        f32_at(p, 0)?
                    } else {
                        ensure!(p.is_empty(), "activation payload");
                        0.0
                    };
                    Op::Activation(node.kind.clone(), alpha)
                }
                _ => bail!("unsupported operator {} at node {index}", node.kind),
            };
        Ok(Self { op })
    }
    pub fn run(&self, x: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        x.validate()?;
        let y = match &self.op {
            Op::Sequence(parts) => {
                let mut out = x.clone();
                for p in parts {
                    out = p.run(&out, condition)?;
                }
                out
            }
            Op::Linear {
                kernels,
                bias,
                stride,
                padding,
                dilation,
            } => {
                ensure!(
                    x.channels == kernels[0].cols,
                    "linear input channels mismatch"
                );
                let extent = dilation * (kernels.len() - 1) + 1;
                let frames = if x.frames + 2 * padding < extent {
                    0
                } else {
                    (x.frames + 2 * padding - extent) / stride + 1
                };
                ensure!(
                    frames
                        .checked_mul(bias.len())
                        .is_some_and(|n| n <= 128 * 1024 * 1024),
                    "convolution output limit"
                );
                let mut y = Tensor::zeros(frames, bias.len());
                for frame in 0..frames {
                    let out = y.row_mut(frame);
                    for (k, m) in kernels.iter().enumerate() {
                        let pos = frame * stride + k * dilation;
                        if pos >= *padding && pos - padding < x.frames {
                            m.add(x.row(pos - padding), out);
                        }
                    }
                    for (v, b) in out.iter_mut().zip(bias) {
                        *v += *b;
                    }
                }
                y
            }
            Op::Gru { .. } => self.run_gru(x, false)?,
            Op::BiGru(a, b) => {
                let a = a.run_gru(x, false)?;
                let b = b.run_gru(x, true)?;
                let mut y = Tensor::zeros(x.frames, a.channels + b.channels);
                for f in 0..x.frames {
                    let row = y.row_mut(f);
                    row[..a.channels].copy_from_slice(a.row(f));
                    row[a.channels..].copy_from_slice(b.row(f));
                }
                y
            }
            Op::Gate {
                input,
                hidden,
                condition: channels,
                parts,
            } => {
                ensure!(x.channels == *input, "gate channels mismatch");
                let mut raw = parts[0].run(x, None)?;
                ensure!(raw.channels == hidden * 2, "gate output mismatch");
                if *channels > 0 {
                    let c = condition.ok_or_else(|| anyhow::anyhow!("condition required"))?;
                    ensure!(c.channels == *channels, "condition channels mismatch");
                    let q = parts[1].run(c, None)?;
                    ensure!(
                        q.frames == raw.frames && q.channels == raw.channels,
                        "condition shape mismatch"
                    );
                    for (a, b) in raw.data.iter_mut().zip(q.data) {
                        *a += b;
                    }
                }
                let mut y = Tensor::zeros(raw.frames, *hidden);
                for f in 0..raw.frames {
                    for h in 0..*hidden {
                        y.data[f * hidden + h] = raw.data[f * hidden * 2 + h].tanh()
                            * sigmoid(raw.data[f * hidden * 2 + hidden + h]);
                    }
                }
                y
            }
            Op::Residual {
                input,
                hidden,
                condition: channels,
                stages,
                projection,
            } => {
                ensure!(x.channels == *input, "residual input channels mismatch");
                let mut y = if let Some(p) = projection {
                    let c = condition.ok_or_else(|| anyhow::anyhow!("condition required"))?;
                    ensure!(c.channels == *channels, "condition channels mismatch");
                    p.run(c, None)?
                } else {
                    Tensor::zeros(x.frames, *hidden)
                };
                let mut current = x.clone();
                for (gate, skip, residual) in stages {
                    let mut gated = gate.run(&current, condition)?;
                    let r = residual.run(&current, None)?;
                    let s = skip.run(&gated, None)?;
                    ensure!(
                        gated.frames == y.frames
                            && gated.channels == y.channels
                            && r.frames == y.frames
                            && r.channels == y.channels
                            && s.frames == y.frames
                            && s.channels == y.channels,
                        "residual branch mismatch"
                    );
                    for (a, b) in y.data.iter_mut().zip(s.data) {
                        *a += b;
                    }
                    for (a, b) in gated.data.iter_mut().zip(r.data) {
                        *a += b;
                    }
                    current = gated;
                }
                y
            }
            Op::Activation(kind, alpha) => {
                let mut y = x.clone();
                for v in &mut y.data {
                    *v = match kind.as_str() {
                        "moda0" => v.max(0.0),
                        "moda1" => v.tanh(),
                        "moda2" => sigmoid(*v),
                        "moda3" => {
                            if *v > 0.0 {
                                *v
                            } else {
                                *alpha * *v
                            }
                        }
                        "moda4" => v.max(0.0) + (*alpha * v.exp_m1()).min(0.0),
                        "moda7" => *v * sigmoid(*v),
                        _ => *v,
                    };
                }
                y
            }
        };
        y.validate()?;
        Ok(y)
    }
    fn run_gru(&self, x: &Tensor, reverse: bool) -> Result<Tensor> {
        let Op::Gru { weights, biases } = &self.op else {
            bail!("expected GRU")
        };
        ensure!(x.channels == weights[0].cols, "GRU input channels mismatch");
        let hidden = weights[0].rows;
        let mut state = vec![0.0; hidden];
        let mut output = Tensor::zeros(x.frames, hidden);
        let mut q = vec![vec![0.0; hidden]; 6];
        for t in 0..x.frames {
            let f = if reverse { x.frames - 1 - t } else { t };
            for i in 0..6 {
                q[i].fill(0.0);
                weights[i].add(if i < 3 { x.row(f) } else { &state }, &mut q[i]);
            }
            for h in 0..hidden {
                let reset = sigmoid(biases[0][h] + biases[3][h] + q[0][h] + q[3][h]);
                let update = sigmoid(biases[1][h] + biases[4][h] + q[1][h] + q[4][h]);
                let candidate = (reset * (biases[5][h] + q[5][h]) + q[2][h] + biases[2][h]).tanh();
                state[h] = update * state[h] + (1.0 - update) * candidate;
            }
            output.row_mut(f).copy_from_slice(&state);
        }
        Ok(output)
    }
}
