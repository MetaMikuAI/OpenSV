use anyhow::{Result, ensure};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub frames: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(frames: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        ensure!(
            frames.checked_mul(channels) == Some(data.len()),
            "tensor shape mismatch"
        );
        ensure!(data.iter().all(|v| v.is_finite()), "non-finite tensor");
        Ok(Self {
            frames,
            channels,
            data,
        })
    }
    pub fn zeros(frames: usize, channels: usize) -> Self {
        Self {
            frames,
            channels,
            data: vec![0.0; frames * channels],
        }
    }
    pub fn row(&self, frame: usize) -> &[f32] {
        &self.data[frame * self.channels..(frame + 1) * self.channels]
    }
    pub fn row_mut(&mut self, frame: usize) -> &mut [f32] {
        &mut self.data[frame * self.channels..(frame + 1) * self.channels]
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.frames.checked_mul(self.channels) == Some(self.data.len()),
            "tensor shape mismatch"
        );
        ensure!(
            self.data.iter().all(|x| x.is_finite()),
            "non-finite inference output"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn column(&self, column: usize) -> Vec<f32> {
        (0..self.rows)
            .map(|r| self.data[r * self.cols + column])
            .collect()
    }
}
