#[derive(Clone)]
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}
impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }
    pub fn next_u32(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let x = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (x >> 1)
                    ^ if x & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
}
pub struct Gaussian {
    engine: Mt19937,
    spare: Option<f32>,
}
impl Gaussian {
    pub fn new(seed: u32) -> Self {
        Self {
            engine: Mt19937::new(seed),
            spare: None,
        }
    }
    pub fn sample(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        loop {
            let mut a = self.engine.next_u32() as f32 * 2f32.powi(-31) - 1.0;
            let mut b = self.engine.next_u32() as f32 * 2f32.powi(-31) - 1.0;
            let mut squared = a * a + b * b;
            if squared >= 1.0 || a == 0.0 || b == 0.0 {
                continue;
            }
            let logarithm = if squared <= 0.0001 {
                let exponent = a.abs().max(b.abs()).log2().floor() as i32;
                a *= 2f32.powi(-exponent);
                b *= 2f32.powi(-exponent);
                squared = a * a + b * b;
                squared.ln() + exponent as f32 * 1.386_294_4_f32
            } else {
                squared.ln()
            };
            let scale = (-2.0 * logarithm / squared).sqrt();
            self.spare = Some(b * scale);
            return a * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_mt_sequence() {
        let mut r = Mt19937::new(5489);
        assert_eq!(
            [r.next_u32(), r.next_u32(), r.next_u32()],
            [3499211612, 581869302, 3890346734]
        );
    }
}
