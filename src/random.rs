use std::cell::Cell;

pub struct MyRandom {
    rng_state: Cell<u64>,
}

impl MyRandom {
    pub fn new(seed: u64) -> Self {
        assert_ne!(seed, 0);
        Self {
            rng_state: Cell::new(seed),
        }
    }

    /// it is safe to use next_u32 from another MyRandom, it will generate a different sequence
    pub fn new_from_u32(seed: u32) -> MyRandom {
        Self::new(((seed as u64) << 32) + (seed as u64))
    }

    pub fn next_u32(&self) -> u32 {
        let mut x = self.rng_state.get();

        // xorshift64* example PRNG
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state.set(x);

        ((x.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32
    }

    pub fn next_f32(&self) -> f32 {
        let x = self.next_u32() >> 8; // keep top 24 bits
        (x as f32) * (1.0 / 16_777_216.0) // 2^24
    }
}