mod constants;

use constants::STATE_BYTES;

use crate::constants::LANE_BYTES;

/// The internal state modified during hashing. The bit indexed by
/// `(x, y, z)` is stored at bit `z mod 8` of `state.0[25 * floor(z / 8) + 5y + x]`.
#[derive(Debug, Clone)]
pub struct State([u8; STATE_BYTES]);

impl State {
    /// Create a new trivial `State`.
    pub fn new() -> Self {
        State([0_u8; STATE_BYTES])
    }

    /// XOR bits in the state along one column.
    fn xor_column(&self, x: usize, z: usize) -> u8 {
        self.0[25 * z + x]
            ^ self.0[25 * z + 5 + x]
            ^ self.0[25 * z + 10 + x]
            ^ self.0[25 * z + 15 + x]
            ^ self.0[25 * z + 20 + x]
    }

    pub fn get_bit(&self, x: usize, y: usize, z: usize) -> u8 {
        let (high_z, low_z) = (z >> 3, z & 7);

        (self.0[25 * high_z + 5 * y + x] >> low_z) & 1
    }
    pub fn theta(&mut self) {
        // compute temp variable (some XOR)
        let mut tmp: [u8; 5 * LANE_BYTES] = [0_u8; 5 * LANE_BYTES];
        for z in 0..LANE_BYTES {
            for x in 0..5 {
                let (left_neighbor, right_neighbor) = match x {
                    0 => (4, 1),
                    4 => (3, 0),
                    _ => (x - 1, x + 1),
                };

                let down_neighbor = match z {
                    0 => LANE_BYTES - 1,
                    _ => z - 1,
                };

                tmp[5 * z + x] = self.xor_column(left_neighbor, z)
                    ^ ((self.xor_column(right_neighbor, z) << 1)
                        | (self.xor_column(right_neighbor, down_neighbor) >> 7));
            }
        }

        // update `self`
        for z in 0..LANE_BYTES {
            for y in 0..5 {
                for x in 0..5 {
                    self.0[25 * z + 5 * y + x] ^= tmp[5 * z + x];
                }
            }
        }
    }

    pub fn rho(&mut self) {
        todo!()
    }

    pub fn pi(&mut self) {
        todo!()
    }

    pub fn chi(&mut self) {
        todo!()
    }

    pub fn iota(&mut self, round_idx: usize) {
        todo!()
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_theta() {
        let mut inner_state: [u8; STATE_BYTES] = [0_u8; STATE_BYTES];
        inner_state[0] = 1;
        let mut state = State(inner_state.clone());
        state.theta();
        // println!("{state:?}");

        // state(0, 0, 0) is unchanged
        assert_eq!(state.get_bit(0, 0, 0), 1);

        // state(1, _, 0) is changed to 1
        assert_eq!(state.get_bit(1, 0, 0), 1);
        assert_eq!(state.get_bit(1, 1, 0), 1);
        assert_eq!(state.get_bit(1, 2, 0), 1);
        assert_eq!(state.get_bit(1, 3, 0), 1);
        assert_eq!(state.get_bit(1, 4, 0), 1);

        // state(4, _, 1) is changed to 1
        assert_eq!(state.get_bit(4, 0, 1), 1);
        assert_eq!(state.get_bit(4, 1, 1), 1);
        assert_eq!(state.get_bit(4, 2, 1), 1);
        assert_eq!(state.get_bit(4, 3, 1), 1);
        assert_eq!(state.get_bit(4, 4, 1), 1);
    }
}
