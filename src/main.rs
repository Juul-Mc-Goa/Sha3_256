mod constants;

use constants::STATE_SIZE_U64;

use crate::constants::{OFFSETS, ROUND_CONSTANT_MAGIC_VAL};

/// The internal state modified during hashing. The bit indexed by
/// `(x, y, z)` is stored at bit `z` of `state.0[5y + x]`.
#[derive(Debug, Clone)]
pub struct State([u64; STATE_SIZE_U64]);

impl State {
    /// Create a new trivial `State`.
    pub fn new() -> Self {
        State([0_u64; STATE_SIZE_U64])
    }

    /// XOR bits in the state along one column.
    fn xor_column(&self, x: usize) -> u64 {
        self.0[x] ^ self.0[5 + x] ^ self.0[10 + x] ^ self.0[15 + x] ^ self.0[20 + x]
    }

    pub fn get_bit(&self, x: usize, y: usize, z: usize) -> u64 {
        (self.0[5 * y + x] >> z) & 1
    }

    /// `XOR` bits inside columns, update `self`.
    pub fn theta(&mut self) {
        // compute temp variable (some XOR)
        // let mut tmp: [u8; 5 * LANE_BYTES] = [0_u8; 5 * LANE_BYTES];
        let mut tmp: [u64; 5] = [0; 5];
        for x in 0..5 {
            let (left_neighbor, right_neighbor) = match x {
                0 => (4, 1),
                4 => (3, 0),
                _ => (x - 1, x + 1),
            };

            tmp[x] = self.xor_column(left_neighbor)
                ^ ((self.xor_column(right_neighbor) << 1)
                    | (self.xor_column(right_neighbor) >> 63));
        }

        // update `self`
        for y in 0..5 {
            for x in 0..5 {
                self.0[5 * y + x] ^= tmp[x];
            }
        }
    }

    /// Rotate inside lanes, update `self`.
    pub fn rho(&mut self) {
        for y in 0..5 {
            for x in 0..5 {
                let offset = (-OFFSETS[y][x]).rem_euclid(64);
                if offset != 0 {
                    let lane = &mut self.0[5 * y + x];
                    *lane = (*lane << offset) | (*lane >> (64 - offset));
                }
            }
        }
    }

    /// Permute the different lanes.
    pub fn pi(&mut self) {
        let mut new_state = [0_u64; STATE_SIZE_U64];

        for y in 0..5 {
            for x in 0..5 {
                let new_x = (x + 3 * y) % 5;
                let new_y = x;
                new_state[5 * y + x] = self.0[5 * new_y + new_x]
            }
        }

        self.0.copy_from_slice(&new_state);
    }

    /// Apply a non-linear function to the state, using nearby rows as arguments.
    pub fn chi(&mut self) {
        let mut new_state = [0_u64; STATE_SIZE_U64];

        for y in 0..5 {
            for x in 0..5 {
                let (x_1, x_2) = ((x + 1) % 5, (x + 2) % 5);
                new_state[5 * y + x] =
                    self.0[5 * y + x] ^ (!self.0[5 * y + x_1] & self.0[5 * y + x_2]);
            }
        }

        self.0.copy_from_slice(&new_state);
    }

    /// Modify some bits in the lane `0, 0`.
    pub fn iota(&mut self, round_idx: usize) {
        // compute the round constant
        let mut rc: u64 = 0;

        for j in 0..=6 {
            // shift by `2^j - 1`
            rc |= round_constant(j + round_idx) << ((1 << j) - 1)
        }

        // update lane `0, 0`.
        self.0[0] ^= rc;
    }

    pub fn apply_round(&mut self, round_index: usize) {
        self.theta();
        self.rho();
        self.pi();
        self.chi();
        self.iota(round_index);
    }
}

fn round_constant(mut t: usize) -> u64 {
    let mut r: u8 = 1;

    t = t % 255;

    if t == 0 {
        return 1;
    }

    for _ in 1..=t {
        if (r >> 7) == 1 {
            r = (r << 1) ^ ROUND_CONSTANT_MAGIC_VAL;
        } else {
            r <<= 1;
        }
    }

    (r & 1) as u64
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_theta() {
        let mut inner_state: [u64; STATE_SIZE_U64] = [0; STATE_SIZE_U64];
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

    #[test]
    fn simple_rho() {
        let mut inner_state: [u64; STATE_SIZE_U64] = [0; STATE_SIZE_U64];
        inner_state[1] = 1;
        let mut state = State(inner_state.clone());
        state.rho();

        assert_eq!(state.get_bit(1, 0, 63), 1);
    }

    #[test]
    fn simple_pi() {
        let mut inner_state: [u64; STATE_SIZE_U64] = [0; STATE_SIZE_U64];
        for i in 0..STATE_SIZE_U64 {
            inner_state[i] = i as u64;
        }
        let mut state = State(inner_state.clone());
        state.pi();

        for i in 0..STATE_SIZE_U64 {
            let (x, y) = (i as u64 % 5, i as u64 / 5);
            let (x_1, y_1) = ((x + 3 * y) % 5, x);
            assert_eq!(state.0[i], 5 * y_1 + x_1);
        }
    }
}
