mod constants;

use std::string::FromUtf8Error;

use crate::constants::{
    N_ROUNDS, OFFSETS, OUTPUT_BYTES, RATE_BYTES, ROUND_CONSTANT_MAGIC_VAL, STATE_SIZE_U64,
};

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

    /// `XOR` bits inside columns.
    pub fn theta(&mut self) {
        // compute temp variable (some XOR)
        let mut tmp: [u64; 5] = [0; 5];
        for x in 0..5 {
            let (left_neighbor, right_neighbor) = match x {
                0 => (4, 1),
                4 => (3, 0),
                _ => (x - 1, x + 1),
            };

            // println!("x: {x}, left: {left_neighbor}, right: {right_neighbor}");
            let unshifted_xor = self.xor_column(right_neighbor);
            tmp[x] =
                self.xor_column(left_neighbor) ^ ((unshifted_xor << 1) | (unshifted_xor >> 63));
        }

        // update `self`
        for y in 0..5 {
            for x in 0..5 {
                self.0[5 * y + x] ^= tmp[x];
            }
        }
    }

    /// Rotate inside lanes.
    pub fn rho(&mut self) {
        for y in 0..5 {
            for x in 0..5 {
                let offset = OFFSETS[y][x].rem_euclid(64);
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
            rc |= round_constant(j + 7 * round_idx) << ((1 << j) - 1)
        }

        // update lane `0, 0`.
        self.0[0] ^= rc;
    }

    pub fn apply_round(&mut self, round_index: usize) {
        self.theta();
        // println!("after theta:");
        // pretty_print_bytes(&self.as_bytes());
        self.rho();
        // println!("after rho:");
        // pretty_print_bytes(&self.as_bytes());
        self.pi();
        // println!("after pi:");
        // pretty_print_bytes(&self.as_bytes());
        self.chi();
        // println!("after chi:");
        // pretty_print_bytes(&self.as_bytes());
        self.iota(round_index);
        // println!("after iota:");
        // pretty_print_bytes(&self.as_bytes());
    }

    pub fn apply_f(&mut self) {
        // println!();
        // println!("initial state:");
        // pretty_print_bytes(&self.as_bytes());

        for i in 0..N_ROUNDS {
            // println!("\nround {i}:");
            self.apply_round(i);
        }
        // println!();
    }

    pub fn as_bytes(&self) -> [u8; 8 * STATE_SIZE_U64] {
        let mut result = [0_u8; 8 * STATE_SIZE_U64];
        for y in 0..5 {
            for x in 0..5 {
                for z in 0..8 {
                    result[8 * (5 * y + x) + z] = (self.0[5 * y + x] >> (8 * z)) as u8;
                }
            }
        }
        result
    }

    pub fn xor_bytes(&mut self, other: &[u8]) {
        for (left, chunk) in self.0.iter_mut().zip(other.chunks(8)) {
            let mut temp: [u8; 8] = [0_u8; 8];
            temp.copy_from_slice(chunk);
            *left ^= u64::from_ne_bytes(temp);
        }
    }

    pub fn as_string(&self) -> Result<String, FromUtf8Error> {
        String::from_utf8(self.as_bytes().to_vec())
    }
}

impl From<&[u8]> for State {
    fn from(value: &[u8]) -> Self {
        let mut state = [0_u64; STATE_SIZE_U64];
        for (idx, chunk) in value.chunks(8).enumerate() {
            let x = idx % 5;
            let y = idx / 5;
            for (i, b) in chunk.iter().enumerate() {
                state[5 * y + x] |= (*b as u64) << (8 * i);
            }
        }

        Self(state)
    }
}

impl From<&str> for State {
    fn from(value: &str) -> Self {
        let mut state = [0_u64; STATE_SIZE_U64];
        for (idx, chunk) in value.as_bytes().chunks(8).enumerate() {
            let x = idx % 5;
            let y = idx / 5;

            let mut temp: [u8; 8] = [0_u8; 8];
            temp.copy_from_slice(chunk);
            state[5 * y + x] = u64::from_ne_bytes(temp);
        }

        Self(state)
    }
}

fn round_constant(mut t: usize) -> u64 {
    t = t % 255;

    if t == 0 {
        return 1;
    }

    let mut r: u8 = 1;

    for _ in 0..t {
        if (r >> 7) == 1 {
            r = (r << 1) ^ ROUND_CONSTANT_MAGIC_VAL;
        } else {
            r <<= 1;
        }
    }

    (r & 1) as u64
}

pub fn xor_bytes(left: &mut [u8], right: &[u8]) {
    for (l, r) in left.iter_mut().zip(right.iter()) {
        *l ^= r;
    }
}

pub fn pad(input: &mut Vec<u8>, bit_len: usize, required_len: usize) {
    let last_idx = input.len();
    input.push(0);

    // add padding: 0b110
    let first_free_bit_idx = bit_len & 7;
    if first_free_bit_idx == 0 {
        input.push(0);
        input[last_idx] = 6;
    } else if first_free_bit_idx > 5 {
        input[last_idx - 1] |= 6 << first_free_bit_idx;
        input.push(6 >> first_free_bit_idx);
    } else {
        input[last_idx - 1] |= 6 << first_free_bit_idx;
    }

    // input + 0b110 + (`count` zeros) + 0b1 should be a multiple of 8 * required_len
    let mut count = (-(bit_len as i64) - 4).rem_euclid(8 * required_len as i64) as usize;
    count >>= 3;

    let padded_last_idx = last_idx + count;
    input.extend((0..count).map(|_| 0));

    // set last bit to 1
    input[padded_last_idx] |= 1 << 7;
}

// fn bytes_to_hex(bytes: &[u8]) -> String {
//     bytes
//         .iter()
//         .map(|b| format!("{:02x}", b))
//         .collect::<String>()
// }

#[allow(dead_code)]
fn pretty_print_bytes(bytes: &[u8]) {
    for chunk in bytes.chunks(16) {
        let string = chunk
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();

        println!("{string}");
    }
}

pub fn sponge(input: &[u8], bit_len: usize) -> String {
    // pad
    let mut padded_input: Vec<u8> = input.to_vec();
    pad(&mut padded_input, bit_len, RATE_BYTES);
    // println!("padded_input:");
    // pretty_print_bytes(&padded_input);
    // println!("padded length: {}", padded_input.len());

    // absorb
    let mut state = State::new();
    for chunk in padded_input.chunks_exact(RATE_BYTES) {
        // println!("state:");
        // pretty_print_bytes(&state.as_bytes());
        // println!("chunk to xor");
        // pretty_print_bytes(chunk);
        state.xor_bytes(chunk);
        // println!("after xor");
        // pretty_print_bytes(&state.as_bytes());
        state.apply_f();
        // println!("after f: ");
        // pretty_print_bytes(&state.as_bytes());
    }

    // squeeze
    let mut result: Vec<u8> = state.as_bytes()[0..RATE_BYTES].to_vec();
    while result.len() < OUTPUT_BYTES {
        state.apply_f();
        result.append(&mut state.as_bytes()[0..RATE_BYTES].to_vec());
    }

    // truncate
    let mut result_string = String::new();
    // use hex representation
    result_string.extend(result[0..OUTPUT_BYTES].iter().map(|c| format!("{:02x}", c)));

    result_string
}

pub fn sha3_256(input: &str) -> String {
    let bytes = input.as_bytes();
    let bit_len = 8 * bytes.len();
    sponge(&input.as_bytes(), bit_len)
}

fn main() {
    // let input = [0x13_u8];
    // let bit_len = 5;
    // println!("input: {input:x?}");

    // let output = sponge(&input, bit_len);
    // println!("output: {output}");

    let input = String::from("test");
    let output = sha3_256(&input);
    println!("\noutput: {}", output);
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
