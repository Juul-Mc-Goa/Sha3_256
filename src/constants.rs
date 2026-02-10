/// number of `u64`s in a `State`.
pub const STATE_SIZE_U64: usize = 25; // 1600 / 64
/// number of rounds performed by the hash algorithm.
pub const N_ROUNDS: usize = 24;
/// constant used when updating the round constant when applying `iota`
/// it's exactly `1 + 2^4 + 2^5 + 2^6`
pub const ROUND_CONSTANT_MAGIC_VAL: u8 = 113;

pub const RATE_BYTES: usize = 136; // 1088 / 8
pub const OUTPUT_BYTES: usize = 32; // 256 / 8

/// Offsets used when applying `rho`
pub const OFFSETS: [[i16; 5]; 5] = [
    [0, 1, 190, 28, 91],
    [36, 300, 6, 55, 276],
    [3, 10, 171, 153, 231],
    [105, 45, 15, 21, 136],
    [210, 66, 253, 120, 78],
];
