/// number of bytes in a `State`.
pub const STATE_BYTES: usize = 200;
/// length of a lane inside a `State`.
pub const LANE_SIZE: usize = 64;
/// length (in bytes) of a lane inside a `State`.
pub const LANE_BYTES: usize = LANE_SIZE >> 3;
/// number of rounds performed by the hash algorithm.
pub const N_ROUNDS: usize = 24;
