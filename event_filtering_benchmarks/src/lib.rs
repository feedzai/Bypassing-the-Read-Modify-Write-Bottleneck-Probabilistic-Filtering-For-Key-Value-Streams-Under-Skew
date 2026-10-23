pub mod event_stream;
pub mod rocksdb_utils;
pub mod time_decayed_sum;

pub use event_stream::{EventStreamGenerator, StreamConfig, StreamStats};
pub use time_decayed_sum::TimeDecayedSum;

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct PersistencePathState {
    pub aggregations: Vec<TimeDecayedSum>,
    pub count: TimeDecayedSum,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Command {
    Unfiltered {
        key: Vec<u8>,
        timestamp: f64,
    },
    PersistencePath {
        key: Vec<u8>,
        timestamp: f64,
        threshold: f64,
    },
    FullStreamFiltering {
        key: Vec<u8>,
        timestamp: f64,
        threshold: f64,
    },
    Shutdown {
        rep: usize,
        timestamp_folder: String,
        endpoint_name: String,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Response {
    pub success: bool,
}
