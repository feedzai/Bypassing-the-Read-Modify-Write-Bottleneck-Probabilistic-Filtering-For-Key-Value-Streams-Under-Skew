use rand::Rng;
use rkyv::{Deserialize as RkyvDeserialize, Infallible};
use std::collections::HashMap;
use std::fs;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use tempfile::TempDir;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use event_filtering_benchmarks::{
    rocksdb_utils::{create_hotkey_optimized_db, extract_and_save_db_stats, print_raw_db_stats},
    Command, PersistencePathState, Response, TimeDecayedSum,
};

pub const N_AGGREGATIONS: usize = 100; // 100 aggregations standard in ml workflows.
pub const DEFAULT_HALF_LIFE: f64 = 50.0;
pub const SOCKET_PATH: &str = "/tmp/rocksdb_bench.sock";

pub struct ServerState {
    pub in_memory_counts: Mutex<HashMap<Vec<u8>, TimeDecayedSum>>,
}

pub fn unfiltered_endpoint(db: &rocksdb::DB, key: &[u8], timestamp: f64, n_aggregations: usize) {
    let db_bytes = db.get(key).expect("RocksDB read error");

    let mut aggregations = if let Some(bytes) = db_bytes {
        let archived = unsafe { rkyv::archived_root::<Vec<TimeDecayedSum>>(&bytes) };
        archived.deserialize(&mut Infallible).unwrap()
    } else {
        vec![TimeDecayedSum::new(DEFAULT_HALF_LIFE); n_aggregations]
    };

    // Unconditionally update
    for agg in aggregations.iter_mut() {
        agg.update(timestamp, 1.0);
    }

    // Unconditionally write
    let new_bytes = rkyv::to_bytes::<_, 4096>(&aggregations).unwrap();
    db.put(key, new_bytes).expect("RocksDB write error");
}

pub fn persistence_path_endpoint(
    db: &rocksdb::DB,
    key: &[u8],
    timestamp: f64,
    n_aggregations: usize,
    threshold: f64,
) {
    let db_bytes = db.get(key).expect("RocksDB read error");

    let mut state = if let Some(bytes) = db_bytes {
        let archived = unsafe { rkyv::archived_root::<PersistencePathState>(&bytes) };
        archived.deserialize(&mut Infallible).unwrap()
    } else {
        PersistencePathState {
            aggregations: vec![TimeDecayedSum::new(DEFAULT_HALF_LIFE); n_aggregations],
            count: TimeDecayedSum::new(DEFAULT_HALF_LIFE),
        }
    };

    // 2. Compute probability on the native Rust struct
    let projected_count = state.count.projected_value(timestamp, 1.0);
    let prob = (threshold / projected_count).min(1.0);

    // 3. Unconditionally update state so it's accurate for the caller
    state.count.update(timestamp, prob);
    for agg in state.aggregations.iter_mut() {
        agg.update(timestamp, prob);
    }

    // 4. Probabilistic Write: Only shed the disk I/O load
    if rand::thread_rng().gen::<f64>() < prob {
        let new_bytes = rkyv::to_bytes::<_, 4096>(&state).unwrap();
        db.put(key, new_bytes).expect("RocksDB write error");
    }
}

pub fn full_stream_filtering_endpoint(
    db: &rocksdb::DB,
    server_state: &ServerState,
    key: &[u8],
    timestamp: f64,
    n_aggregations: usize,
    threshold: f64,
) {
    let db_bytes = db.get(key).expect("RocksDB read error");

    // 1. Unconditionally deserialize to have the struct ready for prod response
    let mut aggregations = if let Some(bytes) = db_bytes {
        let archived = unsafe { rkyv::archived_root::<Vec<TimeDecayedSum>>(&bytes) };
        archived.deserialize(&mut Infallible).unwrap()
    } else {
        vec![TimeDecayedSum::new(DEFAULT_HALF_LIFE); n_aggregations]
    };

    // 2. Compute probability using purely in-memory state
    let prob = {
        let mut counts = server_state.in_memory_counts.lock().unwrap();
        let count_entry = counts
            .entry(key.to_vec())
            .or_insert_with(|| TimeDecayedSum::new(DEFAULT_HALF_LIFE));

        count_entry.update(timestamp, 1.0);
        (threshold / count_entry.value()).min(1.0)
    };

    // 3. Unconditionally update state so it's accurate for the caller
    for agg in aggregations.iter_mut() {
        agg.update(timestamp, prob);
    }

    // 4. Probabilistic Write: Only shed the disk I/O load
    if rand::thread_rng().gen::<f64>() < prob {
        let new_bytes = rkyv::to_bytes::<_, 4096>(&aggregations).unwrap();
        db.put(key, new_bytes).expect("RocksDB write error");
    }
}

fn handle_client_connection(
    mut stream: UnixStream,
    db: Arc<rocksdb::DB>,
    server_state: Arc<ServerState>,
) {
    loop {
        let cmd: Command = match bincode::deserialize_from(&mut stream) {
            Ok(c) => c,
            Err(_) => break,
        };

        match cmd {
            Command::Unfiltered { key, timestamp } => {
                unfiltered_endpoint(&db, &key, timestamp, N_AGGREGATIONS);
                let resp = Response { success: true };
                let _ = bincode::serialize_into(&mut stream, &resp);
            }
            Command::PersistencePath {
                key,
                timestamp,
                threshold,
            } => {
                persistence_path_endpoint(&db, &key, timestamp, N_AGGREGATIONS, threshold);
                let resp = Response { success: true };
                let _ = bincode::serialize_into(&mut stream, &resp);
            }
            Command::FullStreamFiltering {
                key,
                timestamp,
                threshold,
            } => {
                full_stream_filtering_endpoint(
                    &db,
                    &server_state,
                    &key,
                    timestamp,
                    N_AGGREGATIONS,
                    threshold,
                );
                let resp = Response { success: true };
                let _ = bincode::serialize_into(&mut stream, &resp);
            }
            Command::Shutdown {
                rep,
                timestamp_folder,
                endpoint_name,
            } => {
                println!("[Server] Received Shutdown signal. Flushing DB and saving stats...");

                let stats_dir = PathBuf::from("./experiment_results")
                    .join(&timestamp_folder)
                    .join(&endpoint_name);
                let _ = fs::create_dir_all(&stats_dir);

                let stats_path = stats_dir.join(format!("db_stats_rep_{}.csv", rep));

                if let Err(e) = extract_and_save_db_stats(&db, &stats_path, rep) {
                    eprintln!("[Server] Error saving DB stats: {}", e);
                }

                print_raw_db_stats(&db);
                let resp = Response { success: true };
                let _ = bincode::serialize_into(&mut stream, &resp);

                let _ = fs::remove_file(SOCKET_PATH);
                println!("[Server] Clean shutdown complete.");

                // Force the server process to exit so runner script moves to the next endpoint
                std::process::exit(0);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(SOCKET_PATH).exists() {
        fs::remove_file(SOCKET_PATH)?;
    }

    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("rocksdb_bench");
    let db = Arc::new(create_hotkey_optimized_db(db_path.to_str().unwrap()));

    let server_state = Arc::new(ServerState {
        in_memory_counts: Mutex::new(HashMap::with_capacity(200_000)),
    });

    let listener = UnixListener::bind(SOCKET_PATH)?;
    println!("[Server] Engine active on UDS: {}", SOCKET_PATH);

    while let Ok((stream, _)) = listener.accept() {
        let db_clone = Arc::clone(&db);
        let state_clone = Arc::clone(&server_state);

        thread::spawn(move || {
            handle_client_connection(stream, db_clone, state_clone);
        });
    }

    Ok(())
}
