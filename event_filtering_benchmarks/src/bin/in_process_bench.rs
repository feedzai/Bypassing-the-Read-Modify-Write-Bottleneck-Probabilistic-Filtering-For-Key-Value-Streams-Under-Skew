use chrono::Local;
use hdrhistogram::Histogram;
use rand::Rng;
use rkyv::{Deserialize as RkyvDeserialize, Infallible};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::env;
use std::fs::{create_dir_all, File};
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Instant;
use tempfile::TempDir;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use event_filtering_benchmarks::event_stream::StreamStats;
use event_filtering_benchmarks::{
    rocksdb_utils::{create_hotkey_optimized_db, extract_and_save_db_stats, print_raw_db_stats},
    EventStreamGenerator, PersistencePathState, StreamConfig, TimeDecayedSum,
};

pub const N_AGGREGATIONS: usize = 100;
pub const DEFAULT_HALF_LIFE: f64 = 50.0;
pub const EXPERIMENT_FOLDER: &str = "./experiment_results";
pub const DEFAULT_BATCH_SIZE: usize = 100;

#[derive(Clone)]
pub struct EventPayload {
    pub key: Vec<u8>,
    pub timestamp: f64,
}

pub struct BatchedState {
    pub cache: HashMap<Vec<u8>, Vec<TimeDecayedSum>>,
    pub current_batch_count: usize,
}

pub struct ServerState {
    pub in_memory_counts: Mutex<HashMap<Vec<u8>, TimeDecayedSum>>,
    pub batched_state: Mutex<BatchedState>,
}

fn hash_key(key: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

// -----------------------------------------------------------------------------
// Endpoint Business Logic
// -----------------------------------------------------------------------------
pub fn unfiltered_endpoint(db: &rocksdb::DB, key: &[u8], timestamp: f64, n_aggregations: usize) {
    let db_bytes = db.get(key).expect("RocksDB read error");

    let mut aggregations = if let Some(bytes) = db_bytes {
        let archived = unsafe { rkyv::archived_root::<Vec<TimeDecayedSum>>(&bytes) };
        archived.deserialize(&mut Infallible).unwrap()
    } else {
        vec![TimeDecayedSum::new(DEFAULT_HALF_LIFE); n_aggregations]
    };

    for agg in aggregations.iter_mut() {
        agg.update(timestamp, 1.0);
    }

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

    let projected_count = state.count.projected_value(timestamp, 1.0);
    let prob = (threshold / projected_count).min(1.0);

    state.count.update(timestamp, prob);
    for agg in state.aggregations.iter_mut() {
        agg.update(timestamp, prob);
    }

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

    let mut aggregations = if let Some(bytes) = db_bytes {
        let archived = unsafe { rkyv::archived_root::<Vec<TimeDecayedSum>>(&bytes) };
        archived.deserialize(&mut Infallible).unwrap()
    } else {
        vec![TimeDecayedSum::new(DEFAULT_HALF_LIFE); n_aggregations]
    };

    let prob = {
        let mut counts = server_state.in_memory_counts.lock().unwrap();
        let count_entry = counts
            .entry(key.to_vec())
            .or_insert_with(|| TimeDecayedSum::new(DEFAULT_HALF_LIFE));

        count_entry.update(timestamp, 1.0);
        (threshold / count_entry.value()).min(1.0)
    };

    for agg in aggregations.iter_mut() {
        agg.update(timestamp, prob);
    }

    if rand::thread_rng().gen::<f64>() < prob {
        let new_bytes = rkyv::to_bytes::<_, 4096>(&aggregations).unwrap();
        db.put(key, new_bytes).expect("RocksDB write error");
    }
}

pub fn batched_endpoint(
    db: &rocksdb::DB,
    server_state: &ServerState,
    key: &[u8],
    timestamp: f64,
    n_aggregations: usize,
    batch_size: usize,
) {
    // 1. Acquire lock, update cache, and extract batch payload if threshold reached
    let (items_to_flush, should_flush) = {
        let mut batched = server_state.batched_state.lock().unwrap();

        let mut profiles = if let Some(cached_profiles) = batched.cache.get(key) {
            cached_profiles.clone()
        } else if let Some(bytes) = db.get(key).expect("RocksDB read error") {
            let archived = unsafe { rkyv::archived_root::<Vec<TimeDecayedSum>>(&bytes) };
            archived.deserialize(&mut Infallible).unwrap()
        } else {
            vec![TimeDecayedSum::new(DEFAULT_HALF_LIFE); n_aggregations]
        };

        for agg in profiles.iter_mut() {
            agg.update(timestamp, 1.0);
        }

        batched.cache.insert(key.to_vec(), profiles);
        batched.current_batch_count += 1;

        if batched.current_batch_count >= batch_size {
            let items = std::mem::take(&mut batched.cache);
            batched.current_batch_count = 0;
            (items, true)
        } else {
            (std::collections::HashMap::new(), false)
        }
    }; // MutexGuard is dropped here

    // 2. Perform serialization and asynchronous DB write OUTSIDE the mutex lock
    if should_flush {
        let mut batch = rocksdb::WriteBatch::default();

        for (merchant_id, merchant_profiles) in &items_to_flush {
            let serialized = rkyv::to_bytes::<_, 4096>(merchant_profiles).unwrap();
            batch.put(merchant_id, serialized);
        }

        // Default write options (page-cache buffered, no blocking hardware fsync)
        db.write(batch).expect("RocksDB write batch error");
    }
}

pub fn fixed_rate_filtering_endpoint(
    db: &rocksdb::DB,
    key: &[u8],
    timestamp: f64,
    n_aggregations: usize,
    prob: f64,
) {
    let db_bytes = db.get(key).expect("RocksDB read error");

    let mut aggregations = if let Some(bytes) = db_bytes {
        let archived = unsafe { rkyv::archived_root::<Vec<TimeDecayedSum>>(&bytes) };
        archived.deserialize(&mut Infallible).unwrap()
    } else {
        vec![TimeDecayedSum::new(DEFAULT_HALF_LIFE); n_aggregations]
    };

    // Update aggregations with event weight
    for agg in aggregations.iter_mut() {
        agg.update(timestamp, prob);
    }

    // Fixed probabilistic write to disk
    if rand::thread_rng().gen::<f64>() < prob {
        let new_bytes = rkyv::to_bytes::<_, 4096>(&aggregations).unwrap();
        db.put(key, new_bytes).expect("RocksDB write error");
    }
}

// -----------------------------------------------------------------------------
// CSV Exporter
// -----------------------------------------------------------------------------
fn save_client_metrics_csv(
    file_path: &Path,
    rep: usize,
    num_threads: usize,
    endpoint_name: &str,
    threshold_str: &str,
    total_ops: usize,
    duration_secs: f64,
    throughput_ops_sec: f64,
    hist: &Histogram<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "rep,threads,endpoint,threshold,total_ops,duration_secs,throughput_ops_sec,min_ns,mean_ns,p50_ns,p90_ns,p95_ns,p99_ns,p99_9_ns,p99_99_ns,max_ns"
    )?;

    writeln!(
        writer,
        "{},{},{},{},{},{:.6},{:.2},{},{:.2},{},{},{},{},{},{},{}",
        rep,
        num_threads,
        endpoint_name,
        threshold_str,
        total_ops,
        duration_secs,
        throughput_ops_sec,
        hist.min(),
        hist.mean(),
        hist.value_at_quantile(0.50),
        hist.value_at_quantile(0.90),
        hist.value_at_quantile(0.95),
        hist.value_at_quantile(0.99),
        hist.value_at_quantile(0.999),
        hist.value_at_quantile(0.9999),
        hist.max()
    )?;

    writer.flush()?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Execution Harness
// -----------------------------------------------------------------------------
pub fn run_in_process_benchmark(
    endpoint_name: &str,
    run_dir: &Path,
    rep: usize,
    config: &StreamConfig,
    threshold: f64,
    threshold_str: &str,
    num_threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let folder_name = match endpoint_name {
        "unfiltered" | "batched" => endpoint_name.to_string(),
        "persistence_path" | "full_stream_filtering" | "fixed_rate_filtering" => {
            format!("{}_{}", endpoint_name, threshold_str)
        }
        _ => endpoint_name.to_string(),
    };

    let endpoint_dir = run_dir.join(&folder_name);
    create_dir_all(&endpoint_dir)?;

    let metrics_path =
        endpoint_dir.join(format!("client_metrics_rep_{}_t{}.csv", rep, num_threads));
    let db_stats_path = endpoint_dir.join(format!("db_stats_rep_{}_t{}.csv", rep, num_threads));

    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join(format!("rocksdb_t{}", num_threads));
    let db = Arc::new(create_hotkey_optimized_db(db_path.to_str().unwrap()));

    let server_state = Arc::new(ServerState {
        in_memory_counts: Mutex::new(HashMap::with_capacity(200_000)),
        batched_state: Mutex::new(BatchedState {
            cache: HashMap::with_capacity(1_000),
            current_batch_count: 0,
        }),
    });

    let bench_seed = config.seed.wrapping_add(rep as u64);
    let warmup_seed = config.seed.wrapping_add(1000 + rep as u64);

    let mut partitioned_bench: Vec<Vec<EventPayload>> = vec![Vec::new(); num_threads];
    let bench_gen = EventStreamGenerator::new_seeded(
        config.num_keys,
        config.skew,
        config.bench_events,
        bench_seed,
    );

    for event in bench_gen {
        let tid = (hash_key(&event.key) as usize) % num_threads;
        partitioned_bench[tid].push(EventPayload {
            key: event.key,
            timestamp: event.timestamp,
        });
    }

    let mut partitioned_warmup: Vec<Vec<EventPayload>> = vec![Vec::new(); num_threads];
    let warmup_gen = EventStreamGenerator::new_seeded(
        config.num_keys,
        config.skew,
        config.warmup_events,
        warmup_seed,
    );

    for event in warmup_gen {
        let tid = (hash_key(&event.key) as usize) % num_threads;
        partitioned_warmup[tid].push(EventPayload {
            key: event.key,
            timestamp: event.timestamp,
        });
    }

    let bench_payloads = Arc::new(partitioned_bench);
    let warmup_payloads = Arc::new(partitioned_warmup);

    let barrier = Arc::new(Barrier::new(num_threads + 1));
    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let db = Arc::clone(&db);
        let server_state = Arc::clone(&server_state);
        let bench_payloads = Arc::clone(&bench_payloads);
        let warmup_payloads = Arc::clone(&warmup_payloads);
        let barrier = Arc::clone(&barrier);
        let endpoint_name = endpoint_name.to_string();

        let handle = thread::spawn(move || -> Result<Histogram<u64>, String> {
            let my_warmup = &warmup_payloads[thread_id];
            for ev in my_warmup {
                match endpoint_name.as_str() {
                    "unfiltered" => unfiltered_endpoint(&db, &ev.key, ev.timestamp, N_AGGREGATIONS),
                    "persistence_path" => persistence_path_endpoint(
                        &db,
                        &ev.key,
                        ev.timestamp,
                        N_AGGREGATIONS,
                        threshold,
                    ),
                    "full_stream_filtering" => full_stream_filtering_endpoint(
                        &db,
                        &server_state,
                        &ev.key,
                        ev.timestamp,
                        N_AGGREGATIONS,
                        threshold,
                    ),
                    "batched" => batched_endpoint(
                        &db,
                        &server_state,
                        &ev.key,
                        ev.timestamp,
                        N_AGGREGATIONS,
                        DEFAULT_BATCH_SIZE,
                    ),
                    "fixed_rate_filtering" => fixed_rate_filtering_endpoint(
                        &db,
                        &ev.key,
                        ev.timestamp,
                        N_AGGREGATIONS,
                        threshold.min(1.0),
                    ),
                    _ => return Err("Unsupported endpoint target".to_string()),
                }
            }

            let my_bench = &bench_payloads[thread_id];
            let mut thread_hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3)
                .map_err(|e| e.to_string())?;

            barrier.wait();

            for ev in my_bench {
                let op_start = Instant::now();

                match endpoint_name.as_str() {
                    "unfiltered" => unfiltered_endpoint(&db, &ev.key, ev.timestamp, N_AGGREGATIONS),
                    "persistence_path" => persistence_path_endpoint(
                        &db,
                        &ev.key,
                        ev.timestamp,
                        N_AGGREGATIONS,
                        threshold,
                    ),
                    "full_stream_filtering" => full_stream_filtering_endpoint(
                        &db,
                        &server_state,
                        &ev.key,
                        ev.timestamp,
                        N_AGGREGATIONS,
                        threshold,
                    ),
                    "batched" => batched_endpoint(
                        &db,
                        &server_state,
                        &ev.key,
                        ev.timestamp,
                        N_AGGREGATIONS,
                        DEFAULT_BATCH_SIZE,
                    ),
                    "fixed_rate_filtering" => fixed_rate_filtering_endpoint(
                        &db,
                        &ev.key,
                        ev.timestamp,
                        N_AGGREGATIONS,
                        threshold.min(1.0),
                    ),
                    _ => unreachable!(),
                }

                let elapsed_ns = op_start.elapsed().as_nanos() as u64;
                thread_hist.record(elapsed_ns).map_err(|e| e.to_string())?;
            }

            Ok(thread_hist)
        });

        handles.push(handle);
    }

    barrier.wait();
    let global_start = Instant::now();

    let mut combined_hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3)?;
    for handle in handles {
        let thread_hist = handle.join().map_err(|_| "Worker thread panicked")??;
        combined_hist.add(&thread_hist)?;
    }

    let duration_secs = global_start.elapsed().as_secs_f64();
    let total_ops = combined_hist.len() as usize;
    let throughput = total_ops as f64 / duration_secs;

    save_client_metrics_csv(
        &metrics_path,
        rep,
        num_threads,
        endpoint_name,
        threshold_str,
        total_ops,
        duration_secs,
        throughput,
        &combined_hist,
    )?;

    println!(
        "[Bench] Finished {} (threads: {}). Total Ops: {}, Throughput: {:.2} ops/sec",
        folder_name, num_threads, total_ops, throughput
    );

    extract_and_save_db_stats(&db, &db_stats_path, rep)?;
    print_raw_db_stats(&db);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let endpoint_target = args.get(1).map(|s| s.as_str()).unwrap_or("unfiltered");
    let rep: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let timestamp_folder = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| Local::now().format("%Y-%m-%d_%H-%M-%S").to_string());
    let threshold_str = args.get(4).cloned().unwrap_or_else(|| "1.0".to_string());
    let threshold: f64 = threshold_str.parse().unwrap_or(1.0);
    let num_threads: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);

    let base_seed = 42;
    // skew 1.05 -> for 5% entity skew
    // skew 1.00 -> for 10% skew
    // skew 0.94 -> for 20% skew
    let stream_config = StreamConfig {
        num_keys: 2_000_000,
        skew: 1.0, //1.05, //1.0107,
        warmup_events: 100_000,
        bench_events: 5_000_000,
        seed: base_seed,
    };

    if rep == 0 && num_threads == 1 {
        let sample_gen = EventStreamGenerator::new_seeded(
            stream_config.num_keys,
            stream_config.skew,
            stream_config.bench_events,
            stream_config.seed,
        );
        let sample_events: Vec<(Vec<u8>, f64)> = sample_gen.map(|e| (e.key, e.timestamp)).collect();
        let stats = StreamStats::compute(&sample_events);
        stats.print();
    }

    let run_dir = PathBuf::from(EXPERIMENT_FOLDER).join(&timestamp_folder);

    run_in_process_benchmark(
        endpoint_target,
        &run_dir,
        rep,
        &stream_config,
        threshold,
        &threshold_str,
        num_threads,
    )?;

    Ok(())
}
