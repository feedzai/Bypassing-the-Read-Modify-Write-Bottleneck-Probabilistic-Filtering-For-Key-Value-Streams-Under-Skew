use chrono::Local;
use crossbeam_channel::bounded;
use hdrhistogram::Histogram;
use rand::Rng;
use rkyv::{Deserialize as RkyvDeserialize, Infallible};
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs::{create_dir_all, File};
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
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
pub const EXPERIMENT_FOLDER: &str = "./open_loop_experiments";

#[derive(Clone)]
pub struct OpenLoopEventPayload {
    pub key: Vec<u8>,
    pub timestamp: f64,
    pub scheduled_at: Instant, // Scheduled wall-clock dispatch time
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

// -----------------------------------------------------------------------------
// CSV Exporter
// -----------------------------------------------------------------------------
fn save_open_loop_metrics_csv(
    file_path: &Path,
    rep: usize,
    num_threads: usize,
    endpoint_name: &str,
    threshold_str: &str,
    target_rate_ops_sec: f64,
    total_ops: usize,
    duration_secs: f64,
    achieved_throughput_ops_sec: f64,
    hist: &Histogram<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "rep,threads,endpoint,threshold,target_rate_ops_sec,total_ops,duration_secs,achieved_throughput_ops_sec,min_ns,mean_ns,p50_ns,p90_ns,p95_ns,p99_ns,p99_9_ns,p99_99_ns,max_ns"
    )?;

    writeln!(
        writer,
        "{},{},{},{},{:.2},{},{:.6},{:.2},{},{:.2},{},{},{},{},{},{},{}",
        rep,
        num_threads,
        endpoint_name,
        threshold_str,
        target_rate_ops_sec,
        total_ops,
        duration_secs,
        achieved_throughput_ops_sec,
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
// Open-Loop Execution Harness
// -----------------------------------------------------------------------------
pub fn run_open_loop_benchmark(
    endpoint_name: &str,
    run_dir: &Path,
    rep: usize,
    config: &StreamConfig,
    threshold: f64,
    threshold_str: &str,
    num_threads: usize,
    target_rate_ops_sec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let folder_name = match endpoint_name {
        "unfiltered" => endpoint_name.to_string(),
        "persistence_path" => format!("{}_{}", endpoint_name, threshold_str),
        _ => endpoint_name.to_string(),
    };

    let rate_folder = format!("rate_{}", target_rate_ops_sec as u64);
    let endpoint_dir = run_dir.join(&folder_name).join(&rate_folder);
    create_dir_all(&endpoint_dir)?;

    let metrics_path = endpoint_dir.join(format!(
        "client_metrics_rep_{}_t{}_rate{}.csv",
        rep, num_threads, target_rate_ops_sec as u64
    ));
    let db_stats_path = endpoint_dir.join(format!(
        "db_stats_rep_{}_t{}_rate{}.csv",
        rep, num_threads, target_rate_ops_sec as u64
    ));

    let temp_dir = TempDir::new()?;
    let db_path = temp_dir
        .path()
        .join(format!("rocksdb_openloop_t{}", num_threads));
    let db = Arc::new(create_hotkey_optimized_db(db_path.to_str().unwrap()));

    let bench_seed = config.seed.wrapping_add(rep as u64);
    let warmup_seed = config.seed.wrapping_add(1000 + rep as u64);

    // Generate stream datasets
    let bench_gen = EventStreamGenerator::new_seeded(
        config.num_keys,
        config.skew,
        config.bench_events,
        bench_seed,
    );
    let bench_events: Vec<(Vec<u8>, f64)> = bench_gen.map(|e| (e.key, e.timestamp)).collect();

    let warmup_gen = EventStreamGenerator::new_seeded(
        config.num_keys,
        config.skew,
        config.warmup_events,
        warmup_seed,
    );
    let warmup_events: Vec<(Vec<u8>, f64)> = warmup_gen.map(|e| (e.key, e.timestamp)).collect();

    // 1. Execute Warmup (Closed-Loop to prepopulate DB state)
    for (key, ts) in &warmup_events {
        match endpoint_name {
            "unfiltered" => unfiltered_endpoint(&db, key, *ts, N_AGGREGATIONS),
            "persistence_path" => {
                persistence_path_endpoint(&db, key, *ts, N_AGGREGATIONS, threshold)
            }
            _ => panic!("Unsupported endpoint"),
        }
    }

    // 2. Setup Worker Channels & Threads
    let mut senders = Vec::with_capacity(num_threads);
    let mut receivers = Vec::with_capacity(num_threads);

    for _ in 0..num_threads {
        let (s, r) = bounded::<OpenLoopEventPayload>(1_000_000);
        senders.push(s);
        receivers.push(r);
    }

    let barrier = Arc::new(Barrier::new(num_threads + 1));
    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let db = Arc::clone(&db);
        let rx = receivers[thread_id].clone();
        let barrier = Arc::clone(&barrier);
        let endpoint_name = endpoint_name.to_string();

        let handle = thread::spawn(move || -> Result<Histogram<u64>, String> {
            // Track up to 1 hour (3,600,000,000,000 ns) of queue delay
            let mut thread_hist = Histogram::<u64>::new_with_bounds(1, 3_600_000_000_000, 3)
                .map_err(|e| e.to_string())?;

            barrier.wait(); // Synchronize start with producer

            while let Ok(ev) = rx.recv() {
                match endpoint_name.as_str() {
                    "unfiltered" => unfiltered_endpoint(&db, &ev.key, ev.timestamp, N_AGGREGATIONS),
                    "persistence_path" => persistence_path_endpoint(
                        &db,
                        &ev.key,
                        ev.timestamp,
                        N_AGGREGATIONS,
                        threshold,
                    ),
                    _ => unreachable!(),
                }

                // Response Latency = Time Completed - Time Scheduled (includes queueing delay)
                let response_ns = ev.scheduled_at.elapsed().as_nanos() as u64;

                // Saturating record clamps to max bound on queue overflow, preventing panics
                thread_hist.saturating_record(response_ns);
            }

            Ok(thread_hist)
        });

        handles.push(handle);
    }

    // 3. Paced Open-Loop Producer
    let interval = Duration::from_secs_f64(1.0 / target_rate_ops_sec);

    barrier.wait(); // Sync producer with worker threads
    let global_start = Instant::now();

    for (i, (key, timestamp)) in bench_events.into_iter().enumerate() {
        let scheduled_at = global_start + interval * (i as u32);

        let now = Instant::now();
        if scheduled_at > now {
            spin_sleep::sleep(scheduled_at - now);
        }

        let tid = (hash_key(&key) as usize) % num_threads;
        let payload = OpenLoopEventPayload {
            key,
            timestamp,
            scheduled_at,
        };

        if senders[tid].send(payload).is_err() {
            break;
        }
    }

    // Drop senders to signal workers to drain queues and shut down
    drop(senders);

    let mut combined_hist = Histogram::<u64>::new_with_bounds(1, 3_600_000_000_000, 3)?;

    for handle in handles {
        let thread_hist = handle.join().map_err(|_| "Worker thread panicked")??;
        combined_hist.add(&thread_hist)?;
    }

    let duration_secs = global_start.elapsed().as_secs_f64();
    let total_ops = combined_hist.len() as usize;
    let achieved_throughput = total_ops as f64 / duration_secs;

    save_open_loop_metrics_csv(
        &metrics_path,
        rep,
        num_threads,
        endpoint_name,
        threshold_str,
        target_rate_ops_sec,
        total_ops,
        duration_secs,
        achieved_throughput,
        &combined_hist,
    )?;

    println!(
        "[Open-Loop] Finished {} | Rate: {:.0} ops/sec | Threads: {} | Target Ops: {} | P99 Latency: {:.2} ms",
        folder_name,
        target_rate_ops_sec,
        num_threads,
        total_ops,
        combined_hist.value_at_quantile(0.99) as f64 / 1_000_000.0
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
    let target_rate_ops_sec: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(20000.0);

    let base_seed = 42;
    let stream_config = StreamConfig {
        num_keys: 2_000_000,
        // num_keys: 200_000,
        skew: 1.05,
        warmup_events: 100_000,
        bench_events: 5_000_000,
        // bench_events: 500_000,
        seed: base_seed,
    };

    let sample_gen = EventStreamGenerator::new_seeded(
        stream_config.num_keys,
        stream_config.skew,
        stream_config.bench_events,
        stream_config.seed,
    );
    let sample_events: Vec<(Vec<u8>, f64)> = sample_gen.map(|e| (e.key, e.timestamp)).collect();
    let stats = StreamStats::compute(&sample_events);
    stats.print();

    let run_dir = PathBuf::from(EXPERIMENT_FOLDER).join(&timestamp_folder);

    run_open_loop_benchmark(
        endpoint_target,
        &run_dir,
        rep,
        &stream_config,
        threshold,
        &threshold_str,
        num_threads,
        target_rate_ops_sec,
    )?;

    Ok(())
}
