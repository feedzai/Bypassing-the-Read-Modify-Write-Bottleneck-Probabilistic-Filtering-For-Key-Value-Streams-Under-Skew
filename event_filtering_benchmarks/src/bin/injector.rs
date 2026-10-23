use chrono::Local;
use hdrhistogram::Histogram;
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs::{create_dir_all, File};
use std::hash::{Hash, Hasher};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use event_filtering_benchmarks::event_stream::StreamStats;
use event_filtering_benchmarks::{Command, EventStreamGenerator, Response, StreamConfig};

pub const EXPERIMENT_FOLDER: &str = "./experiment_results";
pub const SOCKET_PATH: &str = "/tmp/rocksdb_bench.sock";

fn build_command(endpoint_name: &str, key: Vec<u8>, timestamp: f64, threshold: f64) -> Command {
    match endpoint_name {
        "unfiltered" => Command::Unfiltered { key, timestamp },
        "persistence_path" => Command::PersistencePath {
            key,
            timestamp,
            threshold,
        },
        "full_stream_filtering" => Command::FullStreamFiltering {
            key,
            timestamp,
            threshold,
        },
        _ => panic!("Unsupported endpoint target: {}", endpoint_name),
    }
}

fn hash_key(key: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

fn save_client_metrics_csv(
    file_path: &Path,
    rep: usize,
    total_ops: usize,
    duration_secs: f64,
    throughput_ops_sec: f64,
    hist: &Histogram<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "rep,total_ops,duration_secs,throughput_ops_sec,min_ns,mean_ns,p50_ns,p90_ns,p95_ns,p99_ns,p99_9_ns,max_ns"
    )?;

    writeln!(
        writer,
        "{},{},{:.6},{:.2},{},{:.2},{},{},{},{},{},{}",
        rep,
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
        hist.max()
    )?;

    writer.flush()?;
    Ok(())
}

pub fn run_parallel_endpoint_benchmark(
    endpoint_name: &str,
    run_dir: &Path,
    rep: usize,
    timestamp_folder: &str,
    config: &StreamConfig,
    threshold: f64,
    threshold_str: &str,
    num_threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let folder_name = match endpoint_name {
        "unfiltered" => endpoint_name.to_string(),
        "persistence_path" | "full_stream_filtering" => {
            format!("{}_{}", endpoint_name, threshold_str)
        }
        _ => endpoint_name.to_string(),
    };

    let endpoint_dir = run_dir.join(&folder_name);
    create_dir_all(&endpoint_dir)?;

    let metrics_path = endpoint_dir.join(format!("client_metrics_rep_{}.csv", rep));
    let barrier = Arc::new(Barrier::new(num_threads));

    println!(
        "[Injector] [{}] Spawning {} worker threads & pre-generating payload in RAM...",
        folder_name, num_threads
    );

    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let barrier = Arc::clone(&barrier);
        let endpoint_name = endpoint_name.to_string();
        let config = config.clone();

        let handle = thread::spawn(move || -> Result<Histogram<u64>, String> {
            let mut stream = UnixStream::connect(SOCKET_PATH).map_err(|e| e.to_string())?;

            let warmup_seed = config.seed.wrapping_add(1000 + thread_id as u64);
            let bench_seed = config.seed.wrapping_add(thread_id as u64);

            // -----------------------------------------------------------------
            // 1. PRE-COMPUTE & PRE-SERIALIZE STREAM IN MEMORY (UNTIMED)
            // -----------------------------------------------------------------
            let bench_capacity = config.bench_events / num_threads + 100;
            let mut precomputed_bench: Vec<Vec<u8>> = Vec::with_capacity(bench_capacity);

            let bench_stream = EventStreamGenerator::new_seeded(
                config.num_keys,
                config.skew,
                config.bench_events,
                bench_seed,
            );

            for event in bench_stream {
                if hash_key(&event.key) as usize % num_threads == thread_id {
                    let cmd = build_command(&endpoint_name, event.key, event.timestamp, threshold);
                    let serialized_cmd = bincode::serialize(&cmd).map_err(|e| e.to_string())?;
                    precomputed_bench.push(serialized_cmd);
                }
            }

            // Pre-compute Warmup Stream as well
            let warmup_capacity = (config.warmup_events / num_threads) + 100;
            let mut precomputed_warmup: Vec<Vec<u8>> = Vec::with_capacity(warmup_capacity);

            let warmup_stream = EventStreamGenerator::new_seeded(
                config.num_keys,
                config.skew,
                config.warmup_events / num_threads,
                warmup_seed,
            );

            for event in warmup_stream {
                if hash_key(&event.key) as usize % num_threads == thread_id {
                    let cmd = build_command(&endpoint_name, event.key, event.timestamp, threshold);
                    let serialized_cmd = bincode::serialize(&cmd).map_err(|e| e.to_string())?;
                    precomputed_warmup.push(serialized_cmd);
                }
            }

            // -----------------------------------------------------------------
            // 2. WARMUP EXECUTION
            // -----------------------------------------------------------------
            if thread_id == 0 {
                println!(
                    "[Injector] Executing Warmup stream ({} events total)...",
                    config.warmup_events
                );
            }

            for req_bytes in &precomputed_warmup {
                stream.write_all(req_bytes).map_err(|e| e.to_string())?;
                let _resp: Response =
                    bincode::deserialize_from(&mut stream).map_err(|e| e.to_string())?;
            }

            // -----------------------------------------------------------------
            // 3. BARRIER SYNC
            // -----------------------------------------------------------------
            barrier.wait();

            if thread_id == 0 {
                println!("============================================================");
                println!(
                    "[Injector] Warmup complete! Starting TIMED INJECTION ({} events in RAM)...",
                    config.bench_events
                );
                println!("============================================================");
            }

            // -----------------------------------------------------------------
            // 4. TIMED INJECTION LOOP (Zero generation overhead, socket I/O only)
            // -----------------------------------------------------------------
            let mut thread_hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3)
                .map_err(|e| e.to_string())?;

            for req_bytes in &precomputed_bench {
                let op_start = Instant::now();

                stream
                    .write_all(black_box(req_bytes))
                    .map_err(|e| e.to_string())?;
                let _resp: Response =
                    bincode::deserialize_from(&mut stream).map_err(|e| e.to_string())?;

                let elapsed_ns = op_start.elapsed().as_nanos() as u64;
                thread_hist.record(elapsed_ns).map_err(|e| e.to_string())?;
            }

            Ok(thread_hist)
        });

        handles.push(handle);
    }

    let global_start = Instant::now();

    // Collect and merge histograms from all worker threads
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
        total_ops,
        duration_secs,
        throughput,
        &combined_hist,
    )?;

    println!(
        "[Injector] Finished {}. Total Ops: {}, Throughput: {:.2} ops/sec",
        folder_name, total_ops, throughput
    );
    println!(
        "  p50 Latency: {:.2} µs",
        combined_hist.value_at_quantile(0.50) as f64 / 1_000.0
    );
    println!(
        "  p99 Latency: {:.2} µs",
        combined_hist.value_at_quantile(0.99) as f64 / 1_000.0
    );

    // -----------------------------------------------------------------
    // 5. CONTROL STREAM SHUTDOWN
    // -----------------------------------------------------------------
    println!("[Injector] Requesting Server shutdown & DB stats flush...");
    let mut control_stream = UnixStream::connect(SOCKET_PATH)?;
    let shutdown_cmd = Command::Shutdown {
        rep,
        timestamp_folder: timestamp_folder.to_string(),
        endpoint_name: folder_name,
    };
    bincode::serialize_into(&mut control_stream, &shutdown_cmd)?;

    let _resp: Response = bincode::deserialize_from(&mut control_stream)?;
    println!("[Injector] Server confirmed flush complete.");

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
    let num_threads: usize = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(8);

    let base_seed = 42;
    let stream_config = StreamConfig {
        num_keys: 2_000_000,      // 2 Million Unique Keys (~6.4 GB state footprint)
        skew: 1.0107,             // 80/4 Pareto Rule for 2M keys
        warmup_events: 100_000,   // Warmup events to saturate hot keys
        bench_events: 25_000_000, // 25M timed events
        seed: base_seed + rep as u64,
    };

    // 1. Compute and display stream profile info
    let generator = EventStreamGenerator::new_seeded(
        stream_config.num_keys,
        stream_config.skew,
        stream_config.bench_events,
        stream_config.seed,
    );
    let sample_events: Vec<(Vec<u8>, f64)> = generator.map(|e| (e.key, e.timestamp)).collect();
    let stats = StreamStats::compute(&sample_events);
    stats.print();

    let run_dir = PathBuf::from(EXPERIMENT_FOLDER).join(&timestamp_folder);

    run_parallel_endpoint_benchmark(
        endpoint_target,
        &run_dir,
        rep,
        &timestamp_folder,
        &stream_config,
        threshold,
        &threshold_str,
        num_threads,
    )?;

    Ok(())
}
