use rocksdb::{
    BlockBasedIndexType, BlockBasedOptions, Cache, DBCompactionStyle, FlushOptions, Options, DB,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbMetrics {
    pub cumulative_writes: String,
    pub waf: f64,
    pub flush_gb: f64,
    pub compaction_read_gb: f64,
    pub compaction_write_gb: f64,
    pub uptime_secs: f64,
}

/// Configures and opens a hot-key optimized RocksDB instance.
pub fn create_hotkey_optimized_db(db_path: &str) -> DB {
    let _ = std::fs::remove_dir_all(db_path); // Clear old DB files

    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.enable_statistics(); // Enable internal RocksDB metrics collection

    // --- HOT-KEY ROCKSDB OPTIMIZATIONS ---
    // 1. Inplace Updates: Prevents memtable bloat when mutating existing key slots
    opts.set_inplace_update_support(true);

    // 2. Concurrency rule: Must be false when inplace updates are active
    opts.set_allow_concurrent_memtable_write(false);

    // 3. Memtable Sizing
    opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB Memtable
    opts.set_max_write_buffer_number(4);

    // 4. Universal Compaction: Handles key overwrites with lower Write Amplification
    opts.set_compaction_style(DBCompactionStyle::Universal);

    // 5. Block Cache & L0/Index Pinning
    let cache = Cache::new_lru_cache(512 * 1024 * 1024); // 512MB Cache
    let mut table_opts = BlockBasedOptions::default();
    table_opts.set_block_cache(&cache);
    table_opts.set_cache_index_and_filter_blocks(true);
    table_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

    // --- READ PATH OPTIMIZATIONS ---

    // 1. Bloom filter (~1% false positive rate). Must use 10.0 (float)
    table_opts.set_bloom_filter(10.0, false);

    // 2. Partition filters to avoid mutex lock contention
    table_opts.set_partition_filters(true);
    table_opts.set_index_type(BlockBasedIndexType::TwoLevelIndexSearch);
    table_opts.set_pin_top_level_index_and_filter(true);

    opts.set_block_based_table_factory(&table_opts);
    DB::open(&opts, db_path).expect("Failed to open RocksDB")
}

/// Serializes and saves any Serde-compatible struct (e.g., TimeDecayedSum) to RocksDB.
pub fn put_struct<T: Serialize>(db: &DB, key: &[u8], value: &T) -> Result<(), String> {
    let bytes = bincode::serialize(value).map_err(|e| e.to_string())?;
    db.put(key, bytes).map_err(|e| e.to_string())
}

/// Retrieves and deserializes a struct from RocksDB.
pub fn get_struct<T: DeserializeOwned>(db: &DB, key: &[u8]) -> Result<Option<T>, String> {
    match db.get(key).map_err(|e| e.to_string())? {
        Some(bytes) => {
            let item = bincode::deserialize(&bytes).map_err(|e| e.to_string())?;
            Ok(Some(item))
        }
        None => Ok(None),
    }
}

pub fn print_raw_db_stats(db: &DB) {
    let mut flush_opts = FlushOptions::default();
    flush_opts.set_wait(true);
    let _ = db.flush_opt(&flush_opts);

    if let Ok(Some(stats)) = db.property_value("rocksdb.stats") {
        println!("{}", stats);
    }
}

#[derive(Debug, Default)]
pub struct ParsedDbStats {
    pub total_write_ops: u64,
    pub wal_bytes: u64,
    pub flush_bytes: u64,
    pub compaction_bytes: u64,
}

fn parse_size_with_unit(val: f64, unit: &str) -> u64 {
    let u = unit.trim().trim_matches(',').to_uppercase();
    if u.starts_with("GB") || u == "G" {
        (val * 1_073_741_824.0) as u64
    } else if u.starts_with("MB") || u == "M" {
        (val * 1_048_576.0) as u64
    } else if u.starts_with("KB") || u == "K" {
        (val * 1_024.0) as u64
    } else {
        val as u64
    }
}

pub fn parse_rocksdb_stats_string(stats: &str) -> ParsedDbStats {
    let mut parsed = ParsedDbStats::default();

    for line in stats.lines() {
        let line_trim = line.trim();

        // 1. Parse Total Write Operations and WAL Ingestion Bytes
        // e.g.: "Cumulative writes: 5100K writes, 5100K keys..."
        if line_trim.starts_with("Cumulative writes:") {
            let parts: Vec<&str> = line_trim.split_whitespace().collect();
            if parts.len() >= 3 {
                let count_str = parts[2].trim_end_matches("K");
                if let Ok(k_count) = count_str.parse::<f64>() {
                    if parts[2].ends_with('K') {
                        parsed.total_write_ops = (k_count * 1_000.0) as u64;
                    } else if parts[2].ends_with('M') {
                        parsed.total_write_ops = (k_count * 1_000_000.0) as u64;
                    } else {
                        parsed.total_write_ops = k_count as u64;
                    }
                }
            }
        }

        // e.g.: "Cumulative WAL: 5100K writes, 0 syncs, ..., written: 15.38 GB, 34.15 MB/s"
        if line_trim.contains("Cumulative WAL:") || line_trim.contains("Cumulative writes:") {
            let key = if line_trim.contains("written:") {
                "written:"
            } else {
                "ingest:"
            };
            if let Some(pos) = line_trim.find(key) {
                let after = &line_trim[pos + key.len()..];
                let parts: Vec<&str> = after.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(val) = parts[0].trim_matches(',').parse::<f64>() {
                        parsed.wal_bytes = parse_size_with_unit(val, parts[1]);
                    }
                }
            }
        }

        // 2. Parse MemTable Flush Bytes
        // e.g.: "Flush(GB): cumulative 0.237, interval 0.237"
        if line_trim.contains("Flush(GB):") {
            if let Some(pos) = line_trim.find("cumulative") {
                let after = &line_trim[pos + "cumulative".len()..];
                let parts: Vec<&str> = after.split_whitespace().collect();
                if !parts.is_empty() {
                    if let Ok(val) = parts[0].trim_matches(',').parse::<f64>() {
                        parsed.flush_bytes = (val * 1_073_741_824.0) as u64;
                    }
                }
            }
        }

        // 3. Parse Compaction Write Bytes
        // e.g.: "Cumulative compaction: 0.47 GB write, 1.04 MB/s write..."
        if line_trim.starts_with("Cumulative compaction:") {
            let after = &line_trim["Cumulative compaction:".len()..];
            let parts: Vec<&str> = after.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(val) = parts[0].trim_matches(',').parse::<f64>() {
                    parsed.compaction_bytes = parse_size_with_unit(val, parts[1]);
                }
            }
        }
    }

    parsed
}

pub fn extract_and_save_db_stats(
    db: &DB,
    stats_path: &Path,
    rep: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Force MemTable flush to disk
    let mut flush_opts = FlushOptions::default();
    flush_opts.set_wait(true);
    let _ = db.flush_opt(&flush_opts);

    // 2. Query total SST size
    let sst_size = db
        .property_int_value("rocksdb.total-sst-files-size")?
        .unwrap_or(0);

    // 3. Extract string stats block
    let stats_str = db.property_value("rocksdb.stats")?.unwrap_or_default();
    let parsed = parse_rocksdb_stats_string(&stats_str);

    // 4. Save raw text dump to file so no terminal output is lost
    let raw_log_path = stats_path.with_file_name(format!("rocksdb_raw_stats_rep_{}.txt", rep));
    let mut raw_file = File::create(&raw_log_path)?;
    raw_file.write_all(stats_str.as_bytes())?;

    // 5. Compute derived metrics
    let wal_gb = parsed.wal_bytes as f64 / 1_073_741_824.0;
    let flush_gb = parsed.flush_bytes as f64 / 1_073_741_824.0;
    let compaction_gb = parsed.compaction_bytes as f64 / 1_073_741_824.0;
    let sst_gb = sst_size as f64 / 1_073_741_824.0;
    let total_disk_gb = wal_gb + flush_gb + compaction_gb;

    let compaction_waf = if parsed.flush_bytes > 0 {
        parsed.compaction_bytes as f64 / parsed.flush_bytes as f64
    } else {
        1.0
    };

    let system_waf = if parsed.wal_bytes > 0 {
        total_disk_gb / wal_gb
    } else {
        1.0
    };

    println!(
        "[Server] DB Metrics -> Mutations: {} | WAL: {:.3} GB, Flush: {:.3} GB, Comp: {:.3} GB, Total Disk: {:.3} GB | Comp WAF: {:.2}, Sys WAF: {:.2}",
        parsed.total_write_ops, wal_gb, flush_gb, compaction_gb, total_disk_gb, compaction_waf, system_waf
    );

    // 6. Save structured CSV
    let file = File::create(stats_path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "rep,total_write_ops,wal_bytes,bytes_flushed,compaction_bytes,total_disk_bytes,sst_size_bytes,wal_gb,flush_gb,compaction_gb,total_disk_gb,sst_gb,compaction_waf,system_waf"
    )?;
    writeln!(
        writer,
        "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2},{:.2}",
        rep,
        parsed.total_write_ops,
        parsed.wal_bytes,
        parsed.flush_bytes,
        parsed.compaction_bytes,
        (parsed.wal_bytes + parsed.flush_bytes + parsed.compaction_bytes),
        sst_size,
        wal_gb,
        flush_gb,
        compaction_gb,
        total_disk_gb,
        sst_gb,
        compaction_waf,
        system_waf
    )?;

    writer.flush()?;
    Ok(())
}
