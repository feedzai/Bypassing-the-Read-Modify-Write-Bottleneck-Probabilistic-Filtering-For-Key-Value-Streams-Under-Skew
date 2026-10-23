use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Exp};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    pub num_keys: usize,
    pub skew: f64,
    pub warmup_events: usize,
    pub bench_events: usize,
    pub seed: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            num_keys: 200_000,
            skew: 0.82, // ~80/20 Pareto skew
            warmup_events: 10_000,
            bench_events: 100_000,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Event {
    pub key: Vec<u8>,
    pub timestamp: f64,
}

#[derive(Debug)]
struct ScheduledEntity {
    next_timestamp: f64,
    entity_id: usize,
}

impl PartialEq for ScheduledEntity {
    fn eq(&self, other: &Self) -> bool {
        self.next_timestamp == other.next_timestamp
    }
}

impl Eq for ScheduledEntity {}

impl Ord for ScheduledEntity {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .next_timestamp
            .partial_cmp(&self.next_timestamp)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ScheduledEntity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct EventStreamGenerator {
    total_events: usize,
    generated_count: usize,
    entity_keys: Vec<Vec<u8>>,
    exp_distrs: Vec<Exp<f64>>,
    heap: BinaryHeap<ScheduledEntity>,
    rng: StdRng,
}

impl EventStreamGenerator {
    /// Builds a deterministic event stream guaranteed to produce the identical sequence for a given seed.
    pub fn new_seeded(num_keys: usize, skew: f64, total_events: usize, seed: u64) -> Self {
        Self::with_rate_and_seed(num_keys, skew, total_events, 10_000.0, seed)
    }

    pub fn with_rate_and_seed(
        num_keys: usize,
        skew: f64,
        total_events: usize,
        target_system_rate: f64,
        seed: u64,
    ) -> Self {
        // Seeded PRNG guarantees exact sequence reproducibility
        let mut rng = StdRng::seed_from_u64(seed);

        let norm: f64 = (1..=num_keys).map(|i| 1.0 / (i as f64).powf(skew)).sum();

        let mut entity_keys = Vec::with_capacity(num_keys);
        let mut exp_distrs = Vec::with_capacity(num_keys);
        let mut heap = BinaryHeap::with_capacity(num_keys);

        for i in 1..=num_keys {
            let key = format!("entity_{:08}", i).into_bytes();
            let prob = (1.0 / (i as f64).powf(skew)) / norm;
            let lambda_i = target_system_rate * prob;

            let exp = Exp::new(lambda_i).expect("Lambda must be positive");
            let initial_dt: f64 = exp.sample(&mut rng);

            heap.push(ScheduledEntity {
                next_timestamp: initial_dt,
                entity_id: i - 1,
            });

            entity_keys.push(key);
            exp_distrs.push(exp);
        }

        Self {
            total_events,
            generated_count: 0,
            entity_keys,
            exp_distrs,
            heap,
            rng,
        }
    }
}

impl Iterator for EventStreamGenerator {
    type Item = Event;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.generated_count >= self.total_events {
            return None;
        }

        let current = self.heap.pop()?;

        let event = Event {
            key: self.entity_keys[current.entity_id].clone(),
            timestamp: current.next_timestamp,
        };

        let dt: f64 = self.exp_distrs[current.entity_id].sample(&mut self.rng);
        self.heap.push(ScheduledEntity {
            next_timestamp: current.next_timestamp + dt,
            entity_id: current.entity_id,
        });

        self.generated_count += 1;
        Some(event)
    }
}

#[derive(Debug, Clone)]
pub struct StreamStats {
    pub total_events: usize,
    pub unique_entities: usize,
    pub min_time: f64,
    pub max_time: f64,
    pub pct_entities_for_80_pct_events: f64,
}

impl StreamStats {
    /// Computes stream metrics from an iterator or slice of (entity_id, timestamp) events.
    pub fn compute<'a, I>(events: I) -> Self
    where
        I: IntoIterator<Item = &'a (Vec<u8>, f64)>,
    {
        let mut total_events = 0usize;
        let mut min_time = f64::INFINITY;
        let mut max_time = f64::NEG_INFINITY;
        let mut entity_counts: HashMap<&[u8], usize> = HashMap::new();

        for (entity_id, timestamp) in events {
            total_events += 1;
            if *timestamp < min_time {
                min_time = *timestamp;
            }
            if *timestamp > max_time {
                max_time = *timestamp;
            }
            *entity_counts.entry(entity_id.as_slice()).or_insert(0) += 1;
        }

        if total_events == 0 {
            return Self {
                total_events: 0,
                unique_entities: 0,
                min_time: 0.0,
                max_time: 0.0,
                pct_entities_for_80_pct_events: 0.0,
            };
        }

        let unique_entities = entity_counts.len();

        let mut counts: Vec<usize> = entity_counts.into_values().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));

        let target_80_pct = (total_events as f64 * 0.80) as usize;
        let mut accumulated_events = 0usize;
        let mut entities_needed = 0usize;

        for count in &counts {
            accumulated_events += count;
            entities_needed += 1;
            if accumulated_events >= target_80_pct {
                break;
            }
        }

        let pct_entities_for_80_pct_events = if unique_entities > 0 {
            (entities_needed as f64 / unique_entities as f64) * 100.0
        } else {
            0.0
        };

        Self {
            total_events,
            unique_entities,
            min_time,
            max_time,
            pct_entities_for_80_pct_events,
        }
    }

    pub fn print(&self) {
        println!("==================================================");
        println!("              Input Stream Statistics             ");
        println!("==================================================");
        println!(" Total Events         : {}", self.total_events);
        println!(" Unique Entities      : {}", self.unique_entities);
        println!(" Min Timestamp        : {:.2}", self.min_time);
        println!(" Max Timestamp        : {:.2}", self.max_time);
        println!(
            " Total Stream Span    : {:.2} time units",
            self.max_time - self.min_time
        );
        println!(
            " Top Entities for 80% Tx : {:.2}% ({} / {} entities)",
            self.pct_entities_for_80_pct_events,
            ((self.pct_entities_for_80_pct_events / 100.0) * self.unique_entities as f64).round()
                as usize,
            self.unique_entities
        );
        println!("==================================================\n");
    }
}
