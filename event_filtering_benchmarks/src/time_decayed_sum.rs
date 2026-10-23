use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct TimeDecayedSum {
    half_life: f64,
    decay_rate: f64,
    state: f64,
    last_timestamp: f64,
}

impl TimeDecayedSum {
    /// Initializes the struct with a half-life.
    /// `last_timestamp` is set to 0.0 and decay rate is pre-computed.
    pub fn new(half_life: f64) -> Self {
        let decay_rate = -std::f64::consts::LN_2 / half_life;
        Self {
            half_life,
            decay_rate,
            state: 0.0,
            last_timestamp: 0.0,
        }
    }

    /// Updates state given a current timestamp and probability p:
    /// - If last_timestamp is 0.0, initializes the timestamp and sets state to 1/p without decay.
    /// - Otherwise: new_state = old_state * exp(decay_rate * delta_t) + 1/p
    /// Returns 0.0 if the exponent calculation overflows or becomes invalid.
    pub fn update(&mut self, current_timestamp: f64, probability: f64) -> f64 {
        let inv_prob = 1.0 / probability;

        // If this is the initial update, set the starting timestamp and state without decay
        if self.last_timestamp == 0.0 {
            self.last_timestamp = current_timestamp;
            self.state = inv_prob;
            return self.state;
        }

        let delta_t = current_timestamp - self.last_timestamp;
        self.last_timestamp = current_timestamp;

        let exponent = self.decay_rate * delta_t;
        let exp_factor = exponent.exp();

        // Check for exponent overflow / non-finite float results
        if exponent.is_infinite() || exp_factor.is_infinite() || exp_factor.is_nan() {
            self.state = 0.0;
            return 0.0;
        }

        self.state = self.state * exp_factor + inv_prob;
        self.state
    }

    /// Computes `(state * decay) + delta` at `timestamp` without mutating state.
    pub fn projected_value(&self, timestamp: f64, delta: f64) -> f64 {
        if self.last_timestamp <= 0.0 {
            return delta;
        }
        let dt = timestamp - self.last_timestamp;
        if dt <= 0.0 {
            return self.state + delta;
        }
        let decayed = self.state * (self.decay_rate * dt).exp();
        if decayed.is_nan() || decayed.is_infinite() {
            delta
        } else {
            decayed + delta
        }
    }

    /// Gets the current decayed state without updating the timestamp.
    pub fn value(&self) -> f64 {
        self.state
    }
}
