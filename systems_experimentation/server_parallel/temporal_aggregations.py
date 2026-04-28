import struct
import math
from numba import njit

# Limits for math.exp() to prevent crashes
EXP_MAX_ARG = 709.0
EXP_MIN_ARG = -709.0

_SUM_STRUCT = struct.Struct('dddq')


@njit(fastmath=True, nogil=True)
def jit_sigmoid(x):
    if x < -709.0:
        return 0.0
    if x > 709.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


@njit(fastmath=True, nogil=True)
def jit_logit(p):
    eps = 1e-15
    p_clamped = max(eps, min(1.0 - eps, p))
    return math.log(p_clamped / (1.0 - p_clamped))


@njit(fastmath=True, nogil=True)
def safe_exp(x):
    if x < EXP_MIN_ARG:
        return 0.0
    if x > EXP_MAX_ARG:
        return float('inf')
    return math.exp(x)


@njit(fastmath=True, nogil=True)
def compute_update(value, last_t, n_updates, lam, t, q, p):
    if last_t < 0:
        last_t = t

    dt = t - last_t

    if dt >= 0:
        value = value * safe_exp(-lam * dt)
        value += (q / p)
        last_t = t
    else:
        lag = -dt
        value += (q / p) * safe_exp(-lam * lag)

    n_updates += 1
    return value, last_t, n_updates


@njit(fastmath=True, nogil=True)
def compute_get(value, last_t, t, lam):
    dt = t - last_t
    if last_t < 0 or dt < 0:
        return value
    return value * safe_exp(-lam * dt)


@njit(fastmath=True, nogil=True)
def compute_variance_update(count, s1, s2, last_t, n_updates, lam, t, q, p):
    if last_t < 0:
        last_t = t

    dt = t - last_t

    if dt >= 0:
        decay = safe_exp(-lam * dt)
        count = count * decay + (1.0 / p)
        s1 = s1 * decay + (q / p)
        s2 = s2 * decay + (q**2 / p)
        last_t = t
    else:
        lag = -dt
        decay_lag = safe_exp(-lam * lag)
        count += (1.0 / p) * decay_lag
        s1 += (q / p) * decay_lag
        s2 += (q**2 / p) * decay_lag

    n_updates += 1
    return count, s1, s2, last_t, n_updates


@njit(fastmath=True, nogil=True)
def compute_variance_get(count, s1, s2, last_t, t, lam):
    dt = t - last_t
    if last_t < 0 or dt < 0:
        decay = 1.0
    else:
        decay = safe_exp(-lam * dt)

    c_t = count * decay
    s1_t = s1 * decay
    s2_t = s2 * decay

    if c_t <= 0:
        return 0.0

    mean = s1_t / c_t
    mean_2 = s2_t / c_t
    return max(0.0, mean_2 - mean**2)


@njit(fastmath=True, nogil=True)
def compute_mean_update(count, s1, last_t, n_updates, lam, t, q, p):
    if last_t < 0: last_t = t
    dt = t - last_t
    if dt >= 0:
        decay = safe_exp(-lam * dt)
        count = count * decay + (1.0 / p)
        s1 = s1 * decay + (q / p)
        last_t = t
    else:
        lag = -dt
        decay_lag = safe_exp(-lam * lag)
        count += (1.0 / p) * decay_lag
        s1 += (q / p) * decay_lag
    n_updates += 1
    return count, s1, last_t, n_updates


@njit(fastmath=True, nogil=True)
def compute_mean_get(count, s1, last_t, t, lam):
    dt = t - last_t
    decay = safe_exp(-lam * dt) if (last_t >= 0 and dt >= 0) else 1.0
    c_t = count * decay
    s1_t = s1 * decay
    return s1_t / c_t if c_t > 0 else 0.0


class DecayedSum:
    __slots__ = ['lam', 'value', 'last_t', 'n_updates']

    def __init__(self, halflife):
        self.lam = math.log(2) / halflife
        self.value = 0.0
        self.last_t = -1
        self.n_updates = 0

    def update(self, t, q, p):
        self.value, self.last_t, self.n_updates = compute_update(
            self.value, self.last_t, self.n_updates, self.lam, t, q, p)
        return self.value, self.last_t, self.n_updates

    def get(self, t):
        return compute_get(self.value, self.last_t, t, self.lam)

    def to_bytes(self):
        return _SUM_STRUCT.pack(self.lam, self.value, self.last_t,
                                self.n_updates)

    @classmethod
    def from_bytes(cls, data):
        obj = cls.__new__(cls)  # Create instance without calling __init__
        obj.lam, obj.value, obj.last_t, obj.n_updates = _SUM_STRUCT.unpack(
            data)
        return obj


class DecayedMean:
    __slots__ = ['lam', 'count', 's1', 'last_t', 'n_updates']

    def __init__(self, halflife):
        self.lam = math.log(2) / halflife
        self.count = 0.0
        self.s1 = 0.0
        self.last_t = -1
        self.n_updates = 0

    def update(self, t, q, p):
        self.count, self.s1, self.last_t, self.n_updates = compute_mean_update(
            self.count, self.s1, self.last_t, self.n_updates, self.lam, t, q,
            p)

    def get(self, t):
        return compute_mean_get(self.count, self.s1, self.last_t, t, self.lam)


class DecayedVariance:
    __slots__ = ['lam', 'count', 's1', 's2', 'last_t', 'n_updates']

    def __init__(self, halflife):
        self.lam = math.log(2) / halflife
        self.count = 0.0  # Decayed weight count
        self.s1 = 0.0  # Decayed sum of x
        self.s2 = 0.0  # Decayed sum of x^2
        self.last_t = -1
        self.n_updates = 0

    def update(self, t, q, p):
        self.count, self.s1, self.s2, self.last_t, self.n_updates = compute_variance_update(
            self.count, self.s1, self.s2, self.last_t, self.n_updates,
            self.lam, t, q, p)

    def get(self, t):
        return compute_variance_get(self.count, self.s1, self.s2, self.last_t,
                                    t, self.lam)
