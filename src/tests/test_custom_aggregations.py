import pytest
import numpy as np

from smoothoperators import event_filtering


@pytest.mark.parametrize(
    "halflife, updates, expected_val",
    [
        # Case 1: Simple update
        # t=0, q=10. Value becomes 10.
        (10, [(0, 10)], 10.0),

        # Case 2: Exact Half-life decay
        # t=0, q=10 -> val=10
        # t=10 (1 halflife), q=5. Old val decays to 5. New val = 5 + 5 = 10.
        (10, [(0, 10), (10, 5)], 10.0),

        # Case 3: Two Half-lives decay
        # t=0, q=16 -> val=16
        # t=2 (2 halflives), q=0. Old val (16) * 0.25 = 4. New val = 4 + 0 = 4.
        (1, [(0, 16), (2, 0)], 4.0),

        # Case 4: p (weight) scaling
        # t=0, q=10, p=1/2 -> adds 10.
        (10, [(0, 10, .5)], 20.0),
    ])
def test_decayed_sum_update(halflife, updates, expected_val):
    ds = event_filtering.DecayedSum(halflife)
    for update in updates:
        # Unpack update arguments (handle optional p)
        t, q = update[0], update[1]
        p = update[2] if len(update) > 2 else 1.0
        ds.update(t, q, p)

    # We check internal value directly to verify state after update
    assert np.isclose(ds.value, expected_val)


@pytest.mark.parametrize(
    "halflife, updates, peek_t, peek_q, expected_mean",
    [
        # Case 1: Constant value
        # t=0, q=10. Mean should be 10.
        (10, [], 0, 10, 10.0),

        # Case 2: Constant values over time
        # t=0, q=10.
        # Peek t=10, q=10.
        # Sum: 10->5 + 10 = 15. Count: 1->0.5 + 1 = 1.5. Mean = 10.
        (10, [(0, 10)], 10, 10, 10.0),

        # Case 3: Varying values
        # t=0, q=10.
        # Peek t=10, q=20.
        # Sum: 10->5 + 20 = 25.
        # Count: 1->0.5 + 1 = 1.5.
        # Mean: 25 / 1.5 = 16.666...
        (10, [(0, 10)], 10, 20, 25.0 / 1.5),

        # Case 4: Peek at half the time (0.5 Half-lives)
        # t=0, q=10. Peek t=5 (HL=10). Peek q=20.
        # Decay factor D = exp(-ln(2) * 5/10) = 1/sqrt(2) approx 0.7071
        # Sum = 10*D + 20
        # Count = 1*D + 1
        # Mean = (10D + 20) / (D + 1)
        (10, [(0, 10)], 5, 20, (10 * np.exp(-np.log(2) * 0.5) + 20) /
         (np.exp(-np.log(2) * 0.5) + 1)),
    ])
def test_decayed_mean(halflife, updates, peek_t, peek_q, expected_mean):
    dm = event_filtering.DecayedMean(halflife)
    for t, q in updates:
        dm.update(t, q, p=1.0)

    result = dm.get(peek_t, peek_q)
    assert np.isclose(result, expected_mean)


@pytest.mark.parametrize(
    "halflife, updates, peek_t, peek_q, expected_var",
    [
        # Case 1: Simple variance (No Decay)
        # t=0, q=10. Peek t=0, q=20.
        # Values: [10, 20]. Mean=15.
        # Var = ((10-15)^2 + (20-15)^2) / 2 = 25.0
        (10, [(0, 10)], 0, 20, 25.0),

        # Case 2: Variance with Decay (1 Half-life)
        # t=0, q=10. Peek t=10, q=20.
        # Weights: Old=0.5, New=1.0. Total W=1.5.
        # Mean = (10*0.5 + 20*1) / 1.5 = 25/1.5 = 16.666...
        # Mean_Sq = (100*0.5 + 400*1) / 1.5 = 450/1.5 = 300.
        # Var = Mean_Sq - Mean^2 = 300 - (16.666)^2 = 22.222...
        (10, [(0, 10)], 10, 20, 22.222222222),

        # Case 3: Zero Variance
        # Values are identical, variance must be 0 regardless of decay.
        (10, [(0, 5)], 10, 5, 0.0),

        # Case 4: Larger Separation (2 Half-lives)
        # t=0, q=16. Peek t=2, q=0. HL=1.
        # Weights: Old=0.25 (2 HLs), New=1.0. Total W=1.25.
        # Mean = (16*0.25 + 0) / 1.25 = 4 / 1.25 = 3.2
        # Mean_Sq = (256*0.25 + 0) / 1.25 = 64 / 1.25 = 51.2
        # Var = 51.2 - 3.2^2 = 51.2 - 10.24 = 40.96
        (1, [(0, 16)], 2, 0, 40.96),
    ])
def test_decayed_variance(halflife, updates, peek_t, peek_q, expected_var):
    dv = event_filtering.DecayedVariance(halflife)
    for item in updates:
        t, q = item[0], item[1]
        p = item[2] if len(item) > 2 else 1.0
        dv.update(t, q, p)

    result = dv.get(peek_t, peek_q)
    print(result)
    print(expected_var)
    assert np.isclose(result, expected_var)
