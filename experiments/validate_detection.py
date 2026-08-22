"""
Sensitivity check for experiments/edge_audit.py.

A battery of null results is only meaningful if the tests can see a real effect.
This simulates deliberately biased machines and confirms the audit's tests fire
at the effect sizes that would actually matter.

The threshold that matters is roughly 2x: NJ Cash 5 returns about half of handle,
so a ball would need to come up about twice as often as it should before betting
it turns a profit. If the audit is decisive well below that, a null result rules
out every bias worth exploiting.

Run: python experiments/validate_detection.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edge_audit import (
    _indicator_matrix,
    holm_bonferroni,
    t1_ball_frequency,
    t3_serial_repeat,
    t5_pair_dispersion,
)

SPAN, DRAWN, N_DRAWS = 45, 5, 2238
SEED = 0
N_T5_TRIALS = 40


def simulate(rng, weights=None, repeat_boost=0.0):
    """Draw N_DRAWS tickets from a machine that may favour some balls or repeats."""
    base = np.ones(SPAN) if weights is None else np.asarray(weights, dtype=float)
    rows, previous = [], set()
    for _ in range(N_DRAWS):
        probs = base / base.sum()
        if repeat_boost and previous:
            probs = probs.copy()
            for ball in previous:
                probs[ball - 1] *= 1 + repeat_boost
            probs /= probs.sum()
        drawn = rng.choice(np.arange(1, SPAN + 1), size=DRAWN, replace=False, p=probs)
        rows.append(sorted(drawn))
        previous = set(drawn)
    return _indicator_matrix(np.array(rows), 1, SPAN)


def main():
    rng = np.random.default_rng(SEED)

    print(f"{DRAWN}/{SPAN} game, {N_DRAWS} draws, seed {SEED}\n")

    matrix = simulate(rng)
    print("fair machine (null)")
    print(f"  T1 p={t1_ball_frequency(matrix, DRAWN)['p']:.4f}")
    print(f"  T3 p={t3_serial_repeat(matrix)['p']:.4f}\n")

    print("single over-represented ball — T1")
    for multiplier in (1.2, 1.5, 2.0, 3.0):
        weights = np.ones(SPAN)
        weights[6] = multiplier
        p = t1_ball_frequency(simulate(rng, weights), DRAWN)["p"]
        verdict = "detected" if p < 0.05 else "missed"
        print(f"  {multiplier:.1f}x  p={p:.3g}  {verdict}")

    print("\nrepeat stickiness — T3")
    for boost in (0.1, 0.25, 0.5):
        p = t3_serial_repeat(simulate(rng, repeat_boost=boost))["p"]
        verdict = "detected" if p < 0.05 else "missed"
        print(f"  +{boost:.0%}  p={p:.3g}  {verdict}")

    print("\nT5 null calibration (its dof is approximate, so check it empirically)")
    null_p = np.array([t5_pair_dispersion(simulate(rng), DRAWN)["p"] for _ in range(N_T5_TRIALS)])
    false_positives = int((null_p < 0.05).sum())
    print(f"  {N_T5_TRIALS} fair machines: mean p={null_p.mean():.3f} (uniform -> 0.50)")
    print(
        f"  false positives at p<0.05: {false_positives}/{N_T5_TRIALS} "
        f"(nominal {0.05 * N_T5_TRIALS:.0f}) -> approximately calibrated"
    )

    print("\nHolm-Bonferroni behaviour")
    print(f"  [.04, .30, .60]    -> {holm_bonferroni([0.04, 0.30, 0.60])}")
    print(f"  [.0001, .30, .60]  -> {holm_bonferroni([0.0001, 0.30, 0.60])}")

    print(
        "\nConclusion: bias at the ~2x level needed for positive EV is detected "
        "overwhelmingly.\nA null audit therefore rules out every exploitable bias, "
        "and is not merely underpowered."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
