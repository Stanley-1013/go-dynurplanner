import numpy as np

from godynur.kinodynamics import (
    chord_deviation_bound,
    cubic_linearization_bound,
    position,
)
from godynur.panda import PandaKinematics, Q_MAX, Q_MIN


PK = PandaKinematics()


def _closed_reversal(q0, scales, h):
    """Parameters with q(h)=q0 and an interior hump per nonzero joint."""
    scales = np.asarray(scales, dtype=float)
    return (
        q0,
        2.0 * scales / h,
        -6.0 * scales / h**2,
        6.0 * scales / h**3,
        h,
    )


def _q_at(q0, v0, a0, jerk, tau):
    return np.array(
        [position(q0[i], v0[i], a0[i], jerk[i], tau) for i in range(7)]
    )


def _empirical_cartesian_chord_error(q0, v0, a0, jerk, h, samples=4001):
    q_end = _q_at(q0, v0, a0, jerk, h)
    origins_start = PK.joint_origins(q0)
    origins_end = PK.joint_origins(q_end)
    worst = 0.0
    for tau in np.linspace(0.0, h, samples):
        s = tau / h
        true_origins = PK.joint_origins(_q_at(q0, v0, a0, jerk, tau))
        chord = (1.0 - s) * origins_start + s * origins_end
        worst = max(worst, float(np.max(np.linalg.norm(true_origins - chord, axis=1))))
    return worst


def _assert_cartesian_bound(q0, v0, a0, jerk, h):
    empirical = _empirical_cartesian_chord_error(q0, v0, a0, jerk, h)
    bound = cubic_linearization_bound(q0, v0, a0, jerk, h, PK)
    assert empirical <= bound + 1e-12, (empirical, bound, bound / empirical)
    return bound / empirical


def test_reversal_with_nearly_equal_endpoints_is_bounded_through_fk():
    q0 = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
    params = _closed_reversal(q0, [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0)

    ratio = _assert_cartesian_bound(*params)

    assert ratio >= 1.0


def test_small_angle_reversal_preserves_first_order_bound_through_fk():
    q0 = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
    params = _closed_reversal(q0, [1e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0)

    ratio = _assert_cartesian_bound(*params)

    assert ratio >= 1.0


def test_simultaneous_multi_joint_reversals_are_bounded_through_fk():
    q0 = np.array([0.2, -0.5, 0.1, -2.0, -0.2, 1.8, 0.5])
    params = _closed_reversal(
        q0, [0.20, -0.16, 0.12, -0.10, 0.08, -0.06, 0.04], 1.0
    )

    ratio = _assert_cartesian_bound(*params)

    assert ratio >= 1.0


def test_total_bound_is_exactly_term_a_plus_additive_term_b():
    q0 = np.array([0.1, -0.4, 0.2, -2.1, 0.1, 1.9, 0.6])
    v0 = np.array([0.2, -0.1, 0.05, 0.0, -0.03, 0.04, -0.02])
    a0 = np.array([0.4, -0.3, 0.2, -0.1, 0.08, -0.06, 0.04])
    jerk = np.array([-0.5, 0.4, -0.3, 0.2, -0.1, 0.08, -0.06])
    h = 0.3
    q_end = _q_at(q0, v0, a0, jerk, h)
    term_a = PK.chord_error_bound(q0, q_end - q0)
    term_b = sum(
        np.sqrt(radius) * chord_deviation_bound(q, v, a, j, h)
        for q, v, a, j, radius in zip(
            q0, v0, a0, jerk, PK.downstream_length_bounds(q0)
        )
    )

    total = cubic_linearization_bound(q0, v0, a0, jerk, h, PK)

    assert total == term_a + term_b


def test_near_margin_substitute_bounds_each_joint_against_dense_scan():
    """Scenario (d) substitute: joint-space-only, near position limits."""
    q0 = np.where(np.arange(7) % 2 == 0, Q_MAX - 1e-4, Q_MIN + 1e-4)
    v0 = np.array([0.18, -0.11, 0.07, -0.05, 0.03, -0.02, 0.01])
    a0 = np.array([-0.7, 0.5, -0.4, 0.3, -0.2, 0.1, -0.05])
    jerk = np.array([0.8, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1])
    h = 0.4
    q_end = _q_at(q0, v0, a0, jerk, h)

    for i in range(7):
        claimed = chord_deviation_bound(q0[i], v0[i], a0[i], jerk[i], h)
        empirical = max(
            abs(
                position(q0[i], v0[i], a0[i], jerk[i], tau)
                - (q0[i] + (q_end[i] - q0[i]) * tau / h)
            )
            for tau in np.linspace(0.0, h, 10001)
        )
        assert empirical <= claimed + 1e-12, (i, empirical, claimed)
