import numpy as np
import pytest

from godynur.kinodynamics import (
    acceleration,
    braking_feasible,
    braking_witness_jerk,
    chord_deviation_bound,
    continuous_state,
    discrete_update,
    interval_extrema,
    interval_within_limits,
    position,
    velocity,
    velocity_margin,
)


@pytest.mark.parametrize(
    "state,j,h,expected",
    [
        (
            np.array([1.0, 2.0, 3.0]),
            4.0,
            0.5,
            np.array([2.0 + 3.0 / 8.0 + 1.0 / 12.0, 4.0, 5.0]),
        ),
        (
            np.array([-0.5, -1.0, 0.25]),
            -2.0,
            0.2,
            np.array([-0.5 - 0.2 + 0.005 - 0.008 / 3.0, -0.99, -0.15]),
        ),
    ],
)
def test_discrete_update_matches_hand_computation(state, j, h, expected):
    # Act
    actual = discrete_update(state, j, h)

    # Assert
    assert np.allclose(actual, expected)


def test_continuous_trajectory_matches_both_discrete_endpoints():
    # Arrange
    state = np.array([0.7, -1.2, 0.4])
    j = 3.5
    h = 0.3

    # Act
    start = continuous_state(*state, j, 0.0)
    end = continuous_state(*state, j, h)
    discrete_end = discrete_update(state, j, h)

    # Assert
    assert np.array_equal(start, state)
    assert np.allclose(end, discrete_end)
    assert acceleration(state[2], j, 0.0) == state[2]
    assert velocity(state[1], state[2], j, 0.0) == state[1]
    assert position(*state, j, 0.0) == state[0]


def test_interval_extrema_catch_mid_interval_position_violation():
    """Endpoint-only checking misses this cubic position overshoot."""
    # Arrange: q(0) = q(1) = 0, but v(tau)=0 at tau=1-sqrt(3)/3,
    # where q(tau) is a strict interior maximum of about 0.385.
    q0, v0, a0, j, h = 0.0, 2.0, -6.0, 6.0, 1.0
    q_min, q_max = -1.0, 0.3

    # Act
    q_at_end = position(q0, v0, a0, j, h)
    endpoint_only_would_pass = (
        q_min <= q0 <= q_max and q_min <= q_at_end <= q_max
    )
    extrema = interval_extrema(q0, v0, a0, j, h)
    interval_is_safe = interval_within_limits(
        q0,
        v0,
        a0,
        j,
        h,
        q_min,
        q_max,
        -2.0,
        2.0,
        -6.0,
        0.0,
        0.0,
        6.0,
    )

    # Assert: make the endpoint-checking failure mode explicit.
    assert endpoint_only_would_pass
    assert extrema.q_max > q_max
    assert not interval_is_safe


def test_interval_extrema_include_velocity_critical_point():
    # Arrange: acceleration changes sign at tau=0.5.
    q0, v0, a0, j, h = 0.0, 1.0, -2.0, 4.0, 1.0

    # Act
    extrema = interval_extrema(q0, v0, a0, j, h)

    # Assert
    assert np.isclose(extrema.v_min, 0.5)
    assert np.isclose(extrema.v_max, 1.0)


def test_interval_limit_check_covers_acceleration_and_jerk():
    # Arrange / Act / Assert
    assert interval_within_limits(
        0.0, 0.0, 0.0, 2.0, 0.2, -1.0, 1.0, -1.0, 1.0, -0.5, 0.5, -3.0, 3.0
    )
    assert not interval_within_limits(
        0.0, 0.0, 0.0, 3.0, 0.2, -1.0, 1.0, -1.0, 1.0, -0.5, 0.5, -2.0, 2.0
    )


def test_braking_is_infeasible_near_position_limit():
    # Arrange: positive motion is already only 0.01 rad from q_max.
    limits = (-2.0, 1.0, -2.0, 2.0, -2.0, 2.0, -20.0, 20.0)

    # Act
    feasible = braking_feasible(0.99, 0.5, 0.0, 0.1, 20, *limits)

    # Assert
    assert not feasible


def test_braking_is_feasible_with_clear_stopping_room():
    # Arrange: the same motion starts in the middle of a wide position range.
    limits = (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -20.0, 20.0)

    # Act
    feasible = braking_feasible(0.0, 0.5, 0.0, 0.1, 20, *limits)

    # Assert
    assert feasible


def test_braking_witness_jerk_is_immediately_legal_with_stopping_room():
    # Arrange: mirror the feasible braking scenario above.
    state = np.array([0.0, 0.5, 0.0])
    h = 0.1
    limits = (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -20.0, 20.0)

    # Act
    jerk = braking_witness_jerk(*state, h, 20, *limits)

    # Assert
    assert isinstance(jerk, float)
    next_state = discrete_update(state, jerk, h)
    assert limits[0] <= next_state[0] <= limits[1]
    assert limits[2] <= next_state[1] <= limits[3]
    assert limits[4] <= next_state[2] <= limits[5]
    assert limits[6] <= jerk <= limits[7]
    assert interval_within_limits(*state, jerk, h, *limits)


def test_braking_witness_jerk_is_none_near_position_limit():
    # Arrange: mirror the infeasible braking scenario above.
    limits = (-2.0, 1.0, -2.0, 2.0, -2.0, 2.0, -20.0, 20.0)

    # Act
    jerk = braking_witness_jerk(0.99, 0.5, 0.0, 0.1, 20, *limits)

    # Assert
    assert jerk is None


def test_chord_deviation_bound_matches_reversal_hump_closed_form():
    """The endpoint chord is zero while the cubic makes a large excursion."""
    q0, v0, a0, j, h = 0.0, 2.0, -6.0, 6.0, 1.0

    bound = chord_deviation_bound(q0, v0, a0, j, h)

    assert position(q0, v0, a0, j, h) == pytest.approx(q0)
    assert bound == pytest.approx(2.0 * np.sqrt(3.0) / 9.0, rel=1e-12)


def test_chord_deviation_bound_is_tiny_for_nearly_constant_monotonic_motion():
    q0, v0, a0, j, h = 0.4, 1.0, 1e-6, 0.0, 1.0
    endpoint_delta = position(q0, v0, a0, j, h) - q0

    bound = chord_deviation_bound(q0, v0, a0, j, h)

    assert bound == pytest.approx(a0 * h**2 / 8.0, rel=1e-12)
    assert bound / abs(endpoint_delta) < 1e-6


def test_velocity_margin_is_large_in_both_directions_with_stopping_room():
    limits = (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -20.0, 20.0)

    margin_plus = velocity_margin(
        0.0, 0.0, 0.0, 1.0, 0.1, 20, *limits, upper_bound=2.0
    )
    margin_minus = velocity_margin(
        0.0, 0.0, 0.0, -1.0, 0.1, 20, *limits, upper_bound=2.0
    )

    assert margin_plus > 1.9
    assert margin_minus > 1.9


def test_velocity_margin_is_asymmetric_near_position_limit():
    limits = (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -20.0, 20.0)

    toward_limit = velocity_margin(
        1.99, 0.0, 0.0, 1.0, 0.1, 20, *limits, upper_bound=2.0
    )
    away_from_limit = velocity_margin(
        1.99, 0.0, 0.0, -1.0, 0.1, 20, *limits, upper_bound=2.0
    )

    assert toward_limit < 0.15
    assert away_from_limit > 1.9
