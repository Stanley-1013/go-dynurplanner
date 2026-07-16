import numpy as np
import pytest

from godynur.kinodynamics import (
    acceleration,
    braking_feasible,
    continuous_state,
    discrete_update,
    interval_extrema,
    interval_within_limits,
    position,
    velocity,
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
