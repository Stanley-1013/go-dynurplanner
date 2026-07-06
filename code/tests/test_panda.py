import numpy as np
import pytest

from godynur import PandaKinematics
from godynur.panda import Q_MAX, Q_MIN

rng = np.random.default_rng(0)
kin = PandaKinematics()


def random_q():
    return Q_MIN + rng.random(7) * (Q_MAX - Q_MIN)


def test_origin_count_and_base():
    o = kin.joint_origins(np.zeros(7))
    assert o.shape == (9, 3)
    assert np.allclose(o[0], 0.0)
    # O1 sits at shoulder height d1 = 0.333 on the z-axis.
    assert np.allclose(o[1], [0.0, 0.0, 0.333], atol=1e-12)


def test_joint1_rotation_preserves_heights():
    """Rotating joint 1 (about base z) must not change any origin's z."""
    q = random_q()
    z_ref = kin.joint_origins(q)[:, 2]
    for dq1 in [-1.0, 0.5, 2.0]:
        q2 = q.copy()
        q2[0] = np.clip(q[0] + dq1, Q_MIN[0], Q_MAX[0])
        z = kin.joint_origins(q2)[:, 2]
        assert np.allclose(z, z_ref, atol=1e-10)


def test_rigid_link_lengths_invariant():
    """Distances between consecutive origins are configuration-independent."""
    l_ref = None
    for _ in range(20):
        o = kin.joint_origins(random_q())
        l = np.linalg.norm(np.diff(o, axis=0), axis=1)
        if l_ref is None:
            l_ref = l
        else:
            assert np.allclose(l, l_ref, atol=1e-10)


def test_flange_within_reach():
    """Flange stays within the arm's kinematic sphere (~0.855 m horizontal
    reach; generous bound on 3D distance from base)."""
    for _ in range(200):
        p = kin.flange(random_q())
        assert np.linalg.norm(p) < 1.3


def test_segments_shape_and_positive_length():
    segs = kin.segments(random_q())
    assert segs.shape == (4, 2, 3)
    lengths = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    assert np.all(lengths > 0.05)


def test_chord_error_bound_scaling():
    """Bound is quadratic in step size and zero for zero step."""
    q = random_q()
    assert kin.chord_error_bound(q, np.zeros(7)) == 0.0
    e1 = kin.chord_error_bound(q, np.full(7, 0.05))
    e2 = kin.chord_error_bound(q, np.full(7, 0.10))
    assert e2 == pytest.approx(4.0 * e1, rel=1e-9)
    # Magnitude: at the recommended action clip |dq| <= 0.05 rad/joint the
    # bound stays ~2 cm (fits the lemma inflation budget). The bound is
    # measurably ~30x conservative vs empirical deviation (2-3 mm at 0.1 rad);
    # tightening via exact Hessian norms is a known improvement path.
    assert e1 < 0.021
    assert e2 < 0.085


def test_chord_error_bound_covers_reality():
    """Empirical check of the lemma's eps_lin: the true joint-interpolated
    path deviates from the chord by less than the bound, at several interior
    points. (This test caught the blueprint's original per-joint bound
    dropping the cross terms — keep it strict.)"""
    for _ in range(200):
        q = random_q()
        dq = (rng.random(7) - 0.5) * 0.2  # up to 0.1 rad steps
        q_end = np.clip(q + dq, Q_MIN, Q_MAX)
        dq = q_end - q
        bound = kin.chord_error_bound(q, dq)
        o_start = kin.joint_origins(q)
        o_end = kin.joint_origins(q_end)
        for s in (0.25, 0.5, 0.75):
            o_true = kin.joint_origins(q + s * dq)
            chord = (1 - s) * o_start + s * o_end
            dev = np.max(np.linalg.norm(o_true - chord, axis=1))
            assert dev <= bound + 1e-12, (dev, bound, s)
