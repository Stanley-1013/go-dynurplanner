import numpy as np

from godynur import AABB, segment_box_overlap, uoar

rng = np.random.default_rng(1)


def brute_force_overlap(p_s, p_e, box, n=200_001):
    """Numerical ground truth: fraction of finely sampled points inside."""
    lam = np.linspace(0.0, 1.0, n)
    pts = p_s[None, :] + lam[:, None] * (p_e - p_s)[None, :]
    inside = np.all((pts >= box.lo) & (pts <= box.hi), axis=1)
    return inside.mean() * np.linalg.norm(p_e - p_s)


def test_segment_fully_inside():
    box = AABB(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    l, t0, t1 = segment_box_overlap(np.array([-0.5, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), box)
    assert np.isclose(l, 1.0)
    assert (t0, t1) == (0.0, 1.0)


def test_segment_fully_outside():
    box = AABB(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    l, _, _ = segment_box_overlap(np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0]), box)
    assert l == 0.0


def test_axis_crossing_equals_box_width():
    box = AABB(np.array([0.0, 0.0, 0.0]), np.array([0.4, 1.0, 1.0]))
    l, _, _ = segment_box_overlap(np.array([-1.0, 0.5, 0.5]), np.array([2.0, 0.5, 0.5]), box)
    assert np.isclose(l, 0.4)


def test_degenerate_axis_parallel_segment():
    box = AABB(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    # Segment parallel to x at y=0.5, z=1.5 (outside): no overlap.
    l, _, _ = segment_box_overlap(np.array([-1.0, 0.5, 1.5]), np.array([2.0, 0.5, 1.5]), box)
    assert l == 0.0
    # Same but z=0.5 (inside slab): overlap = 1.0 (the box width along x).
    l, _, _ = segment_box_overlap(np.array([-1.0, 0.5, 0.5]), np.array([2.0, 0.5, 0.5]), box)
    assert np.isclose(l, 1.0)


def test_random_against_brute_force():
    for _ in range(100):
        c = rng.uniform(-1, 1, 3)
        half = rng.uniform(0.05, 0.8, 3)
        box = AABB(c - half, c + half)
        p_s = rng.uniform(-2, 2, 3)
        p_e = rng.uniform(-2, 2, 3)
        l, _, _ = segment_box_overlap(p_s, p_e, box)
        l_bf = brute_force_overlap(p_s, p_e, box)
        assert abs(l - l_bf) < 5e-4, (l, l_bf)


def test_uoar_sign_and_normalization():
    box = AABB(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    segs = np.array(
        [
            [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],  # fully inside, length 1
            [[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]],  # fully outside, length 1
        ]
    )
    r = uoar(segs, [box])
    assert np.isclose(r, -0.5)  # overlap 1 / total length 2
    assert uoar(segs[1:], [box]) == 0.0
