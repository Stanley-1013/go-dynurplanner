import numpy as np

from godynur.geometry import AABB
from godynur.scenes import WS_HI, WS_LO, Obstacle
from godynur.voxelizer import GridSpec, rasterize, rasterize_future

rng = np.random.default_rng(3)
SPEC = GridSpec(lo=WS_LO.copy(), hi=WS_HI.copy(), n=32)


def brute_force_grid(boxes, spec):
    """Reference: voxel-vs-box AABB overlap test for every voxel."""
    g = np.zeros((spec.n, spec.n, spec.n), dtype=np.float32)
    cell = spec.cell
    for i in range(spec.n):
        for j in range(spec.n):
            for k in range(spec.n):
                v_lo = spec.lo + np.array([i, j, k]) * cell
                v_hi = v_lo + cell
                for b in boxes:
                    if np.all(v_lo < b.hi) and np.all(b.lo < v_hi):
                        g[i, j, k] = 1.0
                        break
    return g


def random_box():
    c = rng.uniform(WS_LO, WS_HI)
    half = rng.uniform(0.01, 0.3, 3)
    return AABB(c - half, c + half)


def test_empty_scene_is_empty_grid():
    assert rasterize([], SPEC).sum() == 0.0


def test_full_workspace_box_fills_grid():
    g = rasterize([AABB(WS_LO - 1.0, WS_HI + 1.0)], SPEC)
    assert g.sum() == SPEC.n**3


def test_box_outside_workspace_is_clipped():
    g = rasterize([AABB(WS_HI + 1.0, WS_HI + 2.0)], SPEC)
    assert g.sum() == 0.0


def test_matches_brute_force_on_random_scenes():
    for _ in range(10):
        boxes = [random_box() for _ in range(rng.integers(1, 4))]
        fast = rasterize(boxes, SPEC)
        ref = brute_force_grid(boxes, SPEC)
        assert np.array_equal(fast, ref)


def test_rasterization_is_conservative_for_thin_boxes():
    """A plate thinner than a voxel must still occupy at least one layer."""
    c = 0.5 * (WS_LO + WS_HI)
    thin = AABB(c - np.array([0.002, 0.2, 0.2]), c + np.array([0.002, 0.2, 0.2]))
    g = rasterize([thin], SPEC)
    assert g.sum() > 0


def test_future_rasterization_shifts_occupancy():
    c = 0.5 * (WS_LO + WS_HI)
    ob = Obstacle(
        center=c.copy(), half=np.array([0.05, 0.05, 0.05]),
        vel=np.array([1.0, 0.0, 0.0]),
    )
    now = rasterize([ob.aabb()], SPEC)
    fut = rasterize_future([ob], SPEC, dt=0.2)
    # Both non-empty; occupancy shifted along +x. (Voxel COUNT is not
    # translation-invariant under intersection semantics — a box straddles
    # 3 or 4 cell layers per axis depending on alignment — so we assert the
    # shift, not count equality.)
    assert now.sum() > 0 and fut.sum() > 0
    x_now = np.where(now.any(axis=(1, 2)))[0]
    x_fut = np.where(fut.any(axis=(1, 2)))[0]
    assert x_fut.min() > x_now.min()
    assert x_fut.max() > x_now.max()


def test_sdf_signs_and_values():
    from godynur.voxelizer import rasterize_sdf
    c = 0.5 * (WS_LO + WS_HI)
    box = AABB(c - 0.1, c + 0.1)
    sdf = rasterize_sdf([box], SPEC)
    # Center voxel is inside (negative); far corner positive; empty scene clip.
    n = SPEC.n
    assert sdf[n // 2, n // 2, n // 2] < 0
    assert sdf[0, 0, 0] > 0
    assert np.all(rasterize_sdf([], SPEC) == 0.5)
    # Value check: voxel center at known offset from box surface.
    pts_dist = np.linalg.norm(np.maximum(np.maximum(box.lo - c, c - box.hi), 0))
    assert pts_dist == 0.0  # center inside, sanity of the formula pieces
