import numpy as np

from godynur import (
    AABB,
    MovingBox,
    MovingSegment,
    first_contact_time,
    interval_collision_free,
    lemma_inflation,
    swept_overlap_integral,
)
from godynur.geometry import segment_box_overlap

rng = np.random.default_rng(2)


def dense_scan(seg, mbox, dt, n=4001):
    """Sampling-based reference: first sampled tau with overlap (or None)."""
    for tau in np.linspace(0.0, dt, n):
        a = seg.a0 + (seg.ua - mbox.v) * tau
        b = seg.b0 + (seg.ub - mbox.v) * tau
        l, _, _ = segment_box_overlap(a, b, mbox.box)
        if l > 1e-12:
            return tau
    return None


def random_case(dt):
    c = rng.uniform(-1, 1, 3)
    half = rng.uniform(0.05, 0.5, 3)
    mbox = MovingBox(AABB(c - half, c + half), rng.uniform(-1.5, 1.5, 3))
    seg = MovingSegment(
        a0=rng.uniform(-1.5, 1.5, 3),
        ua=rng.uniform(-2, 2, 3),
        b0=rng.uniform(-1.5, 1.5, 3),
        ub=rng.uniform(-2, 2, 3),
    )
    return seg, mbox


def test_first_contact_agrees_with_dense_sampling():
    dt = 0.5
    n_hit = 0
    for _ in range(300):
        seg, mbox = random_case(dt)
        tau_exact = first_contact_time(seg, mbox, dt)
        tau_dense = dense_scan(seg, mbox, dt)
        if tau_dense is not None:
            n_hit += 1
            # Exact checker must not miss anything sampling finds, and must
            # report contact no later than the first sampled hit.
            assert tau_exact is not None
            assert tau_exact <= tau_dense + 1e-9
        elif tau_exact is not None:
            # Checker found a contact the 4001-point scan missed: it must be
            # a genuine sub-grid event — verify by re-sampling finely around it.
            local = dense_scan(seg, mbox, dt, n=400_001)
            assert local is not None
    assert n_hit > 30  # ensure the test actually exercises collisions


def test_tunneling_case_caught():
    """The paper's Figure-1 seed: a thin fast box crosses a static segment
    strictly between the interval endpoints. Discrete endpoint checking
    misses it; the continuous checker must catch it."""
    dt = 0.1
    # Static segment along y at x=0, z in [0.2, 0.8].
    seg = MovingSegment(
        a0=np.array([0.0, -0.5, 0.5]),
        ua=np.zeros(3),
        b0=np.array([0.0, 0.5, 0.5]),
        ub=np.zeros(3),
    )
    # Thin box (2 cm along x) starting left of the segment, crossing at 4 m/s:
    # at tau=0 box spans x in [-0.22, -0.20]; at tau=dt spans [0.18, 0.20].
    box = AABB(np.array([-0.22, -0.2, 0.3]), np.array([-0.20, 0.2, 0.7]))
    mbox = MovingBox(box, np.array([4.0, 0.0, 0.0]))

    # Discrete endpoint check (what standard RL loops do): both clear.
    for tau in (0.0, dt):
        a = seg.a0 - mbox.v * tau
        b = seg.b0 - mbox.v * tau
        l, _, _ = segment_box_overlap(a, b, mbox.box)
        assert l == 0.0

    # Continuous checker: catches the crossing near tau = 0.21/4 = 0.0525.
    tau_star = first_contact_time(seg, mbox, dt)
    assert tau_star is not None
    assert abs(tau_star - 0.05) < 0.01
    assert not interval_collision_free([seg], [mbox], dt)


def test_no_contact_reported_when_truly_clear():
    dt = 0.2
    seg = MovingSegment(
        a0=np.array([0.0, -0.5, 0.5]),
        ua=np.zeros(3),
        b0=np.array([0.0, 0.5, 0.5]),
        ub=np.zeros(3),
    )
    # Box moving parallel to the segment, never approaching in x.
    box = AABB(np.array([1.0, -0.2, 0.3]), np.array([1.2, 0.2, 0.7]))
    mbox = MovingBox(box, np.array([0.0, 2.0, 0.0]))
    assert first_contact_time(seg, mbox, dt) is None
    assert interval_collision_free([seg], [mbox], dt)


def test_swept_integral_against_dense_quadrature():
    dt = 0.4
    for _ in range(40):
        seg, mbox = random_case(dt)
        exact = swept_overlap_integral(seg, mbox, dt)
        taus = np.linspace(0.0, dt, 20_001)
        ls = []
        for tau in taus:
            a = seg.a0 + (seg.ua - mbox.v) * tau
            b = seg.b0 + (seg.ub - mbox.v) * tau
            l, _, _ = segment_box_overlap(a, b, mbox.box)
            ls.append(l)
        ref = np.trapezoid(ls, taus)
        assert abs(exact - ref) < 5e-4, (exact, ref)


def test_swept_integral_reduces_to_static_uoar_limit():
    """dt -> 0 consistency: integral / dt approaches the instantaneous
    overlap length (URPlanner Eq.9) — strict compatibility claim of Loop 3."""
    seg = MovingSegment(
        a0=np.array([-0.5, 0.0, 0.0]),
        ua=np.array([0.3, 0.1, 0.0]),
        b0=np.array([0.5, 0.0, 0.0]),
        ub=np.array([0.3, -0.1, 0.0]),
    )
    box = AABB(np.array([-0.2, -1.0, -1.0]), np.array([0.2, 1.0, 1.0]))
    mbox = MovingBox(box, np.array([0.05, 0.0, 0.0]))
    l0, _, _ = segment_box_overlap(seg.a0, seg.b0, box)
    for dt in (1e-3, 1e-4):
        avg = swept_overlap_integral(seg, mbox, dt) / dt
        assert abs(avg - l0) < 5e-3


def test_lemma_inflation_values():
    """Blueprint's concrete numbers: eps_acc = a_max dt^2/2, eps_v = eps_v*dt."""
    eps = lemma_inflation(eps_lin=0.009, a_max=2.0, dt=0.05, eps_v=0.1)
    assert np.isclose(eps, 0.009 + 0.0025 + 0.005)
    assert eps < 0.025  # total budget stays ~2 cm scale


def test_inflated_check_is_conservative():
    """With lemma inflation, a clear verdict on the inflated box implies the
    tight box is also clear throughout (spot check via dense sampling)."""
    dt = 0.2
    for _ in range(100):
        seg, mbox = random_case(dt)
        infl = MovingBox(mbox.box.expanded(0.02), mbox.v)
        if first_contact_time(seg, infl, dt) is None:
            assert dense_scan(seg, mbox, dt) is None
