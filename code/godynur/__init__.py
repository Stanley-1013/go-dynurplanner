"""GO-DynURPlanner M0 core: analytic parameterized space with continuous-time
collision ground truth.

Modules
-------
panda       : Franka Emika Panda kinematics (modified DH), URPlanner link
              consolidation, chord-error bound (conservativeness lemma term).
geometry    : static segment-vs-AABB overlap (URPlanner Eq.6-10) and UOAR.
continuous  : exact continuous-time collision reasoning over a control
              interval [0, dt] for moving segments vs moving boxes:
              first-contact time tau*, swept-overlap integral, inflation.
"""

from .panda import PandaKinematics
from .geometry import AABB, segment_box_overlap, uoar
from .continuous import (
    MovingBox,
    MovingSegment,
    first_contact_time,
    interval_collision_free,
    swept_overlap_integral,
    lemma_inflation,
)

__all__ = [
    "PandaKinematics",
    "AABB",
    "segment_box_overlap",
    "uoar",
    "MovingBox",
    "MovingSegment",
    "first_contact_time",
    "interval_collision_free",
    "swept_overlap_integral",
    "lemma_inflation",
]
