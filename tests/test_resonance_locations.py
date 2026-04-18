"""Resonance-location helper gives the textbook values."""

from __future__ import annotations

import math

import pytest

from kirkwood_gpu.physics import resonance_semimajor_axis, SEMIMAJOR_JUPITER


# Reference values for interior p:q MMRs at a_J = 5.2044 AU.
# Quoted to 4 sig figs from direct evaluation of a_J * (q/p)^(2/3);
# consistent with Murray & Dermott Table 9.1 to 3 sig figs.
_REFERENCE = {
    (3, 1): 2.5017,  # 3:1 Kirkwood gap
    (2, 1): 3.2784,  # 2:1 Kirkwood gap
    (3, 2): 3.9714,  # Hilda group (stable libration, not a gap)
    (5, 2): 2.8254,  # 5:2 Kirkwood feature
    (7, 3): 2.9584,  # 7:3 feature
}


@pytest.mark.parametrize("pq,expected", _REFERENCE.items())
def test_resonance_semimajor_axis_matches_reference(pq, expected):
    p, q = pq
    got = resonance_semimajor_axis(p, q, a_J=SEMIMAJOR_JUPITER)
    # 4-sig-fig agreement <=> |Delta| < 5e-4 AU at these magnitudes.
    assert got == pytest.approx(expected, abs=5e-4), (
        f"resonance {p}:{q} -> {got:.4f} AU, expected {expected:.4f} AU"
    )


def test_resonance_matches_published_3sig():
    """Coarser sanity check against canonical literature values."""
    assert resonance_semimajor_axis(3, 1) == pytest.approx(2.50, abs=1e-2)
    assert resonance_semimajor_axis(2, 1) == pytest.approx(3.28, abs=1e-2)
    assert resonance_semimajor_axis(3, 2) == pytest.approx(3.97, abs=1e-2)


def test_resonance_rejects_nonpositive():
    with pytest.raises(ValueError):
        resonance_semimajor_axis(0, 1)
    with pytest.raises(ValueError):
        resonance_semimajor_axis(3, -1)
