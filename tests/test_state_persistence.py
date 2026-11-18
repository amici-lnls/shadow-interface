"""Regression tests for optical-element offset/tilt state persistence.

Shadow overwrites X_ROT/Y_ROT/Z_ROT with radians inside traceOE(), even though
they are supplied in degrees. The wrapper must therefore keep its own canonical
state and re-apply it (in degrees) before every trace; otherwise each trace
shrinks the tilt by pi/180.
"""

import numpy as np
import pytest

from shadow_interface.si_beamlines import CAXSim


@pytest.fixture
def mirror():
    """A freshly built CAX mirror attached to a (small) traced beamline."""
    return CAXSim(total_rays=500, image_size=(32, 32)).mirror


def test_tilt_is_stable_over_many_traces(mirror):
    mirror.ry = 25.25e-3  # mrad
    expected = mirror.tilt.copy()

    for _ in range(8):
        mirror.beamline.trace()
        np.testing.assert_allclose(mirror.tilt, expected, atol=1e-12)


def test_changing_offset_does_not_change_tilt(mirror):
    mirror.ry = 25.25e-3
    mirror.beamline.trace()
    expected_tilt = mirror.tilt.copy()

    mirror.ty = 0.5
    mirror.beamline.trace()

    np.testing.assert_allclose(mirror.tilt, expected_tilt, atol=1e-12)
    assert mirror.ty == pytest.approx(0.5)


def test_getters_are_valid_before_and_after_trace(mirror):
    mirror.tilt = np.array([1.0, 2.0, 3.0])
    mirror.offset = np.array([0.1, 0.2, 0.3])

    # Before tracing, shadow_oe still holds degrees; the cached state does not
    # depend on that unit switch.
    np.testing.assert_allclose(mirror.tilt, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(mirror.offset, [0.1, 0.2, 0.3])

    mirror.beamline.trace()

    np.testing.assert_allclose(mirror.tilt, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(mirror.offset, [0.1, 0.2, 0.3])


def test_reset_restores_degrees_after_shadow_wrote_radians(mirror):
    mirror.ry = 25.25e-3
    mirror.beamline.trace()

    # After traceOE, Shadow has converted the input degrees to radians.
    radians_after_trace = mirror.shadow_oe.X_ROT
    assert radians_after_trace == pytest.approx(-25.25e-6, abs=1e-12)

    # reset() must write degrees again, not preserve the radian values.
    mirror.reset()
    expected_degrees = -np.rad2deg(25.25e-6)
    assert mirror.shadow_oe.X_ROT == pytest.approx(expected_degrees, abs=1e-12)
    np.testing.assert_allclose(mirror.tilt, [0.0, 25.25e-3, 0.0], atol=1e-12)
