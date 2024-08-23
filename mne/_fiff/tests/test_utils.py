"""Run tests for the utilities."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from mne._fiff.utils import _check_orig_units


def test_check_orig_units():
    """Test the checking of original units."""
    orig_units = dict(FC1="nV", Hfp3erz="n/a", Pz="uV", greekMu="μV", microSign="µV")
    orig_units = _check_orig_units(orig_units)
    assert orig_units["FC1"] == "nV"
    assert orig_units["Hfp3erz"] == "n/a"
    assert orig_units["Pz"] == "µV"
    assert orig_units["greekMu"] == "µV"
    assert orig_units["microSign"] == "µV"
