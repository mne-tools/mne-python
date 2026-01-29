# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from mne.misc import parse_config

ave_fname = Path(__file__).parents[1] / "io" / "tests" / "data" / "test.ave"


def test_parse_ave():
    """Test parsing of .ave file."""
    conditions = parse_config(ave_fname)
    assert len(conditions) == 4
