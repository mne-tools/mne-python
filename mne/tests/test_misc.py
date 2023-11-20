# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from mne.misc import parse_config

ave_fname = Path(__file__).parent.parent / "io" / "tests" / "data" / "test.ave"


def test_parse_ave():
    """Test parsing of .ave file."""
    conditions = parse_config(ave_fname)
    assert len(conditions) == 4
