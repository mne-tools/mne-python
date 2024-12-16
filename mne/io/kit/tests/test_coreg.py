# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne.io.kit import read_mrk

mrk_fname = Path(__file__).parent / "data" / "test_mrk.sqd"


def test_io_mrk(tmp_path):
    """Test IO for mrk files."""
    pts = read_mrk(mrk_fname)

    # txt
    path = tmp_path / "mrk.txt"
    with open(path, "wb") as fid:
        fid.write(b"%% %d 3D points, x y z per line\n" % len(pts))
        np.savetxt(fid, pts, delimiter="\t", newline="\n")

    pts_2 = read_mrk(path)
    assert_array_equal(pts, pts_2, "read/write mrk to text")

    # unsupported extension
    fname = tmp_path / "file.ext"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_mrk(fname)
    fname.write_text("")
    with pytest.raises(ValueError, match="file extension"):
        read_mrk(fname)
