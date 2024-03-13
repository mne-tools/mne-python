# Authors: Velu Prabhakar Kumaravel <vpr.kumaravel@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys
from pathlib import Path

import pytest

from mne.io import read_raw_fif
from mne.preprocessing import find_bad_channels_lof
from mne.utils import check_version

base_dir = Path(__file__).parent.parent.parent / "io" / "tests" / "data"
raw_fname = base_dir / "test_raw.fif"


@pytest.mark.parametrize(
    "n_neighbors, ch_type, n_ch, n_bad",
    [
        (8, "eeg", 60, 8),
        (10, "grad", 204, 2),
        (20, "mag", 102, 0),
        (30, "grad", 204, 2),
    ],
)
def test_lof(n_neighbors, ch_type, n_ch, n_bad):
    """Test LOF detection."""
    pytest.importorskip("sklearn")
    if sys.platform == "win32" and check_version("sklearn", "1.5.dev"):
        pytest.skip("https://github.com/scikit-learn/scikit-learn/issues/28625")
    raw = read_raw_fif(raw_fname).load_data()
    assert raw.info["bads"] == []
    bads, scores = find_bad_channels_lof(
        raw, n_neighbors, picks=ch_type, return_scores=True
    )
    bads_2 = find_bad_channels_lof(raw, n_neighbors, picks=ch_type)
    assert len(scores) == n_ch
    assert len(bads) == n_bad
    assert bads == bads_2
    with pytest.raises(ValueError, match="channel type"):
        find_bad_channels_lof(raw)
