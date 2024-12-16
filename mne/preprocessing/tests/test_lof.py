# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import pytest

from mne.io import read_raw_fif
from mne.preprocessing import find_bad_channels_lof

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
