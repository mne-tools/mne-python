# Authors: Velu Prabhakar Kumaravel <vkumaravel@fbk.eu>
# License: BSD-3-Clause

from pathlib import Path

import pytest

from mne.io import read_raw_fif
from mne.preprocessing import find_bad_channels_lof

base_dir = Path(__file__).parent.parent.parent / "io" / "tests" / "data"
raw_fname = base_dir / "test_raw.fif"


@pytest.mark.parametrize(
    "n_neighbors, ch_type, n_bad",
    [
        (8, "eeg", 8),
        (10, "grad", 2),
        (20, "mag", 0),
        (30, "grad", 2),
    ],
)
def test_lof(n_neighbors, ch_type, n_bad):
    """Test LOF detection."""
    pytest.importorskip("sklearn")
    raw = read_raw_fif(raw_fname).load_data()
    assert raw.info["bads"] == []
    bads = find_bad_channels_lof(raw, n_neighbors, picks=ch_type)
    assert len(bads) == n_bad
    with pytest.raises(ValueError, match="channel type"):
        find_bad_channels_lof(raw)
