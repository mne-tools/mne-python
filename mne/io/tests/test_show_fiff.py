# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from pathlib import Path

from mne.io import show_fiff

base_dir = Path(__file__).parent / "data"
fname_evoked = base_dir / "test-ave.fif"
fname_raw = base_dir / "test_raw.fif"
fname_c_annot = base_dir / "test_raw-annot.fif"


def test_show_fiff():
    """Test show_fiff."""
    # this is not exhaustive, but hopefully bugs will be found in use
    info = show_fiff(fname_evoked)
    assert "BAD" not in info
    keys = [
        "FIFF_EPOCH",
        "FIFFB_HPI_COIL",
        "FIFFB_PROJ_ITEM",
        "FIFFB_PROCESSED_DATA",
        "FIFFB_EVOKED",
        "FIFF_NAVE",
        "FIFF_EPOCH",
        "COORD_TRANS",
    ]
    assert all(key in info for key in keys)
    info = show_fiff(fname_raw, read_limit=1024)
    assert "BAD" not in info
    info = show_fiff(fname_c_annot)
    assert "BAD" not in info
    assert ">B" in info, info
