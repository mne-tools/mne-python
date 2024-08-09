# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from mne._fiff.constants import FIFF
from mne.io import read_info

base_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = base_dir / "test_chpi_raw_sss.fif"


def test_maxfilter_io():
    """Test maxfilter io."""
    info = read_info(raw_fname)
    mf = info["proc_history"][1]["max_info"]

    assert mf["sss_info"]["frame"] == FIFF.FIFFV_COORD_HEAD
    # based on manual 2.0, rev. 5.0 page 23
    assert 5 <= mf["sss_info"]["in_order"] <= 11
    assert mf["sss_info"]["out_order"] <= 5
    assert mf["sss_info"]["nchan"] > len(mf["sss_info"]["components"])

    assert (
        info["ch_names"][: mf["sss_info"]["nchan"]] == mf["sss_ctc"]["proj_items_chs"]
    )
    assert mf["sss_ctc"]["decoupler"].shape == (
        mf["sss_info"]["nchan"],
        mf["sss_info"]["nchan"],
    )
    assert_array_equal(
        np.unique(np.diag(mf["sss_ctc"]["decoupler"].toarray())),
        np.array([1.0], dtype=np.float32),
    )
    assert mf["sss_cal"]["cal_corrs"].shape == (306, 14)
    assert mf["sss_cal"]["cal_chans"].shape == (306, 2)
    vv_coils = [v for k, v in FIFF.items() if "FIFFV_COIL_VV" in k]
    assert all(k in vv_coils for k in set(mf["sss_cal"]["cal_chans"][:, 1]))
