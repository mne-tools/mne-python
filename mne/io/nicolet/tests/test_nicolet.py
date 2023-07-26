# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD-3-Clause

from pathlib import Path

from mne.io import read_raw_nicolet
from mne.io.tests.test_raw import _test_raw_reader

import pytest

base_dir = Path(__file__).parent / "data"
fname_data = base_dir / "test_nicolet_raw.data"
fname_head = base_dir / "test_nicolet_raw.head"


def test_data():
    """Test reading raw nicolet files."""
    _test_raw_reader(
        read_raw_nicolet,
        input_fname=fname_data,
        ch_type="eeg",
        ecg="auto",
        eog="auto",
        emg="auto",
        misc=["PHO"],
    )

    with pytest.raises(
        ValueError, match='File name should end with .data not ".head".'
    ):
        read_raw_nicolet(fname_head, "eeg")
