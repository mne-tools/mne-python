import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne.io import read_raw_neuralynx
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets.testing import data_path, requires_testing_data

testing_path = data_path(download=False) / "neuralynx"


@requires_testing_data
def test_neuralynx():
    """Test basic reading."""
    pytest.importorskip("neo")

    from neo.io import NeuralynxIO

    raw = read_raw_neuralynx(fname=testing_path, preload=True)

    d1, t1 = raw.get_data(return_times=True)

    # read all segments and concatenate
    nlx_reader = NeuralynxIO(dirname=testing_path)
    bl = nlx_reader.read(
        lazy=False
    )  # read a single block which contains the data split in segments

    # concatenate all signals and times from all segments (== total recording)
    d2 = np.concatenate(
        [sig.magnitude for seg in bl[0].segments for sig in seg.analogsignals]
    ).T
    t2 = np.concatenate(
        [sig.times.magnitude for seg in bl[0].segments for sig in seg.analogsignals]
    ).T
    ch2 = [ch[0] for ch in nlx_reader.header["signal_channels"]]

    assert ch2 == raw.ch_names
    assert_allclose(t1, t2, rtol=1e-6, err_msg="times")
    assert_allclose(d1, d2, rtol=1e-6, err_msg="data")

    _test_raw_reader(read_raw_neuralynx, fname=testing_path)
