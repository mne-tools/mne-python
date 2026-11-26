# Authors: Roberto Guidotti  <rob.guidotti@gmail.com>
#          simplified BSD-3 license

from numpy.testing import assert_array_almost_equal

from mne.datasets import testing
from mne.io import read_raw_fieldtrip, read_raw_itab
from mne.io.tests.test_raw import _test_raw_reader

data_path = testing.data_path(download=False)

mat_itab_fname = data_path / "itab" / "test_itab.mat"
raw_itab_fname = data_path / "itab" / "test_itab.raw"


# @testing.requires_testing_data
def test_itab_raw():
    """Test reading ITAB .raw files."""
    raw = read_raw_itab(raw_itab_fname, preload=True)
    assert "RawITAB" in repr(raw)

    _test_raw_reader(
        read_raw_itab,
        fname=raw_itab_fname,
        test_scaling=False,
    )

    test_ft_raw = read_raw_fieldtrip(mat_itab_fname, info=raw.info, data_name="data")

    itab_data = raw.get_data()
    ft_data = test_ft_raw.get_data()

    assert_array_almost_equal(ft_data, itab_data)

    assert itab_data.shape == ft_data.shape
    assert test_ft_raw.info["sfreq"] == raw.info["sfreq"]
