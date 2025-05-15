#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import pytest
from numpy import empty
from numpy.testing import assert_allclose, assert_array_equal

from mne.datasets import testing
from mne.io.curry import read_impedances_curry, read_raw_curry
from mne.io.edf import read_raw_bdf
from mne.io.tests.test_raw import _test_raw_reader

pytest.importorskip("curryreader")

data_dir = testing.data_path(download=False)
curry_dir = data_dir / "curry"
bdf_file = data_dir / "BDF" / "test_bdf_stim_channel.bdf"
curry7_bdf_file = curry_dir / "test_bdf_stim_channel Curry 7.dat"
curry7_bdf_ascii_file = curry_dir / "test_bdf_stim_channel Curry 7 ASCII.dat"
curry8_bdf_file = curry_dir / "test_bdf_stim_channel Curry 8.cdt"
curry8_bdf_ascii_file = curry_dir / "test_bdf_stim_channel Curry 8 ASCII.cdt"


@pytest.fixture(scope="session")
def bdf_curry_ref():
    """Return a view of the reference bdf used to create the curry files."""
    raw = read_raw_bdf(bdf_file, preload=True).drop_channels(["Status"])
    return raw


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname,tol",
    [
        pytest.param(curry7_bdf_file, 1e-7, id="curry 7"),
        pytest.param(curry8_bdf_file, 1e-7, id="curry 8"),
        pytest.param(curry7_bdf_ascii_file, 1e-4, id="curry 7 ascii"),
        pytest.param(curry8_bdf_ascii_file, 1e-4, id="curry 8 ascii"),
    ],
)
@pytest.mark.parametrize("preload", [True, False])
def test_read_raw_curry(fname, tol, preload, bdf_curry_ref):
    """Test reading CURRY files."""
    with pytest.raises(RuntimeWarning):  # TODO change way to add montage in curry.py!
        raw = read_raw_curry(fname, preload=preload)

        assert hasattr(raw, "_data") == preload
        assert raw.n_times == bdf_curry_ref.n_times
        assert raw.info["sfreq"] == bdf_curry_ref.info["sfreq"]

        for field in ["kind", "ch_name"]:
            assert_array_equal(
                [ch[field] for ch in raw.info["chs"]],
                [ch[field] for ch in bdf_curry_ref.info["chs"]],
            )

        assert_allclose(
            raw.get_data(verbose="error"), bdf_curry_ref.get_data(), atol=tol
        )

        picks, start, stop = ["C3", "C4"], 200, 800
        assert_allclose(
            raw.get_data(picks=picks, start=start, stop=stop, verbose="error"),
            bdf_curry_ref.get_data(picks=picks, start=start, stop=stop),
            rtol=tol,
        )
        # assert raw.info["dev_head_t"] is None  # TODO do we need this value?


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname",
    [
        pytest.param(curry7_bdf_file, id="curry 7"),
        pytest.param(curry8_bdf_file, id="curry 8"),
        pytest.param(curry7_bdf_ascii_file, id="curry 7 ascii"),
        pytest.param(curry8_bdf_ascii_file, id="curry 8 ascii"),
    ],
)
def test_read_raw_curry_test_raw(fname):
    """Test read_raw_curry with _test_raw_reader."""
    with pytest.raises(RuntimeWarning):  # TODO change way to add montage in curry.py!
        _test_raw_reader(read_raw_curry, fname=fname)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname",
    [
        pytest.param(curry7_bdf_file, id="curry 7"),
        pytest.param(curry8_bdf_file, id="curry 8"),
        pytest.param(curry7_bdf_ascii_file, id="curry 7 ascii"),
        pytest.param(curry8_bdf_ascii_file, id="curry 8 ascii"),
    ],
)
def test_read_impedances_curry(fname):
    """Test reading impedances from CURRY files."""
    _, imp = read_impedances_curry(fname)
    actual_imp = empty(shape=(0, 3))
    assert_allclose(
        imp,
        actual_imp,
    )
