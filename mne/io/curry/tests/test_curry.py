#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import numpy as np
import pytest
from numpy import empty
from numpy.testing import assert_allclose, assert_array_equal

from mne.annotations import events_from_annotations, read_annotations
from mne.datasets import testing
from mne.event import find_events
from mne.io.curry import read_impedances_curry, read_raw_curry
from mne.io.curry.curry import _check_curry_filename, _check_curry_header_filename
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
    ],
)
def test_read_raw_curry_preload_equal(fname):
    """Test raw identity with preload=True/False."""
    with pytest.raises(RuntimeWarning):  # TODO change way to add montage in curry.py!
        raw1 = read_raw_curry(fname, preload=False)
        raw1.load_data()
        assert raw1 == read_raw_curry(fname, preload=True)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname",
    [
        pytest.param(curry7_bdf_file, id="curry 7"),
        pytest.param(curry8_bdf_file, id="curry 8"),
    ],
)
def test_read_events_curry_are_same_as_bdf(fname):
    """Test events from curry annotations recovers the right events."""
    EVENT_ID = {str(ii): ii for ii in range(5)}
    REF_EVENTS = find_events(read_raw_bdf(bdf_file, preload=True))

    with pytest.raises(RuntimeWarning):  # TODO change way to add montage in curry.py!
        raw = read_raw_curry(fname)
        events, _ = events_from_annotations(raw, event_id=EVENT_ID)
        assert_allclose(events, REF_EVENTS)
        assert raw.info["dev_head_t"] is None


@testing.requires_testing_data
def test_check_missing_files():
    """Test checking for missing curry files (smoke test)."""
    invalid_fname = "/invalid/path/name.xy"

    with pytest.raises(FileNotFoundError, match="no curry data file"):
        _check_curry_filename(invalid_fname)

    with pytest.raises(FileNotFoundError, match="no corresponding header"):
        _check_curry_header_filename(invalid_fname)

    with pytest.raises(FileNotFoundError, match="no curry data file"):
        read_raw_curry(invalid_fname)

    with pytest.raises(FileNotFoundError, match="no curry data file"):
        read_impedances_curry(invalid_fname)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname",
    [
        pytest.param(curry_dir / "test_bdf_stim_channel Curry 7.cef", id="7"),
        pytest.param(
            curry_dir / "test_bdf_stim_channel Curry 7 ASCII.cef", id="7 ascii"
        ),
    ],
)
def test_read_curry_annotations(fname):
    """Test reading for Curry events file."""
    EXPECTED_ONSET = [
        0.484,
        0.486,
        0.62,
        0.622,
        1.904,
        1.906,
        3.212,
        3.214,
        4.498,
        4.5,
        5.8,
        5.802,
        7.074,
        7.076,
        8.324,
        8.326,
        9.58,
        9.582,
    ]
    EXPECTED_DURATION = np.zeros_like(EXPECTED_ONSET)
    EXPECTED_DESCRIPTION = [
        "4",
        "50000",
        "2",
        "50000",
        "1",
        "50000",
        "1",
        "50000",
        "1",
        "50000",
        "1",
        "50000",
        "1",
        "50000",
        "1",
        "50000",
        "1",
        "50000",
    ]

    annot = read_annotations(fname, sfreq="auto")

    assert annot.orig_time is None
    assert_array_equal(annot.onset, EXPECTED_ONSET)
    assert_array_equal(annot.duration, EXPECTED_DURATION)
    assert_array_equal(annot.description, EXPECTED_DESCRIPTION)

    with pytest.raises(ValueError, match="must be numeric or 'auto'"):
        _ = read_annotations(fname, sfreq="nonsense")
    with pytest.warns(RuntimeWarning, match="does not match freq from fileheader"):
        _ = read_annotations(fname, sfreq=12.0)


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
