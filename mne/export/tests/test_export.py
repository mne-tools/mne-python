"""Test exporting functions."""
# Authors: MNE Developers
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from contextlib import nullcontext
from datetime import datetime, timezone
from os import remove
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

from mne import (
    Annotations,
    Epochs,
    create_info,
    read_epochs_eeglab,
    read_evokeds,
    read_evokeds_mff,
)
from mne.datasets import misc, testing
from mne.export import export_evokeds, export_evokeds_mff
from mne.fixes import _compare_version
from mne.io import (
    RawArray,
    read_raw_brainvision,
    read_raw_edf,
    read_raw_eeglab,
    read_raw_fif,
)
from mne.tests.test_epochs import _get_data
from mne.utils import (
    _check_edfio_installed,
    _record_warnings,
    _resource_path,
    object_diff,
)

fname_evoked = _resource_path("mne.io.tests.data", "test-ave.fif")
fname_raw = _resource_path("mne.io.tests.data", "test_raw.fif")

data_path = testing.data_path(download=False)
egi_evoked_fname = data_path / "EGI" / "test_egi_evoked.mff"
misc_path = misc.data_path(download=False)


@pytest.mark.parametrize(
    ["meas_date", "orig_time", "ext"],
    [
        [None, None, ".vhdr"],
        [datetime(2022, 12, 3, 19, 1, 10, 720100, tzinfo=timezone.utc), None, ".eeg"],
    ],
)
def test_export_raw_pybv(tmp_path, meas_date, orig_time, ext):
    """Test saving a Raw instance to BrainVision format via pybv."""
    pytest.importorskip("pybv")
    raw = read_raw_fif(fname_raw, preload=True)
    raw.apply_proj()

    raw.set_meas_date(meas_date)

    # add some annotations
    annots = Annotations(
        onset=[3, 6, 9, 12, 14],  # seconds
        duration=[1, 1, 0.5, 0.25, 9],  # seconds
        description=[
            "Stimulus/S  1",
            "Stimulus/S2.50",
            "Response/R101",
            "Look at this",
            "Comment/And at this",
        ],
        ch_names=[(), (), (), ("EEG 001",), ("EEG 001", "EEG 002")],
        orig_time=orig_time,
    )
    raw.set_annotations(annots)

    temp_fname = tmp_path / ("test" + ext)
    with pytest.warns(RuntimeWarning, match="'short' format. Converting"):
        raw.export(temp_fname)
    raw_read = read_raw_brainvision(str(temp_fname).replace(".eeg", ".vhdr"))
    assert raw.ch_names == raw_read.ch_names
    assert_allclose(raw.times, raw_read.times)
    assert_allclose(raw.get_data(), raw_read.get_data())


def test_export_raw_eeglab(tmp_path):
    """Test saving a Raw instance to EEGLAB's set format."""
    pytest.importorskip("eeglabio")
    raw = read_raw_fif(fname_raw, preload=True)
    raw.apply_proj()
    temp_fname = tmp_path / "test.set"
    raw.export(temp_fname)
    raw.drop_channels([ch for ch in ["epoc"] if ch in raw.ch_names])

    with pytest.warns(RuntimeWarning, match="is above the 99th percentile"):
        raw_read = read_raw_eeglab(temp_fname, preload=True, montage_units="m")
    assert raw.ch_names == raw_read.ch_names

    cart_coords = np.array([d["loc"][:3] for d in raw.info["chs"]])  # just xyz
    cart_coords_read = np.array([d["loc"][:3] for d in raw_read.info["chs"]])
    assert_allclose(cart_coords, cart_coords_read)
    assert_allclose(raw.times, raw_read.times)
    assert_allclose(raw.get_data(), raw_read.get_data())

    # test overwrite
    with pytest.raises(FileExistsError, match="Destination file exists"):
        raw.export(temp_fname, overwrite=False)
    raw.export(temp_fname, overwrite=True)

    # test pathlib.Path files
    raw.export(Path(temp_fname), overwrite=True)

    # test warning with unapplied projectors
    raw = read_raw_fif(fname_raw, preload=True)
    with pytest.warns(RuntimeWarning, match="Raw instance has unapplied projectors."):
        raw.export(temp_fname, overwrite=True)


def _create_raw_for_edf_tests(stim_channel_index=None):
    rng = np.random.RandomState(12345)
    ch_types = [
        "eeg",
        "eeg",
        "ecog",
        "ecog",
        "seeg",
        "eog",
        "ecg",
        "emg",
        "dbs",
        "bio",
    ]
    if stim_channel_index is not None:
        ch_types.insert(stim_channel_index, "stim")
    ch_names = np.arange(len(ch_types)).astype(str).tolist()
    info = create_info(ch_names, sfreq=1000, ch_types=ch_types)
    data = rng.random(size=(len(ch_names), 2000)) * 1e-5
    return RawArray(data, info)


@pytest.mark.skipif(
    not _check_edfio_installed(strict=False), reason="edfio not installed"
)
def test_double_export_edf(tmp_path):
    """Test exporting an EDF file multiple times."""
    raw = _create_raw_for_edf_tests(stim_channel_index=2)
    raw.info.set_meas_date("2023-09-04 14:53:09.000")

    # include subject info and measurement date
    raw.info["subject_info"] = dict(
        his_id="12345",
        first_name="mne",
        last_name="python",
        birthday=(1992, 1, 20),
        sex=1,
        weight=78.3,
        height=1.75,
        hand=3,
    )

    # export once
    temp_fname = tmp_path / "test.edf"
    with pytest.warns(RuntimeWarning, match="Exporting STIM channels"):
        raw.export(temp_fname, add_ch_type=True)
    raw_read = read_raw_edf(temp_fname, infer_types=True, preload=True)

    # export again
    raw_read.export(temp_fname, add_ch_type=True, overwrite=True)
    raw_read = read_raw_edf(temp_fname, infer_types=True, preload=True)

    # stim channel should be dropped
    raw.drop_channels("2")

    assert raw.ch_names == raw_read.ch_names
    assert_array_almost_equal(raw.get_data(), raw_read.get_data(), decimal=10)
    assert_array_equal(raw.times, raw_read.times)

    # check info
    for key in set(raw.info) - {"chs"}:
        assert raw.info[key] == raw_read.info[key]

    orig_ch_types = raw.get_channel_types()
    read_ch_types = raw_read.get_channel_types()
    assert_array_equal(orig_ch_types, read_ch_types)

    # check handling of missing subject metadata
    del raw.info["subject_info"]["sex"]
    raw.export(temp_fname, add_ch_type=True, overwrite=True)


@pytest.mark.skipif(
    not _check_edfio_installed(strict=False), reason="edfio not installed"
)
def test_export_edf_annotations(tmp_path):
    """Test that exporting EDF preserves annotations."""
    raw = _create_raw_for_edf_tests()
    annotations = Annotations(
        onset=[0.01, 0.05, 0.90, 1.05],
        duration=[0, 1, 0, 0],
        description=["test1", "test2", "test3", "test4"],
        ch_names=[["0"], ["0", "1"], [], ["1"]],
    )
    raw.set_annotations(annotations)

    # export
    temp_fname = tmp_path / "test.edf"
    raw.export(temp_fname)

    # read in the file
    raw_read = read_raw_edf(temp_fname, preload=True)
    assert_array_equal(raw.annotations.onset, raw_read.annotations.onset)
    assert_array_equal(raw.annotations.duration, raw_read.annotations.duration)
    assert_array_equal(raw.annotations.description, raw_read.annotations.description)
    assert_array_equal(raw.annotations.ch_names, raw_read.annotations.ch_names)


@pytest.mark.skipif(
    not _check_edfio_installed(strict=False), reason="edfio not installed"
)
def test_rawarray_edf(tmp_path):
    """Test saving a Raw array with integer sfreq to EDF."""
    raw = _create_raw_for_edf_tests()

    # include subject info and measurement date
    raw.info["subject_info"] = dict(
        first_name="mne",
        last_name="python",
        birthday=(1992, 1, 20),
        sex=1,
        hand=3,
    )
    time_now = datetime.now()
    meas_date = datetime(
        year=time_now.year,
        month=time_now.month,
        day=time_now.day,
        hour=time_now.hour,
        minute=time_now.minute,
        second=time_now.second,
        tzinfo=timezone.utc,
    )
    raw.set_meas_date(meas_date)
    temp_fname = tmp_path / "test.edf"

    raw.export(temp_fname, add_ch_type=True)
    raw_read = read_raw_edf(temp_fname, infer_types=True, preload=True)

    assert raw.ch_names == raw_read.ch_names
    assert_array_almost_equal(raw.get_data(), raw_read.get_data(), decimal=10)
    assert_array_equal(raw.times, raw_read.times)

    orig_ch_types = raw.get_channel_types()
    read_ch_types = raw_read.get_channel_types()
    assert_array_equal(orig_ch_types, read_ch_types)
    assert raw.info["meas_date"] == raw_read.info["meas_date"]

    # channel name can't be longer than 16 characters with the type added
    raw_bad = raw.copy()
    raw_bad.rename_channels({"1": "abcdefghijklmnopqrstuvwxyz"})
    with pytest.raises(RuntimeError, match="Signal label"), pytest.warns(
        RuntimeWarning, match="Data has a non-integer"
    ):
        raw_bad.export(temp_fname, overwrite=True)

    # include bad measurement date that is non-EDF compliant
    raw = _create_raw_for_edf_tests()
    meas_date = datetime(year=1984, month=1, day=1, tzinfo=timezone.utc)
    raw.set_meas_date(meas_date)
    with pytest.raises(ValueError, match="EDF only allows dates from 1985 to 2084"):
        raw.export(temp_fname, overwrite=True)

    # test that warning is raised if there are non-voltage based channels
    raw = _create_raw_for_edf_tests()
    raw.set_channel_types({"9": "hbr"}, on_unit_change="ignore")
    with pytest.warns(RuntimeWarning, match="Non-voltage channels"):
        raw.export(temp_fname, overwrite=True)

    # data should match up to the non-accepted channel
    raw_read = read_raw_edf(temp_fname, preload=True)
    assert raw.ch_names == raw_read.ch_names
    assert_array_almost_equal(raw.get_data()[:-1], raw_read.get_data()[:-1], decimal=10)
    assert_array_equal(raw.times, raw_read.times)


@pytest.mark.skipif(
    not _check_edfio_installed(strict=False), reason="edfio not installed"
)
@pytest.mark.parametrize(
    "dataset",
    [
        "test",
        pytest.param("misc", marks=[pytest.mark.slowtest, misc._pytest_mark()]),
    ],
)
def test_export_raw_edf(tmp_path, dataset):
    """Test saving a Raw instance to EDF format."""
    if dataset == "test":
        raw = read_raw_fif(fname_raw)
    elif dataset == "misc":
        fname = misc_path / "ecog" / "sample_ecog_ieeg.fif"
        raw = read_raw_fif(fname)

    # only test with EEG channels
    raw.pick(picks=["eeg", "ecog", "seeg"]).load_data()
    orig_ch_names = raw.ch_names
    temp_fname = tmp_path / "test.edf"

    # test runtime errors
    with pytest.warns() as record:
        raw.export(temp_fname, physical_range=(-1e6, 0))
    if dataset == "test":
        assert any("Data has a non-integer" in str(rec.message) for rec in record)
    assert any("The maximum" in str(rec.message) for rec in record)
    remove(temp_fname)

    with pytest.warns() as record:
        raw.export(temp_fname, physical_range=(0, 1e6))
    if dataset == "test":
        assert any("Data has a non-integer" in str(rec.message) for rec in record)
    assert any("The minimum" in str(rec.message) for rec in record)
    remove(temp_fname)

    if dataset == "test":
        with pytest.warns(RuntimeWarning, match="Data has a non-integer"):
            raw.export(temp_fname)
    elif dataset == "misc":
        with pytest.warns(RuntimeWarning, match="EDF format requires"):
            raw.export(temp_fname)

    if "epoc" in raw.ch_names:
        raw.drop_channels(["epoc"])

    raw_read = read_raw_edf(temp_fname, preload=True)
    assert orig_ch_names == raw_read.ch_names
    # only compare the original length, since extra zeros are appended
    orig_raw_len = len(raw)

    # assert data and times are not different
    # Due to the physical range of the data, reading and writing is
    # not lossless. For example, a physical min/max of -/+ 3200 uV
    # will result in a resolution of 0.09 uV. This resolution
    # though is acceptable for most EEG manufacturers.
    assert_array_almost_equal(
        raw.get_data(), raw_read.get_data()[:, :orig_raw_len], decimal=8
    )

    # Due to the data record duration limitations of EDF files, one
    # cannot store arbitrary float sampling rate exactly. Usually this
    # results in two sampling rates that are off by very low number of
    # decimal points. This for practical purposes does not matter
    # but will result in an error when say the number of time points
    # is very very large.
    assert_allclose(raw.times, raw_read.times[:orig_raw_len], rtol=0, atol=1e-5)


@pytest.mark.skipif(
    not _check_edfio_installed(strict=False), reason="edfio not installed"
)
def test_export_raw_edf_does_not_fail_on_empty_header_fields(tmp_path):
    """Test writing a Raw instance with empty header fields to EDF."""
    rng = np.random.RandomState(123456)

    ch_types = ["eeg"]
    info = create_info(len(ch_types), sfreq=1000, ch_types=ch_types)
    info["subject_info"] = {
        "his_id": "",
        "first_name": "",
        "middle_name": "",
        "last_name": "",
    }
    info["device_info"] = {"type": "123"}

    data = rng.random(size=(len(ch_types), 1000)) * 1e-5
    raw = RawArray(data, info)

    raw.export(tmp_path / "test.edf", add_ch_type=True)


@pytest.mark.xfail(reason="eeglabio (usage?) bugs that should be fixed")
@pytest.mark.parametrize("preload", (True, False))
def test_export_epochs_eeglab(tmp_path, preload):
    """Test saving an Epochs instance to EEGLAB's set format."""
    eeglabio = pytest.importorskip("eeglabio")
    raw, events = _get_data()[:2]
    raw.load_data()
    epochs = Epochs(raw, events, preload=preload)
    temp_fname = tmp_path / "test.set"
    # TODO: eeglabio 0.2 warns about invalid events
    if _compare_version(eeglabio.__version__, "==", "0.0.2-1"):
        ctx = _record_warnings
    else:
        ctx = nullcontext
    with ctx():
        epochs.export(temp_fname)
    epochs.drop_channels([ch for ch in ["epoc", "STI 014"] if ch in epochs.ch_names])
    epochs_read = read_epochs_eeglab(temp_fname)
    assert epochs.ch_names == epochs_read.ch_names
    cart_coords = np.array([d["loc"][:3] for d in epochs.info["chs"]])  # just xyz
    cart_coords_read = np.array([d["loc"][:3] for d in epochs_read.info["chs"]])
    assert_allclose(cart_coords, cart_coords_read)
    assert_array_equal(epochs.events[:, 0], epochs_read.events[:, 0])  # latency
    assert epochs.event_id.keys() == epochs_read.event_id.keys()  # just keys
    assert_allclose(epochs.times, epochs_read.times)
    assert_allclose(epochs.get_data(), epochs_read.get_data())

    # test overwrite
    with pytest.raises(FileExistsError, match="Destination file exists"):
        epochs.export(temp_fname, overwrite=False)
    with ctx():
        epochs.export(temp_fname, overwrite=True)

    # test pathlib.Path files
    with ctx():
        epochs.export(Path(temp_fname), overwrite=True)

    # test warning with unapplied projectors
    epochs = Epochs(raw, events, preload=preload, proj=False)
    with pytest.warns(
        RuntimeWarning, match="Epochs instance has unapplied projectors."
    ):
        epochs.export(Path(temp_fname), overwrite=True)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@testing.requires_testing_data
@pytest.mark.parametrize("fmt", ("auto", "mff"))
@pytest.mark.parametrize("do_history", (True, False))
def test_export_evokeds_to_mff(tmp_path, fmt, do_history):
    """Test exporting evoked dataset to MFF."""
    pytest.importorskip("mffpy", "0.5.7")
    evoked = read_evokeds_mff(egi_evoked_fname)
    export_fname = tmp_path / "evoked.mff"
    history = [
        {
            "name": "Test Segmentation",
            "method": "Segmentation",
            "settings": ["Setting 1", "Setting 2"],
            "results": ["Result 1", "Result 2"],
        },
        {
            "name": "Test Averaging",
            "method": "Averaging",
            "settings": ["Setting 1", "Setting 2"],
            "results": ["Result 1", "Result 2"],
        },
    ]
    if do_history:
        export_evokeds_mff(export_fname, evoked, history=history)
    else:
        export_evokeds(export_fname, evoked, fmt=fmt)
    # Drop non-EEG channels
    evoked = [ave.drop_channels(["ECG", "EMG"]) for ave in evoked]
    evoked_exported = read_evokeds_mff(export_fname)
    assert len(evoked) == len(evoked_exported)
    for ave, ave_exported in zip(evoked, evoked_exported):
        # Compare infos
        assert object_diff(ave_exported.info, ave.info) == ""
        # Compare data
        assert_allclose(ave_exported.data, ave.data)
        # Compare properties
        assert ave_exported.nave == ave.nave
        assert ave_exported.kind == ave.kind
        assert ave_exported.comment == ave.comment
        assert_allclose(ave_exported.times, ave.times)

    # test overwrite
    with pytest.raises(FileExistsError, match="Destination file exists"):
        if do_history:
            export_evokeds_mff(export_fname, evoked, history=history, overwrite=False)
        else:
            export_evokeds(export_fname, evoked, overwrite=False)

    if do_history:
        export_evokeds_mff(export_fname, evoked, history=history, overwrite=True)
    else:
        export_evokeds(export_fname, evoked, overwrite=True)

    # test export from evoked directly
    evoked[0].export(export_fname, overwrite=True)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@testing.requires_testing_data
def test_export_to_mff_no_device():
    """Test no device type throws ValueError."""
    pytest.importorskip("mffpy", "0.5.7")
    evoked = read_evokeds_mff(egi_evoked_fname, condition="Category 1")
    evoked.info["device_info"] = None
    with pytest.raises(ValueError, match="No device type."):
        export_evokeds("output.mff", evoked)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_export_to_mff_incompatible_sfreq():
    """Test non-whole number sampling frequency throws ValueError."""
    pytest.importorskip("mffpy", "0.5.7")
    evoked = read_evokeds(fname_evoked)
    with pytest.raises(ValueError, match=f'sfreq: {evoked[0].info["sfreq"]}'):
        export_evokeds("output.mff", evoked)


@pytest.mark.parametrize(
    "fmt,ext",
    [("EEGLAB", "set"), ("EDF", "edf"), ("BrainVision", "vhdr"), ("auto", "vhdr")],
)
def test_export_evokeds_unsupported_format(fmt, ext):
    """Test exporting evoked dataset to non-supported formats."""
    evoked = read_evokeds(fname_evoked)
    errstr = fmt.lower() if fmt != "auto" else "vhdr"
    with pytest.raises(ValueError, match=f"Format '{errstr}' is not .*"):
        export_evokeds(f"output.{ext}", evoked, fmt=fmt)
