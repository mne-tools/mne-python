# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)
from scipy.io import loadmat

from mne import Annotations, pick_types
from mne._fiff.pick import channel_indices_by_type, get_channel_type_constants
from mne.annotations import _ndarray_ch_names, events_from_annotations, read_annotations
from mne.datasets import testing
from mne.io import edf, read_raw_bdf, read_raw_edf, read_raw_fif, read_raw_gdf
from mne.io.edf.edf import (
    _edf_str,
    _parse_prefilter_string,
    _prefilter_float,
    _read_annotations_edf,
    _read_ch,
    _read_edf_header,
    _read_header,
    _set_prefilter,
)
from mne.io.tests.test_raw import _test_raw_reader
from mne.tests.test_annotations import _assert_annotations_equal
from mne.utils import _record_warnings

td_mark = testing._pytest_mark()

data_dir = Path(__file__).parent / "data"
montage_path = data_dir / "biosemi.hpts"  # XXX: missing reader
bdf_path = data_dir / "test.bdf"
edf_path = data_dir / "test.edf"
duplicate_channel_labels_path = data_dir / "duplicate_channel_labels.edf"
edf_uneven_path = data_dir / "test_uneven_samp.edf"
bdf_eeglab_path = data_dir / "test_bdf_eeglab.mat"
edf_stim_channel_path = data_dir / "test_edf_stim_channel.edf"
edf_txt_stim_channel_path = data_dir / "test_edf_stim_channel.txt"

data_path = testing.data_path(download=False)
edf_stim_resamp_path = data_path / "EDF" / "test_edf_stim_resamp.edf"
edf_overlap_annot_path = data_path / "EDF" / "test_edf_overlapping_annotations.edf"
edf_reduced = data_path / "EDF" / "test_reduced.edf"
edf_annot_only = data_path / "EDF" / "SC4001EC-Hypnogram.edf"
bdf_stim_channel_path = data_path / "BDF" / "test_bdf_stim_channel.bdf"
bdf_multiple_annotations_path = data_path / "BDF" / "multiple_annotation_chans.bdf"
test_generator_bdf = data_path / "BDF" / "test_generator_2.bdf"
test_generator_edf = data_path / "EDF" / "test_generator_2.edf"
edf_annot_sub_s_path = data_path / "EDF" / "subsecond_starttime.edf"
edf_chtypes_path = data_path / "EDF" / "chtypes_edf.edf"
edf_utf8_annotations = data_path / "EDF" / "test_utf8_annotations.edf"

eog = ["REOG", "LEOG", "IEOG"]
misc = ["EXG1", "EXG5", "EXG8", "M1", "M2"]


def test_orig_units():
    """Test exposure of original channel units."""
    raw = read_raw_edf(edf_path, preload=True)

    # Test original units
    orig_units = raw._orig_units
    assert len(orig_units) == len(raw.ch_names)
    assert orig_units["A1"] == "µV"  # formerly 'uV' edit by _check_orig_units
    del orig_units

    raw.rename_channels(dict(A1="AA"))
    assert raw._orig_units["AA"] == "µV"
    raw.rename_channels(dict(AA="A1"))

    raw_back = raw.copy().pick(raw.ch_names[:1])  # _pick_drop_channels
    assert raw_back.ch_names == ["A1"]
    assert set(raw_back._orig_units) == {"A1"}
    raw_back.add_channels([raw.copy().pick(raw.ch_names[1:])])
    assert raw_back.ch_names == raw.ch_names
    assert set(raw_back._orig_units) == set(raw.ch_names)
    raw_back.reorder_channels(raw.ch_names[::-1])
    assert set(raw_back._orig_units) == set(raw.ch_names)


def test_units_params():
    """Test enforcing original channel units."""
    with pytest.raises(
        ValueError, match=r"Unit for channel .* is present .* cannot overwrite it"
    ):
        _ = read_raw_edf(edf_path, units="V", preload=True)


def test_edf_temperature(monkeypatch):
    """Test that we can parse temperature channel type."""
    raw = read_raw_edf(edf_path)
    assert raw.get_channel_types()[0] == "eeg"

    def _first_chan_temp(*args, **kwargs):
        out, orig_units = _read_edf_header(*args, **kwargs)
        out["ch_types"][0] = "TEMP"
        return out, orig_units

    monkeypatch.setattr(edf.edf, "_read_edf_header", _first_chan_temp)
    raw = read_raw_edf(edf_path)
    assert "temperature" in raw
    assert raw.get_channel_types()[0] == "temperature"


@testing.requires_testing_data
def test_subject_info(tmp_path):
    """Test exposure of original channel units."""
    raw = read_raw_edf(edf_stim_resamp_path, preload=True)

    # check subject_info from `info`
    assert raw.info["subject_info"] is not None
    want = {
        "his_id": "X",
        "sex": 1,
        "birthday": datetime.date(1967, 10, 9),
        "last_name": "X",
    }
    for key, val in want.items():
        assert raw.info["subject_info"][key] == val, key

    # add information
    raw.info["subject_info"]["hand"] = 0

    # save raw to FIF and load it back
    fname = tmp_path / "test_raw.fif"
    raw.save(fname)
    raw = read_raw_fif(fname)

    # check subject_info from `info`
    assert raw.info["subject_info"] is not None
    want = {
        "his_id": "X",
        "sex": 1,
        "birthday": datetime.date(1967, 10, 9),
        "last_name": "X",
        "hand": 0,
    }
    for key, val in want.items():
        assert raw.info["subject_info"][key] == val


def test_bdf_data():
    """Test reading raw bdf files."""
    # XXX BDF data for these is around 0.01 when it should be in the uV range,
    # probably some bug
    test_scaling = False
    raw_py = _test_raw_reader(
        read_raw_bdf,
        input_fname=bdf_path,
        eog=eog,
        misc=misc,
        exclude=["M2", "IEOG"],
        test_scaling=test_scaling,
    )
    assert len(raw_py.ch_names) == 71
    raw_py = _test_raw_reader(
        read_raw_bdf,
        input_fname=bdf_path,
        montage="biosemi64",
        eog=eog,
        misc=misc,
        exclude=["M2", "IEOG"],
        test_scaling=test_scaling,
    )
    assert len(raw_py.ch_names) == 71
    assert "RawEDF" in repr(raw_py)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude="bads")
    data_py, _ = raw_py[picks]

    # this .mat was generated using the EEG Lab Biosemi Reader
    raw_eeglab = loadmat(bdf_eeglab_path)
    raw_eeglab = raw_eeglab["data"] * 1e-6  # data are stored in microvolts
    data_eeglab = raw_eeglab[picks]
    # bdf saved as a single, resolution to seven decimal points in matlab
    assert_array_almost_equal(data_py, data_eeglab, 8)

    # Manually checking that float coordinates are imported
    assert (raw_py.info["chs"][0]["loc"]).any()
    assert (raw_py.info["chs"][25]["loc"]).any()
    assert (raw_py.info["chs"][63]["loc"]).any()


@testing.requires_testing_data
def test_bdf_crop_save_stim_channel(tmp_path):
    """Test EDF with various sampling rates."""
    raw = read_raw_bdf(bdf_stim_channel_path)
    raw.save(tmp_path / "test-raw.fif", tmin=1.2, tmax=4.0, overwrite=True)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname",
    [
        edf_reduced,
        edf_overlap_annot_path,
    ],
)
@pytest.mark.parametrize("stim_channel", (None, False, "auto"))
def test_edf_others(fname, stim_channel):
    """Test EDF with various sampling rates and overlapping annotations."""
    _test_raw_reader(
        read_raw_edf,
        input_fname=fname,
        stim_channel=stim_channel,
        verbose="error",
        test_preloading=False,
        preload=True,  # no preload=False for mixed sfreqs
    )


@testing.requires_testing_data
@pytest.mark.parametrize("stim_channel", (None, False, "auto"))
def test_edf_different_sfreqs(stim_channel):
    """Test EDF with various sampling rates."""
    rng = np.random.RandomState(0)
    # load with and without preloading, should produce the same results
    raw1 = read_raw_edf(
        input_fname=edf_reduced,
        stim_channel=stim_channel,
        verbose="error",
        preload=False,
    )
    raw2 = read_raw_edf(
        input_fname=edf_reduced,
        stim_channel=stim_channel,
        verbose="error",
        preload=True,
    )

    picks = rng.permutation(np.arange(len(raw1.ch_names) - 1))[:10]
    data1, times1 = raw1[picks, :]
    data2, times2 = raw2[picks, :]
    assert_allclose(data1, data2, err_msg="Data mismatch with preload")
    assert_allclose(times1, times2)

    # loading slices should throw a warning as they have different
    # edge artifacts than when loading the entire file at once
    with pytest.warns(RuntimeWarning, match="mixed sampling frequencies"):
        data1, times1 = raw1[picks, :512]
    data2, times2 = raw2[picks, :512]

    # should NOT throw a warning when loading channels that have all the same
    # sampling frequency - here, no edge artifacts can appear
    picks = np.arange(15, 20)  # these channels all have 512 Hz
    data1, times1 = raw1[picks, :512]
    data2, times2 = raw2[picks, :512]
    assert_allclose(data1, data2, err_msg="Data mismatch with preload")
    assert_allclose(times1, times2)


@testing.requires_testing_data
@pytest.mark.parametrize("stim_channel", (None, False, "auto"))
def test_edf_different_sfreqs_nopreload(stim_channel):
    """Test loading smaller sfreq channels without preloading."""
    # load without preloading, then load a channel that has smaller sfreq
    # as other channels, produced an error, see mne-python/issues/12897

    for i in range(1, 13):
        raw = read_raw_edf(input_fname=edf_reduced, verbose="error", preload=False)

        # this should work for channels of all sfreq, even if larger sfreqs
        # are present in the file
        x1 = raw.get_data(picks=[f"A{i}"], return_times=False)
        # load next ch, this is sometimes with a higher sometimes a lower sfreq
        x2 = raw.get_data([f"A{i + 1}"], return_times=False)
        assert x1.shape == x2.shape


def test_edf_data_broken(tmp_path):
    """Test edf files."""
    raw = _test_raw_reader(
        read_raw_edf,
        input_fname=edf_path,
        exclude=["Ergo-Left", "H10"],
        verbose="error",
    )
    raw_py = read_raw_edf(edf_path)
    data = raw_py.get_data()
    assert_equal(len(raw.ch_names) + 2, len(raw_py.ch_names))

    # Test with number of records not in header (-1).
    broken_fname = tmp_path / "broken.edf"
    with open(edf_path, "rb") as fid_in:
        fid_in.seek(0, 2)
        n_bytes = fid_in.tell()
        fid_in.seek(0, 0)
        rbytes = fid_in.read()
    with open(broken_fname, "wb") as fid_out:
        fid_out.write(rbytes[:236])
        fid_out.write(b"-1      ")
        fid_out.write(rbytes[244 : 244 + int(n_bytes * 0.4)])
    with pytest.warns(RuntimeWarning, match="records .* not match the file size"):
        raw = read_raw_edf(broken_fname, preload=True)
        read_raw_edf(broken_fname, exclude=raw.ch_names[:132], preload=True)

    # Test with \x00's in the data
    with open(broken_fname, "wb") as fid_out:
        fid_out.write(rbytes[:184])
        assert rbytes[184:192] == b"36096   "
        fid_out.write(rbytes[184:192].replace(b" ", b"\x00"))
        fid_out.write(rbytes[192:])
    raw_py = read_raw_edf(broken_fname)
    data_new = raw_py.get_data()
    assert_allclose(data, data_new)


def test_duplicate_channel_labels_edf():
    """Test reading edf file with duplicate channel names."""
    EXPECTED_CHANNEL_NAMES = ["EEG F1-Ref-0", "EEG F2-Ref", "EEG F1-Ref-1"]
    with pytest.warns(RuntimeWarning, match="Channel names are not unique"):
        raw = read_raw_edf(duplicate_channel_labels_path, preload=False)

    assert raw.ch_names == EXPECTED_CHANNEL_NAMES


def test_parse_annotation(tmp_path):
    """Test parsing the tal channel."""
    # test the parser
    annot = (
        b"+180\x14Lights off\x14Close door\x14\x00\x00\x00\x00\x00"
        b"+180\x14Lights off\x14\x00\x00\x00\x00\x00\x00\x00\x00"
        b"+180\x14Close door\x14\x00\x00\x00\x00\x00\x00\x00\x00"
        b"+3.14\x1504.20\x14nothing\x14\x00\x00\x00\x00"
        b"+1800.2\x1525.5\x14Apnea\x14\x00\x00\x00\x00\x00\x00\x00"
        b"+123\x14\x14\x00\x00\x00\x00\x00\x00\x00"
    )
    annot_file = tmp_path / "annotations.txt"
    with open(annot_file, "wb") as f:
        f.write(annot)

    annot = [a for a in bytes(annot)]
    annot[1::2] = [a * 256 for a in annot[1::2]]
    tal_channel_A = np.array(
        list(map(sum, zip(annot[0::2], annot[1::2]))), dtype=np.int64
    )

    with open(annot_file, "rb") as fid:
        # ch_data = np.fromfile(fid, dtype='<i2', count=len(annot))
        tal_channel_B = _read_ch(
            fid,
            subtype="EDF",
            dtype="<i2",
            samp=(len(annot) - 1) // 2,
            dtype_byte="This_parameter_is_not_used",
        )

    want_onset, want_duration, want_description = zip(
        *[
            [3.14, 4.2, "nothing"],
            [180.0, 0.0, "Lights off"],
            [180.0, 0.0, "Close door"],
            [180.0, 0.0, "Lights off"],
            [180.0, 0.0, "Close door"],
            [1800.2, 25.5, "Apnea"],
        ]
    )
    for tal_channel in [tal_channel_A, tal_channel_B]:
        annotations = _read_annotations_edf([tal_channel])
        assert_allclose(annotations.onset, want_onset)
        assert_allclose(annotations.duration, want_duration)
        assert_array_equal(annotations.description, want_description)


def test_find_events_backward_compatibility():
    """Test if events are detected correctly in a typical MNE workflow."""
    EXPECTED_EVENTS = [[68, 0, 2], [199, 0, 2], [1024, 0, 3], [1280, 0, 2]]
    # test an actual file
    raw = read_raw_edf(edf_path, preload=True)
    event_id = {
        a: n for n, a in enumerate(sorted(set(raw.annotations.description)), start=1)
    }
    event_id.pop("start")
    events_from_EFA, _ = events_from_annotations(
        raw, event_id=event_id, use_rounding=False
    )

    assert_array_equal(events_from_EFA, EXPECTED_EVENTS)


@testing.requires_testing_data
def test_no_data_channels():
    """Test that we can load with no data channels."""
    # analog
    raw = read_raw_edf(edf_path, preload=True)
    picks = pick_types(raw.info, stim=True)
    assert list(picks) == [len(raw.ch_names) - 1]
    stim_data = raw[picks][0]
    raw = read_raw_edf(edf_path, exclude=raw.ch_names[:-1])
    stim_data_2 = raw[0][0]
    assert_array_equal(stim_data, stim_data_2)
    raw.plot()  # smoke test
    # annotations
    raw = read_raw_edf(edf_overlap_annot_path)
    picks = pick_types(raw.info, stim=True)
    assert picks.size == 0
    annot = raw.annotations
    raw = read_raw_edf(edf_overlap_annot_path, exclude=raw.ch_names)
    annot_2 = raw.annotations
    _assert_annotations_equal(annot, annot_2)
    # only annotations (should warn)
    with _record_warnings(), pytest.warns(RuntimeWarning, match="read_annotations"):
        read_raw_edf(edf_annot_only)


@pytest.mark.parametrize("fname", [edf_path, bdf_path])
def test_to_data_frame(fname):
    """Test EDF/BDF Raw Pandas exporter."""
    pytest.importorskip("pandas")
    ext = fname.suffix
    if ext == ".edf":
        raw = read_raw_edf(fname, preload=True, verbose="error")
    elif ext == ".bdf":
        raw = read_raw_bdf(fname, preload=True, verbose="error")
    _, times = raw[0, :10]
    df = raw.to_data_frame(index="time")
    assert (df.columns == raw.ch_names).all()
    assert_array_equal(times, df.index.values[:10])
    df = raw.to_data_frame(index=None, scalings={"eeg": 1e13})
    assert "time" in df.columns
    assert_array_equal(df.values[:, 1], raw._data[0] * 1e13)


def test_read_raw_edf_stim_channel_input_parameters():
    """Test edf raw reader stim channel kwarg changes."""
    read_raw_edf(edf_path)  # smoke test, no warnings
    for invalid_stim_parameter in ["EDF Annotations", "BDF Annotations"]:
        with pytest.raises(ValueError, match="stim channel is not supported"):
            read_raw_edf(edf_path, stim_channel=invalid_stim_parameter)


def test_read_annot(tmp_path):
    """Test parsing the tal channel."""
    EXPECTED_ANNOTATIONS = [
        [180.0, 0, "Lights off"],
        [180.0, 0, "Close door"],
        [180.0, 0, "Lights off"],
        [180.0, 0, "Close door"],
        [3.14, 4.2, "nothing"],
        [1800.2, 25.5, "Apnea"],
    ]

    EXPECTED_ONSET = [180.0, 180.0, 180.0, 180.0, 3.14, 1800.2]
    EXPECTED_DURATION = [0, 0, 0, 0, 4.2, 25.5]
    EXPECTED_DESC = [
        "Lights off",
        "Close door",
        "Lights off",
        "Close door",
        "nothing",
        "Apnea",
    ]
    EXPECTED_ANNOTATIONS = Annotations(
        onset=EXPECTED_ONSET,
        duration=EXPECTED_DURATION,
        description=EXPECTED_DESC,
        orig_time=None,
    )

    annot = (
        b"+180\x14Lights off\x14Close door\x14\x00\x00\x00\x00\x00"
        b"+180\x14Lights off\x14\x00\x00\x00\x00\x00\x00\x00\x00"
        b"+180\x14Close door\x14\x00\x00\x00\x00\x00\x00\x00\x00"
        b"+3.14\x1504.20\x14nothing\x14\x00\x00\x00\x00"
        b"+1800.2\x1525.5\x14Apnea\x14\x00\x00\x00\x00\x00\x00\x00"
        b"+123\x14\x14\x00\x00\x00\x00\x00\x00\x00"
    )
    annot_file = tmp_path / "annotations.txt"
    with open(annot_file, "wb") as f:
        f.write(annot)

    annotations = _read_annotations_edf(annotations=str(annot_file))
    _assert_annotations_equal(annotations, EXPECTED_ANNOTATIONS)

    # Now test when reading from buffer of data
    with open(annot_file, "rb") as fid:
        ch_data = np.fromfile(fid, dtype="<i2", count=len(annot))
    annotations = _read_annotations_edf([ch_data])
    _assert_annotations_equal(annotations, EXPECTED_ANNOTATIONS)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [test_generator_edf, test_generator_bdf])
def test_read_annotations(fname, recwarn):
    """Test IO of annotations from edf and bdf files via regexp."""
    annot = read_annotations(fname)
    assert len(annot.onset) == 2


@testing.requires_testing_data
def test_read_utf8_annotations():
    """Test if UTF8 annotations can be read."""
    raw = read_raw_edf(edf_utf8_annotations)
    assert raw.annotations[0]["description"] == "RECORD START"
    assert raw.annotations[1]["description"] == "仰卧"


def test_read_annotations_edf(tmp_path):
    """Test reading annotations from EDF file."""
    annot = (
        b"+1.1\x14Event A@@CH1\x14\x00\x00"
        b"+1.2\x14Event A\x14\x00\x00"
        b"+1.3\x14Event B@@CH1\x14\x00\x00"
        b"+1.3\x14Event B@@CH2\x14\x00\x00"
        b"+1.4\x14Event A@@CH3\x14\x00\x00"
        b"+1.5\x14Event B\x14\x00\x00"
    )
    annot_file = tmp_path / "annotations.edf"
    with open(annot_file, "wb") as f:
        f.write(annot)

    # Test reading annotations from channel data
    with open(annot_file, "rb") as f:
        tal_channel = _read_ch(
            f,
            subtype="EDF",
            dtype="<i2",
            samp=-1,
            dtype_byte=None,
        )

    # Read annotations without input channel names: annotations are left untouched and
    # assigned as global
    annotations = _read_annotations_edf(tal_channel, ch_names=None, encoding="latin1")
    assert_allclose(annotations.onset, [1.1, 1.2, 1.3, 1.3, 1.4, 1.5])
    assert not any(annotations.duration)  # all durations are 0
    assert_array_equal(
        annotations.description,
        [
            "Event A@@CH1",
            "Event A",
            "Event B@@CH1",
            "Event B@@CH2",
            "Event A@@CH3",
            "Event B",
        ],
    )
    assert_array_equal(
        annotations.ch_names, _ndarray_ch_names([(), (), (), (), (), ()])
    )

    # Read annotations with complete input channel names: each annotation is parsed and
    # associated to a channel
    annotations = _read_annotations_edf(
        tal_channel, ch_names=["CH1", "CH2", "CH3"], encoding="latin1"
    )
    assert_allclose(annotations.onset, [1.1, 1.2, 1.3, 1.4, 1.5])
    assert not any(annotations.duration)  # all durations are 0
    assert_array_equal(
        annotations.description, ["Event A", "Event A", "Event B", "Event A", "Event B"]
    )
    assert_array_equal(
        annotations.ch_names,
        _ndarray_ch_names([("CH1",), (), ("CH1", "CH2"), ("CH3",), ()]),
    )

    # Read annotations with incomplete input channel names: "CH3" is missing from input
    # channels, turning the related annotation into a global one
    annotations = _read_annotations_edf(
        tal_channel, ch_names=["CH1", "CH2"], encoding="latin1"
    )
    assert_allclose(annotations.onset, [1.1, 1.2, 1.3, 1.4, 1.5])
    assert not any(annotations.duration)  # all durations are 0
    assert_array_equal(
        annotations.description,
        ["Event A", "Event A", "Event B", "Event A@@CH3", "Event B"],
    )
    assert_array_equal(
        annotations.ch_names, _ndarray_ch_names([("CH1",), (), ("CH1", "CH2"), (), ()])
    )


def test_read_latin1_annotations(tmp_path):
    """Test if annotations encoded as Latin-1 can be read.

    Note that the correct encoding according to the EDF+ standard should be
    UTF8, but many real-world files are saved with the Latin-1 encoding.
    """
    annot = (
        b"+1.1\x14\xe9\x14\x00\x00"  # +1.1 é
        b"+1.2\x14\xe0\x14\x00\x00"  # +1.2 à
        b"+1.3\x14\xe8\x14\x00\x00"  # +1.3 è
        b"+1.4\x14\xf9\x14\x00\x00"  # +1.4 ù
        b"+1.5\x14\xe2\x14\x00\x00"  # +1.5 â
        b"+1.6\x14\xea\x14\x00\x00"  # +1.6 ê
        b"+1.7\x14\xee\x14\x00\x00"  # +1.7 î
        b"+1.8\x14\xf4\x14\x00\x00"  # +1.8 ô
        b"+1.9\x14\xfb\x14\x00\x00"  # +1.9 û
    )
    annot_file = tmp_path / "annotations.edf"
    with open(annot_file, "wb") as f:
        f.write(annot)

    # Test reading directly from file
    annotations = read_annotations(fname=annot_file, encoding="latin1")
    assert_allclose(annotations.onset, [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    assert not any(annotations.duration)  # all durations are 0
    assert_array_equal(
        annotations.description, ["é", "à", "è", "ù", "â", "ê", "î", "ô", "û"]
    )

    # Test reading annotations from channel data
    with open(annot_file, "rb") as f:
        tal_channel = _read_ch(
            f,
            subtype="EDF",
            dtype="<i2",
            samp=-1,
            dtype_byte=None,
        )
    annotations = _read_annotations_edf(tal_channel, encoding="latin1")
    assert_allclose(annotations.onset, [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    assert not any(annotations.duration)  # all durations are 0
    assert_array_equal(
        annotations.description, ["é", "à", "è", "ù", "â", "ê", "î", "ô", "û"]
    )

    with pytest.raises(Exception, match="Encountered invalid byte in"):
        _read_annotations_edf(tal_channel)  # default encoding="utf8" fails
    with pytest.raises(Exception, match="'utf-8' codec can't decode.*"):
        _read_annotations_edf(str(annot_file))  # default encoding="utf8" fails


@pytest.mark.parametrize(
    "prefiltering, hp, lp",
    [
        pytest.param(["HP: 1Hz LP: 30Hz"], ["1"], ["30"], id="basic edf"),
        pytest.param(["LP: 30Hz HP: 1Hz"], ["1"], ["30"], id="reversed order"),
        pytest.param(["HP: 1 LP: 30"], ["1"], ["30"], id="w/o Hz"),
        pytest.param(["HP: 0,1 LP: 30,5"], ["0.1"], ["30.5"], id="using comma"),
        pytest.param(
            ["HP:0.1Hz LP:75Hz N:50Hz"], ["0.1"], ["75"], id="with notch filter"
        ),
        pytest.param([""], [""], [""], id="empty string"),
        pytest.param(["HP: DC; LP: 410"], ["DC"], ["410"], id="bdf_dc"),
        pytest.param(
            ["", "HP:0.1Hz LP:75Hz N:50Hz", ""],
            ["", "0.1", ""],
            ["", "75", ""],
            id="multi-ch",
        ),
    ],
)
def test_edf_parse_prefilter_string(prefiltering, hp, lp):
    """Test prefilter strings from header are parsed correctly."""
    highpass, lowpass = _parse_prefilter_string(prefiltering)
    assert_array_equal(highpass, hp)
    assert_array_equal(lowpass, lp)


@pytest.mark.parametrize(
    "prefilter_string, expected",
    [
        ("0", 0),
        ("1.1", 1.1),
        ("DC", 0),
        ("", np.nan),
        ("1.1.1", np.nan),
        (1.1, 1.1),
        (1, 1),
        (np.float32(1.1), np.float32(1.1)),
        (np.nan, np.nan),
    ],
)
def test_edf_prefilter_float(prefilter_string, expected):
    """Test to make float from prefilter string."""
    assert_equal(_prefilter_float(prefilter_string), expected)


@pytest.mark.parametrize(
    "edf_info, hp, lp, hp_warn, lp_warn",
    [
        ({"highpass": ["0"], "lowpass": ["1.1"]}, -1, 1.1, False, False),
        ({"highpass": [""], "lowpass": [""]}, -1, -1, False, False),
        ({"highpass": ["DC"], "lowpass": [""]}, -1, -1, False, False),
        ({"highpass": [1], "lowpass": [2]}, 1, 2, False, False),
        ({"highpass": [np.nan], "lowpass": [np.nan]}, -1, -1, False, False),
        ({"highpass": ["1", "2"], "lowpass": ["3", "4"]}, 2, 3, True, True),
        ({"highpass": [np.nan, 1], "lowpass": ["", 3]}, 1, 3, True, True),
        ({"highpass": [np.nan, np.nan], "lowpass": [1, 2]}, -1, 1, False, True),
        ({}, -1, -1, False, False),
    ],
)
def test_edf_set_prefilter(edf_info, hp, lp, hp_warn, lp_warn):
    """Test _set_prefilter function."""
    info = {"lowpass": -1, "highpass": -1}

    if hp_warn:
        ctx = pytest.warns(
            RuntimeWarning,
            match=(
                "Channels contain different highpass filters. "
                "Highest filter setting will be stored."
            ),
        )
    else:
        ctx = nullcontext()
    with ctx:
        _set_prefilter(
            info, edf_info, list(range(len(edf_info.get("highpass", [])))), "highpass"
        )

    if lp_warn:
        ctx = pytest.warns(
            RuntimeWarning,
            match=(
                "Channels contain different lowpass filters. "
                "Lowest filter setting will be stored."
            ),
        )
    else:
        ctx = nullcontext()
    with ctx:
        _set_prefilter(
            info, edf_info, list(range(len(edf_info.get("lowpass", [])))), "lowpass"
        )
    assert info["highpass"] == hp
    assert info["lowpass"] == lp


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [test_generator_edf, test_generator_bdf])
def test_load_generator(fname, recwarn):
    """Test IO of annotations from edf and bdf files with raw info."""
    if fname.suffix == ".edf":
        raw = read_raw_edf(fname)
    elif fname.suffix == ".bdf":
        raw = read_raw_bdf(fname)
    assert len(raw.annotations.onset) == 2
    found_types = [
        k for k, v in channel_indices_by_type(raw.info, picks=None).items() if v
    ]
    assert len(found_types) == 1
    events, event_id = events_from_annotations(raw)
    ch_names = [
        "squarewave",
        "ramp",
        "pulse",
        "ECG",
        "noise",
        "sine 1 Hz",
        "sine 8 Hz",
        "sine 8.5 Hz",
        "sine 15 Hz",
        "sine 17 Hz",
        "sine 50 Hz",
    ]
    assert raw.get_data().shape == (11, 120000)
    assert raw.ch_names == ch_names
    assert event_id == {"RECORD START": 2, "REC STOP": 1}
    assert_array_equal(events, [[0, 0, 2], [120000, 0, 1]])


@pytest.mark.parametrize(
    "EXPECTED, test_input",
    [
        pytest.param(
            {"stAtUs": "stim", "tRigGer": "stim", "sine 1 Hz": "eeg"}, "auto", id="auto"
        ),
        pytest.param(
            {"stAtUs": "eeg", "tRigGer": "eeg", "sine 1 Hz": "eeg"}, None, id="None"
        ),
        pytest.param(
            {"stAtUs": "eeg", "tRigGer": "eeg", "sine 1 Hz": "stim"},
            "sine 1 Hz",
            id="single string",
        ),
        pytest.param(
            {"stAtUs": "eeg", "tRigGer": "eeg", "sine 1 Hz": "stim"}, 2, id="single int"
        ),
        pytest.param(
            {"stAtUs": "eeg", "tRigGer": "eeg", "sine 1 Hz": "stim"},
            -1,
            id="single int (revers indexing)",
        ),
        pytest.param(
            {"stAtUs": "stim", "tRigGer": "stim", "sine 1 Hz": "eeg"},
            [0, 1],
            id="int list",
        ),
    ],
)
def test_edf_stim_ch_pick_up(test_input, EXPECTED):
    """Test stim_channel."""
    # This is fragile for EEG/EEG-CSD, so just omit csd
    KIND_DICT = get_channel_type_constants()
    TYPE_LUT = {
        v["kind"]: k for k, v in KIND_DICT.items() if k not in ("csd", "chpi")
    }  # chpi not needed, and unhashable (a list)
    fname = data_dir / "test_stim_channel.edf"

    raw = read_raw_edf(fname, stim_channel=test_input)
    ch_types = {ch["ch_name"]: TYPE_LUT[ch["kind"]] for ch in raw.info["chs"]}
    assert ch_types == EXPECTED


@testing.requires_testing_data
@pytest.mark.parametrize(
    "exclude_after_unique, warns",
    [
        (False, False),
        (True, True),
    ],
)
def test_bdf_multiple_annotation_channels(exclude_after_unique, warns):
    """Test BDF with multiple annotation channels."""
    if warns:
        ctx = pytest.warns(RuntimeWarning, match="Channel names are not unique")
    else:
        ctx = nullcontext()
    with ctx:
        raw = read_raw_bdf(
            bdf_multiple_annotations_path, exclude_after_unique=exclude_after_unique
        )
    assert len(raw.annotations) == 10
    descriptions = np.array(
        [
            "signal_start",
            "EEG-check#1",
            "TestStim#1",
            "TestStim#2",
            "TestStim#3",
            "TestStim#4",
            "TestStim#5",
            "TestStim#6",
            "TestStim#7",
            "Ligths-Off#1",
        ],
        dtype="<U12",
    )
    assert_array_equal(descriptions, raw.annotations.description)


@testing.requires_testing_data
def test_edf_lowpass_zero():
    """Test if a lowpass filter of 0Hz is mapped to the Nyquist frequency."""
    raw = read_raw_edf(edf_stim_resamp_path)
    assert raw.ch_names[100] == "EEG LDAMT_01-REF"
    assert_allclose(raw.info["lowpass"], raw.info["sfreq"] / 2)


@testing.requires_testing_data
def test_edf_annot_sub_s_onset():
    """Test reading of sub-second annotation onsets."""
    raw = read_raw_edf(edf_annot_sub_s_path)
    assert_allclose(raw.annotations.onset, [1.951172, 3.492188])


def test_invalid_date(tmp_path):
    """Test handling of invalid date in EDF header."""
    with open(edf_path, "rb") as f:  # read valid test file
        edf = bytearray(f.read())

    # original date in header is 29.04.14 (2014-04-29) at pos 168:176
    # but we also use Startdate if available,
    # which starts at byte 88 and is b'Startdate 29-APR-2014 X X X'
    # create invalid date 29.02.14 (2014 is not a leap year)

    # one wrong: no warning
    edf[101:104] = b"FEB"
    assert edf[172] == ord("4")
    fname = tmp_path / "temp.edf"
    with open(fname, "wb") as f:
        f.write(edf)
    read_raw_edf(fname)

    # other wrong: no warning
    edf[101:104] = b"APR"
    edf[172] = ord("2")
    with open(fname, "wb") as f:
        f.write(edf)
    read_raw_edf(fname)

    # both wrong: warning
    edf[101:104] = b"FEB"
    edf[172] = ord("2")
    with open(fname, "wb") as f:
        f.write(edf)
    with pytest.warns(RuntimeWarning, match="Invalid measurement date"):
        read_raw_edf(fname)

    # another invalid date 29.00.14 (0 is not a month)
    assert edf[101:104] == b"FEB"
    edf[172] = ord("0")
    with open(fname, "wb") as f:
        f.write(edf)
    with pytest.warns(RuntimeWarning, match="Invalid measurement date"):
        read_raw_edf(fname)


def test_empty_chars():
    """Test blank char support."""
    assert int(_edf_str(b"1819\x00 ")) == 1819


def _hp_lp_rev(*args, **kwargs):
    out, orig_units = _read_edf_header(*args, **kwargs)
    out["lowpass"], out["highpass"] = out["highpass"], out["lowpass"]
    return out, orig_units


def _hp_lp_mod(*args, **kwargs):
    out, orig_units = _read_edf_header(*args, **kwargs)
    out["lowpass"][:] = "1"
    out["highpass"][:] = "10"
    return out, orig_units


@pytest.mark.filterwarnings("ignore:.*too long.*:RuntimeWarning")
@pytest.mark.parametrize(
    "fname, lo, hi, warns, patch_func",
    [
        (edf_path, 256, 0, False, "rev"),
        (edf_uneven_path, 50, 0, False, "rev"),
        (edf_stim_channel_path, 64, 0, False, "rev"),
        pytest.param(edf_overlap_annot_path, 64, 0, False, "rev", marks=td_mark),
        pytest.param(edf_reduced, 256, 0, False, "rev", marks=td_mark),
        pytest.param(test_generator_edf, 100, 0, False, "rev", marks=td_mark),
        pytest.param(edf_stim_resamp_path, 256, 0, False, "rev", marks=td_mark),
        pytest.param(edf_stim_resamp_path, 256, 0, True, "mod", marks=td_mark),
    ],
)
def test_hp_lp_reversed(fname, lo, hi, warns, patch_func, monkeypatch):
    """Test HP/LP reversed (gh-8584)."""
    fname = str(fname)
    raw = read_raw_edf(fname)
    assert raw.info["lowpass"] == lo
    assert raw.info["highpass"] == hi
    if patch_func == "rev":
        monkeypatch.setattr(edf.edf, "_read_edf_header", _hp_lp_rev)
    elif patch_func == "mod":
        monkeypatch.setattr(edf.edf, "_read_edf_header", _hp_lp_mod)
    if warns:
        ctx = pytest.warns(RuntimeWarning, match="greater than lowpass")
        new_lo, new_hi = raw.info["sfreq"] / 2.0, 0.0
    else:
        ctx = nullcontext()
        new_lo, new_hi = lo, hi
    with ctx:
        raw = read_raw_edf(fname)
    assert raw.info["lowpass"] == new_lo
    assert raw.info["highpass"] == new_hi


def test_degenerate():
    """Test checking of some bad inputs."""
    for func in (
        read_raw_edf,
        read_raw_bdf,
        read_raw_gdf,
        partial(_read_header, exclude=(), infer_types=False),
    ):
        with pytest.raises(NotImplementedError, match="Only.*txt.*"):
            func(edf_txt_stim_channel_path)


def test_exclude():
    """Test exclude parameter."""
    exclude = ["I1", "I2", "I3", "I4"]  # list of excluded channels

    raw = read_raw_edf(edf_path, exclude=["I1", "I2", "I3", "I4"])
    for ch in exclude:
        assert ch not in raw.ch_names

    raw = read_raw_edf(edf_path, exclude="I[1-4]")
    for ch in exclude:
        assert ch not in raw.ch_names


@pytest.mark.parametrize(
    "EXPECTED, exclude, exclude_after_unique, warns",
    [
        (["EEG F2-Ref"], "EEG F1-Ref", False, False),
        (["EEG F1-Ref-0", "EEG F2-Ref", "EEG F1-Ref-1"], "EEG F1-Ref-1", False, True),
        (["EEG F2-Ref"], ["EEG F1-Ref"], False, False),
        (["EEG F2-Ref"], "EEG F1-Ref", True, True),
        (["EEG F1-Ref-0", "EEG F2-Ref"], "EEG F1-Ref-1", True, True),
        (["EEG F1-Ref-0", "EEG F2-Ref", "EEG F1-Ref-1"], ["EEG F1-Ref"], True, True),
    ],
)
def test_exclude_duplicate_channel_data(exclude, exclude_after_unique, warns, EXPECTED):
    """Test exclude parameter for duplicate channel data."""
    if warns:
        ctx = pytest.warns(RuntimeWarning, match="Channel names are not unique")
    else:
        ctx = nullcontext()
    with ctx:
        raw = read_raw_edf(
            duplicate_channel_labels_path,
            exclude=exclude,
            exclude_after_unique=exclude_after_unique,
        )
    assert raw.ch_names == EXPECTED


def test_include():
    """Test include parameter."""
    raw = read_raw_edf(edf_path, include=["I1", "I2"])
    assert sorted(raw.ch_names) == ["I1", "I2"]

    raw = read_raw_edf(edf_path, include="I[1-4]")
    assert sorted(raw.ch_names) == ["I1", "I2", "I3", "I4"]

    with pytest.raises(ValueError, match="'exclude' must be empty if 'include' is "):
        raw = read_raw_edf(edf_path, include=["I1", "I2"], exclude="I[1-4]")


@pytest.mark.parametrize(
    "EXPECTED, include, exclude_after_unique, warns",
    [
        (["EEG F1-Ref-0", "EEG F1-Ref-1"], "EEG F1-Ref", False, True),
        ([], "EEG F1-Ref-1", False, False),
        (["EEG F1-Ref-0", "EEG F1-Ref-1"], ["EEG F1-Ref"], False, True),
        (["EEG F1-Ref-0", "EEG F1-Ref-1"], "EEG F1-Ref", True, True),
        (["EEG F1-Ref-1"], "EEG F1-Ref-1", True, True),
        ([], ["EEG F1-Ref"], True, True),
    ],
)
def test_include_duplicate_channel_data(include, exclude_after_unique, warns, EXPECTED):
    """Test include parameter for duplicate channel data."""
    if warns:
        ctx = pytest.warns(RuntimeWarning, match="Channel names are not unique")
    else:
        ctx = nullcontext()
    with ctx:
        raw = read_raw_edf(
            duplicate_channel_labels_path,
            include=include,
            exclude_after_unique=exclude_after_unique,
        )
    assert raw.ch_names == EXPECTED


@testing.requires_testing_data
def test_ch_types():
    """Test reading of channel types from EDF channel label."""
    raw = read_raw_edf(edf_chtypes_path)  # infer_types=False

    labels = [
        "EEG Fp1-Ref",
        "EEG Fp2-Ref",
        "EEG F3-Ref",
        "EEG F4-Ref",
        "EEG C3-Ref",
        "EEG C4-Ref",
        "EEG P3-Ref",
        "EEG P4-Ref",
        "EEG O1-Ref",
        "EEG O2-Ref",
        "EEG F7-Ref",
        "EEG F8-Ref",
        "EEG T7-Ref",
        "EEG T8-Ref",
        "EEG P7-Ref",
        "EEG P8-Ref",
        "EEG Fz-Ref",
        "EEG Cz-Ref",
        "EEG Pz-Ref",
        "POL E",
        "POL PG1",
        "POL PG2",
        "EEG A1-Ref",
        "EEG A2-Ref",
        "POL T1",
        "POL T2",
        "ECG ECG1",
        "ECG ECG2",
        "EEG F9-Ref",
        "EEG T9-Ref",
        "EEG P9-Ref",
        "EEG F10-Ref",
        "EEG T10-Ref",
        "EEG P10-Ref",
        "SaO2 X9",
        "SaO2 X10",
        "POL DC01",
        "POL DC02",
        "POL DC03",
        "POL DC04",
        "POL $A1",
        "POL $A2",
    ]

    # by default all types are 'eeg'
    assert all(t == "eeg" for t in raw.get_channel_types())
    assert raw.ch_names == labels

    raw = read_raw_edf(edf_chtypes_path, infer_types=True)
    data = raw.get_data()

    labels = [
        "Fp1-Ref",
        "Fp2-Ref",
        "F3-Ref",
        "F4-Ref",
        "C3-Ref",
        "C4-Ref",
        "P3-Ref",
        "P4-Ref",
        "O1-Ref",
        "O2-Ref",
        "F7-Ref",
        "F8-Ref",
        "T7-Ref",
        "T8-Ref",
        "P7-Ref",
        "P8-Ref",
        "Fz-Ref",
        "Cz-Ref",
        "Pz-Ref",
        "POL E",
        "POL PG1",
        "POL PG2",
        "A1-Ref",
        "A2-Ref",
        "POL T1",
        "POL T2",
        "ECG1",
        "ECG2",
        "F9-Ref",
        "T9-Ref",
        "P9-Ref",
        "F10-Ref",
        "T10-Ref",
        "P10-Ref",
        "X9",
        "X10",
        "POL DC01",
        "POL DC02",
        "POL DC03",
        "POL DC04",
        "POL $A1",
        "POL $A2",
    ]
    types = [
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "ecg",
        "ecg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "bio",
        "bio",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
        "eeg",
    ]

    assert raw.get_channel_types() == types
    assert raw.ch_names == labels

    with pytest.raises(ValueError, match="cannot overwrite"):
        read_raw_edf(edf_chtypes_path, units="V")
    raw = read_raw_edf(edf_chtypes_path, units="uV")  # should be okay
    data_units = raw.get_data()
    assert_allclose(data, data_units)


@testing.requires_testing_data
def test_anonymization():
    """Test that RawEDF anonymizes data in memory."""
    # gh-11966
    raw = read_raw_edf(edf_stim_resamp_path)
    for key in ("meas_date", "subject_info"):
        assert key not in raw._raw_extras[0]
    bday = raw.info["subject_info"]["birthday"]
    assert bday == datetime.date(1967, 10, 9)
    raw.anonymize()
    assert raw.info["subject_info"]["birthday"] != bday
