#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne._fiff.constants import FIFF
from mne._fiff.tag import _loc_to_coil_trans
from mne.annotations import events_from_annotations, read_annotations
from mne.bem import _fit_sphere
from mne.channels import DigMontage, read_dig_curry
from mne.datasets import testing
from mne.epochs import Epochs
from mne.event import find_events
from mne.io.bti import read_raw_bti
from mne.io.curry import read_impedances_curry, read_raw_curry
from mne.io.curry.curry import (
    _check_curry_filename,
    _check_curry_header_filename,
    _get_curry_version,
)
from mne.io.edf import read_raw_bdf
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import catch_logging

pytest.importorskip("curryreader")

data_dir = testing.data_path(download=False)
curry_dir = data_dir / "curry"
bdf_file = data_dir / "BDF" / "test_bdf_stim_channel.bdf"
curry7_bdf_file = curry_dir / "test_bdf_stim_channel Curry 7.dat"
curry7_bdf_ascii_file = curry_dir / "test_bdf_stim_channel Curry 7 ASCII.dat"
curry8_bdf_file = curry_dir / "test_bdf_stim_channel Curry 8.cdt"
curry8_bdf_ascii_file = curry_dir / "test_bdf_stim_channel Curry 8 ASCII.cdt"
Ref_chan_omitted_file = curry_dir / "Ref_channel_omitted Curry7.dat"
Ref_chan_omitted_reordered_file = curry_dir / "Ref_channel_omitted reordered Curry7.dat"
curry_epoched_file = curry_dir / "Epoched.cdt"
curry_hpi_file = curry_dir / "HPI.cdt"
bti_rfDC_file = data_dir / "BTi" / "erm_HFH" / "c,rfDC"
curry7_rfDC_file = curry_dir / "c,rfDC Curry 7.dat"
curry8_rfDC_file = curry_dir / "c,rfDC Curry 8.cdt"


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
    raw = read_raw_curry(fname, preload=preload)

    assert hasattr(raw, "_data") == preload
    assert raw.n_times == bdf_curry_ref.n_times
    assert raw.info["sfreq"] == bdf_curry_ref.info["sfreq"]

    for field in ["kind", "ch_name"]:
        assert_array_equal(
            [ch[field] for ch in raw.info["chs"]],
            [ch[field] for ch in bdf_curry_ref.info["chs"]],
        )

    assert_allclose(raw.get_data(verbose="error"), bdf_curry_ref.get_data(), atol=tol)

    picks, start, stop = ["C3", "C4"], 200, 800
    assert_allclose(
        raw.get_data(picks=picks, start=start, stop=stop, verbose="error"),
        bdf_curry_ref.get_data(picks=picks, start=start, stop=stop),
        rtol=tol,
    )
    assert not raw.info["dev_head_t"]


@testing.requires_testing_data
def test_read_raw_curry_epoched():
    """Test reading epoched file."""
    ep = read_raw_curry(curry_epoched_file)
    assert isinstance(ep, Epochs)
    assert len(ep.events) == 26
    assert len(ep.annotations) == 0


GOOD_HPI_MATCH = """
FileVersion:	804
NumCoils:	10

0	1	52.73	-74.87	111.56	0.002538		1	57.87	23.21	126.11	0.002692		1	-3.68	-18.54	130.38	0.008380		1	-19.40	49.74	90.95	0.008395		1	-56.17	-6.00	62.95	0.003832		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000
4100	1	50.54	-62.26	151.17	0.002511		1	28.10	56.94	148.74	0.002720		1	-21.68	-43.61	175.47	0.008313		1	-57.26	23.61	147.11	0.008390		1	-80.77	-32.52	125.38	0.003828		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000

"""  # noqa: E501


@pytest.mark.parametrize("good_match", [True, False])
@testing.requires_testing_data
def test_read_raw_curry_hpi(good_match, tmp_path):
    """Test reading hpi file."""
    fname = curry_hpi_file
    if not good_match:
        # real data does not have a good fit
        read_raw_curry(fname, on_bad_hpi_match="ignore")
        with pytest.warns(match="Poor HPI matching"):
            read_raw_curry(fname, on_bad_hpi_match="warn")
        with pytest.raises(ValueError, match="Poor HPI matching"):
            read_raw_curry(fname, on_bad_hpi_match="raise")
    else:
        # tweak HPI point to have a good fit
        for ext in (".cdt", ".cdt.dpa"):
            src = fname.with_suffix(ext)
            dst = tmp_path / fname.with_suffix(ext).name
            copyfile(src, dst)
        fname = tmp_path / fname.name
        with open(fname.with_suffix(fname.suffix + ".hpi"), "w") as fid:
            fid.write(GOOD_HPI_MATCH)
        read_raw_curry(fname, on_bad_hpi_match="ignore")
        read_raw_curry(fname, on_bad_hpi_match="warn")
        read_raw_curry(fname, on_bad_hpi_match="raise")


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
    _test_raw_reader(read_raw_curry, fname=fname)


# These values taken from a different recording but allow us to test
# using our existing filres

HPI_CONTENT = """\
FileVersion:	804
NumCoils:	10

0	1	-50.67	50.98	133.15	0.006406		1	46.45	51.51	143.15	0.006789		1	39.38	-26.67	155.51	0.008034		1	-36.72	-39.95	142.83	0.007700		1	1.61	16.95	172.76	0.001788		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000		0	0.00	0.00	0.00	0.000000
"""  # noqa: E501


LM_CONTENT = """

LANDMARKS_MAG1 START
   ListDescription      = functional landmark positions
   ListUnits            = mm
   ListNrColumns        =  3
   ListNrRows           =  8
   ListNrTimepts        =  1
   ListNrBlocks         =  1
   ListBinary           =  0
   ListType             =  1
   ListTrafoType        =  1
   ListGridType         =  2
   ListFirstColumn      =  1
   ListIndexMin         = -1
   ListIndexMax         = -1
   ListIndexAbsMax      = -1
LANDMARKS_MAG1 END

LANDMARKS_MAG1 START_LIST	# Do not edit!
  75.4535	 5.32907e-15	 2.91434e-16
  1.42109e-14	-75.3212	 9.71445e-16
 -74.4568	-1.42109e-14	 2.51188e-15
 -59.7558	 35.5804	 66.822
  43.15	 43.4107	 78.0027
  38.8415	-41.1884	 81.9941
 -36.683	-59.5119	 66.4338
 -1.07259	-1.88025	 103.747
LANDMARKS_MAG1 END_LIST

LM_INDICES_MAG1 START
   ListDescription      = functional landmark PAN info
   ListUnits            =
   ListNrColumns        =  1
   ListNrRows           =  3
   ListNrTimepts        =  1
   ListNrBlocks         =  1
   ListBinary           =  0
   ListType             =  0
   ListTrafoType        =  0
   ListGridType         =  2
   ListFirstColumn      =  1
   ListIndexMin         = -1
   ListIndexMax         = -1
   ListIndexAbsMax      = -1
LM_INDICES_MAG1 END

LM_INDICES_MAG1 START_LIST	# Do not edit!
  2
  1
  3
LM_INDICES_MAG1 END_LIST

LM_REMARKS_MAG1 START
   ListDescription      = functional landmark labels
   ListUnits            =
   ListNrColumns        =  40
   ListNrRows           =  8
   ListNrTimepts        =  1
   ListNrBlocks         =  1
   ListBinary           =  0
   ListType             =  5
   ListTrafoType        =  0
   ListGridType         =  2
   ListFirstColumn      =  1
   ListIndexMin         = -1
   ListIndexMax         = -1
   ListIndexAbsMax      = -1
LM_REMARKS_MAG1 END

LM_REMARKS_MAG1 START_LIST	# Do not edit!
Left ear
Nasion
Right ear
HPI1
HPI2
HPI3
HPI4
HPI5
LM_REMARKS_MAG1 END_LIST

"""

WANT_TRANS = np.array(
    [
        [0.99729224, -0.07353067, -0.00119791, 0.00126953],
        [0.07319243, 0.99085848, 0.11332405, 0.02670814],
        [-0.00714583, -0.11310488, 0.99355736, 0.04721836],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname,tol",
    [
        pytest.param(curry7_rfDC_file, 1e-6, id="curry 7"),
        pytest.param(curry8_rfDC_file, 1e-3, id="curry 8"),
    ],
)
@pytest.mark.parametrize("mock_dev_head_t", [True, False])
def test_read_raw_curry_rfDC(fname, tol, mock_dev_head_t, tmp_path):
    """Test reading CURRY files."""
    if mock_dev_head_t:
        if "Curry 7" in fname.name:  # not supported yet
            return
        # copy files to tmp_path
        for ext in (".cdt", ".cdt.dpa"):
            src = fname.with_suffix(ext)
            dst = tmp_path / fname.with_suffix(ext).name
            copyfile(src, dst)
            if ext == ".cdt.dpa":
                with open(dst, "a") as fid:
                    fid.write(LM_CONTENT)
        fname = tmp_path / fname.name
        with open(fname.with_suffix(fname.suffix + ".hpi"), "w") as fid:
            fid.write(HPI_CONTENT)

    # check data
    bti_rfDC = read_raw_bti(pdf_fname=bti_rfDC_file, head_shape_fname=None)
    with catch_logging() as log:
        raw = read_raw_curry(fname, verbose=True)
    log = log.getvalue()
    if mock_dev_head_t:
        assert "Composing device" in log
    else:
        assert "Leaving device" in log
        assert "no landmark" in log

    # test on the eeg chans, since these were not renamed by curry
    eeg_names = [
        ch["ch_name"] for ch in raw.info["chs"] if ch["kind"] == FIFF.FIFFV_EEG_CH
    ]

    assert_allclose(raw.get_data(eeg_names), bti_rfDC.get_data(eeg_names), rtol=tol)
    assert bti_rfDC.info["dev_head_t"] is not None  # XXX probably a BTI bug
    if mock_dev_head_t:
        assert raw.info["dev_head_t"] is not None
        assert_allclose(raw.info["dev_head_t"]["trans"], WANT_TRANS, atol=1e-5)
    else:
        assert not raw.info["dev_head_t"]

    # check that most MEG sensors are approximately oriented outward from
    # the device origin
    n_meg = n_eeg = n_other = 0
    pos = list()
    nn = list()
    for ch in raw.info["chs"]:
        if ch["kind"] == FIFF.FIFFV_MEG_CH:
            assert ch["coil_type"] == FIFF.FIFFV_COIL_COMPUMEDICS_ADULT_GRAD
            t = _loc_to_coil_trans(ch["loc"])
            pos.append(t[:3, 3])
            nn.append(t[:3, 2])
            assert_allclose(np.linalg.norm(nn[-1]), 1.0)
            n_meg += 1
        elif ch["kind"] == FIFF.FIFFV_EEG_CH:
            assert ch["coil_type"] == FIFF.FIFFV_COIL_EEG
            n_eeg += 1
        else:
            assert ch["coil_type"] == FIFF.FIFFV_COIL_NONE
            n_other += 1
    assert n_meg == 148
    assert n_eeg == 31
    assert n_other == 15
    pos = np.array(pos)
    nn = np.array(nn)
    rad, origin = _fit_sphere(pos)
    assert 0.11 < rad < 0.13
    pos -= origin
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    angles = np.abs(np.rad2deg(np.arccos((pos * nn).sum(-1))))
    assert (angles < 20).sum() > 100


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
    raw = read_raw_curry(fname)
    events, _ = events_from_annotations(raw, event_id=EVENT_ID)
    assert_allclose(events, REF_EVENTS)
    assert not raw.info["dev_head_t"]


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


def _mock_info_file(src, dst, sfreq, time_step):
    with open(src) as in_file, open(dst, "w") as out_file:
        for line in in_file:
            if "SampleFreqHz" in line:
                out_file.write(line.replace("500", str(sfreq)))
            elif "SampleTimeUsec" in line:
                out_file.write(line.replace("2000", str(time_step)))
            else:
                out_file.write(line)


# In the new version based on curryreader package, time_step is always prioritized, i.e.
# sfreq in the header file will be ignored and overridden by sampling interval
@pytest.fixture(
    params=[
        pytest.param(
            dict(sfreq=500, time_step=1),
            id="correct sfreq",
        ),
        pytest.param(dict(sfreq=0, time_step=2000), id="correct time_step"),
        pytest.param(dict(sfreq=500, time_step=2000), id="both correct"),
        pytest.param(
            dict(sfreq=0, time_step=0),
            id="both 0",
        ),
        pytest.param(
            dict(sfreq=500, time_step=42),
            id="mismatch",
        ),
    ]
)
def sfreq_testing_data(tmp_path, request):
    """Generate different sfreq, time_step scenarios to be tested."""
    sfreq, time_step = request.param["sfreq"], request.param["time_step"]

    # create dummy empty files for 'dat' and 'rs3'
    for fname in ["curry.dat", "curry.rs3"]:
        open(tmp_path / fname, "a").close()

    _mock_info_file(
        src=curry7_bdf_file.with_suffix(".dap"),
        dst=tmp_path / "curry.dap",
        sfreq=sfreq,
        time_step=time_step,
    )
    _mock_info_file(
        src=curry7_bdf_file.with_suffix(".rs3"),
        dst=tmp_path / "curry.rs3",
        sfreq=sfreq,
        time_step=time_step,
    )
    copyfile(curry7_bdf_file, tmp_path / "curry.dat")

    return tmp_path / "curry.dat", sfreq, time_step


@testing.requires_testing_data
def test_sfreq(sfreq_testing_data):
    """Test sfreq and time_step."""
    fname, sfreq, time_step = sfreq_testing_data
    if time_step == 0:
        with pytest.raises(ValueError, match="sampling interval of 0µs."):
            read_raw_curry(fname, preload=False)
    else:
        if sfreq != 1e6 / time_step:
            with pytest.warns(
                RuntimeWarning, match="sfreq will be derived from sample distance."
            ):
                raw = read_raw_curry(fname, preload=False)
        else:
            raw = read_raw_curry(fname, preload=False)
        assert raw.info["sfreq"] == 1e6 / time_step


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname",
    [
        pytest.param(curry_dir / "test_bdf_stim_channel Curry 7.cef", id="7"),
        pytest.param(
            curry_dir / "test_bdf_stim_channel Curry 7 ASCII.cef", id="7 ascii"
        ),
        pytest.param(curry_dir / "test_bdf_stim_channel Curry 8.cdt.cef", id="8"),
        pytest.param(
            curry_dir / "test_bdf_stim_channel Curry 8 ASCII.cdt.cef", id="8 ascii"
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


FILE_EXTENSIONS = {
    "Curry 7": {
        "info": ".dap",
        "data": ".dat",
        "labels": ".rs3",
        "events_cef": ".cef",
        "events_ceo": ".ceo",
        "hpi": ".hpi",
    },
    "Curry 8": {
        "info": ".cdt.dpa",
        "data": ".cdt",
        "labels": ".cdt.dpa",
        "events_cef": ".cdt.cef",
        "events_ceo": ".cdt.ceo",
        "hpi": ".cdt.hpi",
    },
    "Curry 9": {
        "info": ".cdt.dpo",
        "data": ".cdt",
        "labels": ".cdt.dpo",
        "events_cef": ".cdt.cef",
        "events_ceo": ".cdt.ceo",
        "hpi": ".cdt.hpi",
    },
}


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname",
    [
        pytest.param(curry7_bdf_file, id="7"),
        pytest.param(curry8_bdf_file, id="8"),
        pytest.param(curry7_bdf_ascii_file, id="7 (ascii)"),
        pytest.param(curry8_bdf_ascii_file, id="8 (ascii)"),
    ],
)
def test_incomplete_file_suite(tmp_path, fname):
    """Test reading incomplete Curry filesets."""
    original, modified = dict(), dict()

    version = _get_curry_version(fname)

    original["base"] = fname.with_suffix("")
    original["event"] = fname.with_suffix(FILE_EXTENSIONS[version]["events_cef"])
    original["info"] = fname.with_suffix(FILE_EXTENSIONS[version]["info"])
    original["data"] = fname.with_suffix(FILE_EXTENSIONS[version]["data"])
    original["labels"] = fname.with_suffix(FILE_EXTENSIONS[version]["labels"])

    modified["base"] = tmp_path / "curry"
    modified["event"] = modified["base"].with_suffix(
        FILE_EXTENSIONS[version]["events_cef"]
    )
    modified["info"] = modified["base"].with_suffix(FILE_EXTENSIONS[version]["info"])
    modified["data"] = modified["base"].with_suffix(FILE_EXTENSIONS[version]["data"])
    modified["labels"] = modified["base"].with_suffix(
        FILE_EXTENSIONS[version]["labels"]
    )

    # only data
    copyfile(src=original["data"], dst=modified["data"])
    _msg = rf"does not exist: .*{modified['event'].name}.*"
    with pytest.raises(FileNotFoundError, match=_msg):
        read_annotations(modified["event"], sfreq="auto")

    # events missing
    copyfile(src=original["info"], dst=modified["info"])
    with pytest.raises(FileNotFoundError, match=_msg):
        read_annotations(modified["event"], sfreq="auto")

    # all there
    copyfile(src=original["event"], dst=modified["event"])
    if not modified["labels"].exists():
        copyfile(src=original["labels"], dst=modified["labels"])
    read_raw_curry(modified["data"])


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname,expected_channel_list",
    [
        pytest.param(
            Ref_chan_omitted_file,
            ["FP1", "FPZ", "FP2", "VEO", "EKG", "Trigger"],
            id="Ref omitted, normal order",
        ),
        pytest.param(
            Ref_chan_omitted_reordered_file,
            ["FP2", "FPZ", "FP1", "VEO", "EKG", "Trigger"],
            id="Ref omitted, reordered",
        ),
    ],
)
def test_read_files_missing_channel(fname, expected_channel_list):
    """Test reading data files that has an omitted channel."""
    # This for Git issue #8391.  In some cases, the 'labels' (.rs3 file will
    # list channels that are not actually saved in the datafile (such as the
    # 'Ref' channel).  These channels are denoted in the 'info' (.dap) file
    # in the CHAN_IN_FILE section with a '0' as their index.
    # If the CHAN_IN_FILE section is present, the code also assures that the
    # channels are sorted in the prescribed order.
    # This test makes sure the data load correctly, and that we end up with
    # the proper channel list.
    raw = read_raw_curry(fname, preload=True)
    assert raw.ch_names == expected_channel_list


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname,expected_meas_date",
    [
        pytest.param(
            Ref_chan_omitted_file,
            datetime(2018, 11, 21, 12, 53, 48, 525000, tzinfo=timezone.utc),
            id="valid start date",
        ),
        pytest.param(curry7_rfDC_file, None, id="start date year is 0"),
        pytest.param(
            curry7_bdf_file,
            None,
            id="start date seconds invalid",
        ),
    ],
)
def test_meas_date(fname, expected_meas_date):
    """Test reading acquisition start datetime info info['meas_date']."""
    # This for Git issue #8398.  The 'info' (.dap) file includes acquisition
    # start date & time.  Test that this goes into raw.info['meas_date'].
    # If the information is not valid, raw.info['meas_date'] should be None
    raw = read_raw_curry(fname, preload=False)
    assert raw.info["meas_date"] == expected_meas_date


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname, others",
    [
        pytest.param(curry7_bdf_file, (".dap", ".rs3"), id="curry7"),
        pytest.param(curry8_bdf_file, (".cdt.cef", ".cdt.dpa"), id="curry8"),
    ],
)
def test_dot_names(fname, others, tmp_path):
    """Test that dots are parsed properly (e.g., in paths)."""
    my_path = tmp_path / "dot.dot.dot"
    my_path.mkdir()
    my_path = my_path / Path(fname).parts[-1]
    fname = Path(fname)
    copyfile(fname, my_path)
    for ext in others:
        this_fname = fname.with_suffix(ext)
        to_fname = my_path.with_suffix(ext)
        copyfile(this_fname, to_fname)
    read_raw_curry(my_path)


@testing.requires_testing_data
def test_read_device_info():
    """Test extraction of device_info."""
    raw = read_raw_curry(curry7_bdf_file)
    assert not raw.info["device_info"]
    raw2 = read_raw_curry(Ref_chan_omitted_file)
    assert isinstance(raw2.info["device_info"], dict)


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
    actual_imp = np.empty(shape=(0, 3))  # TODO - need better testing data
    assert_allclose(
        imp,
        actual_imp,
    )


def _mock_info_noeeg(src, dst):
    # artificially remove eeg channels
    content_hdr = src.read_text()
    if ".dap" in src.name:
        # curry 7
        content_hdr_ = content_hdr
    elif ".rs3" in src.name:
        # curry 7
        content_hdr_ = (
            content_hdr.split("NUMBERS START")[0]
            + content_hdr.split("TRANSFORM END_LIST")[-1]
        )
    else:
        # curry 8
        content_hdr_ = (
            content_hdr.split("LABELS START")[0]
            + content_hdr.split("SENSORS END_LIST")[-1]
        )
    # both
    content_hdr_ = content_hdr_.replace(
        "NumChannels          =  194", "NumChannels          =  163"
    )
    content_hdr_ = content_hdr_.replace(
        "NumChanThisGroup     =  31", "NumChanThisGroup     =  0"
    )
    content_hdr_ = content_hdr_.replace(
        "NumSensorsThisGroup     =  31", "NumSensorsThisGroup     =  0"
    )
    with dst.open("w+") as f:
        f.write(content_hdr_)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname,mont_present",
    [
        pytest.param(curry7_bdf_file, True, id="curry 7"),
        pytest.param(curry8_bdf_file, True, id="curry 8"),
        pytest.param(curry7_bdf_ascii_file, True, id="curry 7 ascii"),
        pytest.param(curry8_bdf_ascii_file, True, id="curry 8 ascii"),
        pytest.param(curry7_rfDC_file, False, id="no eeg, curry 7"),
        pytest.param(curry8_rfDC_file, False, id="no eeg, curry 8"),
        pytest.param(curry_hpi_file, True, id="curry 8, w/ HPI data"),
    ],
)
def test_read_montage_curry(tmp_path, fname, mont_present):
    """Test reading montage from CURRY files."""
    if mont_present:
        assert isinstance(read_dig_curry(fname), DigMontage)
    else:
        # copy files to tmp_path
        for ext in (".cdt", ".cdt.hpi", ".cdt.dpa", ".dat", ".dap", ".rs3"):
            src = fname.with_suffix(ext)
            dst = tmp_path / fname.with_suffix(ext).name
            if src.exists():
                if ext in [".cdt.dpa", ".dap", ".rs3"]:
                    _mock_info_noeeg(src, dst)
                else:
                    copyfile(src, dst)
        with pytest.raises(ValueError, match="No eeg sensor locations found"):
            read_dig_curry(tmp_path / fname.name)
