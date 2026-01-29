#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..._fiff._digitization import _make_dig_points
from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ..._fiff.tag import _coil_trans_to_loc
from ..._fiff.utils import _mult_cal_one, _read_segments_file
from ...annotations import annotations_from_events
from ...epochs import Epochs
from ...surface import _normal_orth
from ...transforms import (
    Transform,
    _angle_between_quats,
    apply_trans,
    combine_transforms,
    get_ras_to_neuromag_trans,
    invert_transform,
    rot_to_quat,
)
from ...utils import (
    _on_missing,
    _soft_import,
    catch_logging,
    logger,
    verbose,
    warn,
)
from ..base import BaseRaw
from ..ctf.trans import _quaternion_align

CURRY_SUFFIX_DATA = [".cdt", ".dat"]
CURRY_SUFFIX_HDR = [".cdt.dpa", ".cdt.dpo", ".dap"]
CURRY_SUFFIX_LABELS = [".cdt.dpa", ".cdt.dpo", ".rs3"]


def _get_curry_version(fname):
    """Check out the curry file version."""
    fname_hdr = _check_curry_header_filename(_check_curry_filename(fname))
    content_hdr = fname_hdr.read_text()
    return (
        "Curry 7"
        if ".dap" in str(fname_hdr)
        else "Curry 8"
        if re.compile(r"FileVersion\s*=\s*[0-9]+")
        .search(content_hdr)
        .group(0)
        .split()[-1][0]
        == "8"
        else "Curry 9"
        if re.compile(r"FileVersion\s*=\s*[0-9]+")
        .search(content_hdr)
        .group(0)
        .split()[-1][0]
        == "9"
        else None
    )


def _check_curry_filename(fname):
    fname_in = Path(fname).expanduser()
    fname_out = None
    # try suffixes
    if fname_in.suffix in CURRY_SUFFIX_DATA:
        fname_out = fname_in
    elif (
        fname_in.with_suffix("").exists()
        and fname_in.with_suffix("").suffix in CURRY_SUFFIX_DATA
    ):
        fname_out = fname_in.with_suffix("")
    else:
        for data_suff in CURRY_SUFFIX_DATA:
            if fname_in.with_suffix(data_suff).exists():
                fname_out = fname_in.with_suffix(data_suff)
                break
    # final check
    if not fname_out or not fname_out.exists():
        raise FileNotFoundError(
            f"no curry data file found (.dat or .cdt), checked {fname_out or fname_in}"
        )
    return fname_out


def _check_curry_header_filename(fname):
    fname_in = Path(fname)
    fname_hdr = None
    # try suffixes
    for hdr_suff in CURRY_SUFFIX_HDR:
        if fname_in.with_suffix(hdr_suff).exists():
            fname_hdr = fname_in.with_suffix(hdr_suff)
            break
    # final check
    if not fname_hdr or not fname_in.exists():
        raise FileNotFoundError(
            f"no corresponding header file found {CURRY_SUFFIX_HDR}"
        )
    return fname_hdr


def _check_curry_labels_filename(fname):
    fname_in = Path(fname)
    fname_labels = None
    # try suffixes
    for hdr_suff in CURRY_SUFFIX_LABELS:
        if fname_in.with_suffix(hdr_suff).exists():
            fname_labels = fname_in.with_suffix(hdr_suff)
            break
    # final check
    if not fname_labels or not fname_in.exists():
        raise FileNotFoundError(
            f"no corresponding labels file found {CURRY_SUFFIX_HDR}"
        )
    return fname_labels


def _check_curry_sfreq_consistency(fname_hdr):
    content_hdr = fname_hdr.read_text()
    stime = float(
        re.compile(r"SampleTimeUsec\s*=\s*.+").search(content_hdr).group(0).split()[-1]
    )
    sfreq = float(
        re.compile(r"SampleFreqHz\s*=\s*.+").search(content_hdr).group(0).split()[-1]
    )
    if stime == 0:
        raise ValueError("Header file indicates a sampling interval of 0µs.")
    if not np.isclose(1e6 / stime, sfreq):
        warn(
            f"Sample distance ({stime}µs) and sample frequency ({sfreq}Hz) in header "
            "file do not match! sfreq will be derived from sample distance."
        )


def _get_curry_meas_info(fname):
    # Note that the time zone information is not stored in the Curry info
    # file, and it seems the start time info is in the local timezone
    # of the acquisition system (which is unknown); therefore, just set
    # the timezone to be UTC.  If the user knows otherwise, they can
    # change it later.  (Some Curry files might include StartOffsetUTCMin,
    # but its presence is unpredictable, so we won't rely on it.)
    fname_hdr = _check_curry_header_filename(fname)
    content_hdr = fname_hdr.read_text()

    # read meas_date
    meas_date = [
        int(re.compile(rf"{v}\s*=\s*-?\d+").search(content_hdr).group(0).split()[-1])
        for v in [
            "StartYear",
            "StartMonth",
            "StartDay",
            "StartHour",
            "StartMin",
            "StartSec",
            "StartMillisec",
        ]
    ]
    try:
        meas_date = datetime(
            *meas_date[:-1],
            meas_date[-1] * 1000,  # -> microseconds
            timezone.utc,
        )
    except Exception:
        meas_date = None

    # read datatype
    byteorder = (
        re.compile(r"DataByteOrder\s*=\s*[A-Z]+")
        .search(content_hdr)
        .group()
        .split()[-1]
    )
    is_ascii = byteorder == "ASCII"

    # amplifier info
    # TODO - PRIVACY
    # seems like there can be identifiable information (serial numbers, dates).
    # MNE anonymization functions only overwrite "serial" and "site", though
    # TODO - FUTURE ENHANCEMENT
    # # there can be filter details in AmplifierInfo, too
    amp_info = (
        re.compile(r"AmplifierInfo\s*=.*\n")
        .search(content_hdr)
        .group()
        .strip("\n")
        .split("= ")[-1]
        .strip()
    )

    device_info = (
        dict(serial=amp_info)
        if amp_info != ""
        else None  # model="", serial="", site=""
    )

    return meas_date, is_ascii, device_info


def _get_curry_recording_type(fname):
    _soft_import("curryreader", "read recording modality")

    import curryreader

    epochinfo = curryreader.read(str(fname), plotdata=0, verbosity=1)["epochinfo"]
    if epochinfo.size == 0:
        return "raw"
    else:
        n_average = epochinfo[:, 0]
        if (n_average == 1).all():
            return "epochs"
        else:
            return "evoked"


def _get_curry_epoch_info(fname):
    _soft_import("curryreader", "read epoch info")
    _soft_import("pandas", "dataframe integration")

    import curryreader
    import pandas as pd

    # use curry-python-reader
    currydata = curryreader.read(str(fname), plotdata=0, verbosity=1)

    # get epoch info
    sfreq = currydata["info"]["samplingfreq"]
    n_samples = currydata["info"]["samples"]
    n_epochs = len(currydata["epochlabels"])
    epochinfo = currydata["epochinfo"]
    epochtypes = epochinfo[:, 2].astype(int).tolist()
    epochlabels = currydata["epochlabels"]
    epochmetainfo = pd.DataFrame(
        epochinfo[:, -4:], columns=["accept", "correct", "response", "response time"]
    )
    # create mne events
    events = np.array(
        [[i * n_samples for i in range(n_epochs)], [0] * n_epochs, epochtypes]
    ).T
    event_id = dict(zip(epochlabels, epochtypes))
    return dict(
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=(n_samples - 1) / sfreq,
        baseline=None,
        detrend=None,
        verbose=False,
        metadata=epochmetainfo,
        reject_by_annotation=False,
        reject=None,
    )


def _get_curry_meg_normals(fname):
    fname_lbl = _check_curry_labels_filename(fname)
    normals_str = fname_lbl.read_text().split("\n")
    # i_start, i_stop = [
    #    i
    #    for i, ll in enumerate(normals_str)
    #    if ("NORMALS" in ll and "START_LIST" in ll)
    #    or ("NORMALS" in ll and "END_LIST" in ll)
    # ]
    # normals_str = [nn.split("\t") for nn in normals_str[i_start + 1 : i_stop]]
    i_list = [
        i
        for i, ll in enumerate(normals_str)
        if ("NORMALS" in ll and "START_LIST" in ll)
        or ("NORMALS" in ll and "END_LIST" in ll)
    ]
    assert len(i_list) % 2 == 0
    i_start_list = i_list[::2]
    i_stop_list = i_list[1::2]
    normals_str = [
        nn.split("\t")
        for i_start, i_stop in zip(i_start_list, i_stop_list)
        for nn in normals_str[i_start + 1 : i_stop]
    ]
    return np.array([[float(nnn.strip()) for nnn in nn] for nn in normals_str])


def _extract_curry_info(fname):
    _soft_import("curryreader", "read file header")

    import curryreader

    # check if sfreq values make sense
    fname_hdr = _check_curry_header_filename(fname)
    _check_curry_sfreq_consistency(fname_hdr)

    # use curry-python-reader
    currydata = curryreader.read(str(fname), plotdata=0, verbosity=1)

    # basic info
    sfreq = currydata["info"]["samplingfreq"]
    n_samples = currydata["info"]["samples"]
    if n_samples != currydata["data"].shape[0]:  # normal in epoched data
        n_samples = currydata["data"].shape[0]
        if _get_curry_recording_type(fname) == "raw":
            warn(
                "sample count from header doesn't match actual data! "
                "file corrupted? will use data shape"
            )

    # channel information
    n_ch = currydata["info"]["channels"]
    ch_names = currydata["labels"]
    ch_pos = currydata["sensorpos"]
    landmarks = currydata["landmarks"]
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    landmarkslabels = currydata["landmarkslabels"]
    hpimatrix = currydata["hpimatrix"]
    if isinstance(currydata["hpimatrix"], np.ndarray) and hpimatrix.ndim == 1:
        hpimatrix = hpimatrix[np.newaxis, :]

    # data
    orig_format = "int"
    # curryreader.py always reads float32, but this is probably just numpy.
    # legacy MNE code states int.

    # events
    events = currydata["events"]
    annotations = currydata["annotations"]
    assert len(annotations) == len(events)
    if len(events) > 0:
        event_desc = dict()
        for k, v in zip(events[:, 1], annotations):
            if int(k) not in event_desc.keys():
                event_desc[int(k)] = v.strip() if (v.strip() != "") else str(int(k))
    else:
        event_desc = None

    # impedance measurements
    # moved to standalone def; see read_impedances_curry
    # impedances = currydata["impedances"]

    # get other essential info not provided by curryreader
    # channel types and units
    ch_types, units = [], []
    ch_groups = fname_hdr.read_text().split("DEVICE_PARAMETERS")[1::2]
    for ch_group in ch_groups:
        ch_group = re.compile(r"\s+").sub(" ", ch_group).strip()
        groupid = ch_group.split()[0]
        unit = ch_group.split("DataUnit = ")[1].split()[0]
        n_ch_group = int(ch_group.split("NumChanThisGroup = ")[1].split()[0])
        ch_type = (
            "mag" if ("MAG" in groupid) else "misc" if ("OTHER" in groupid) else "eeg"
        )
        # combine info
        ch_types += [ch_type] * n_ch_group
        units += [unit] * n_ch_group

    # This for Git issue #8391.  In some cases, the 'labels' (.rs3 file will
    # list channels that are not actually saved in the datafile (such as the
    # 'Ref' channel).  These channels are denoted in the 'info' (.dap) file
    # in the CHAN_IN_FILE section with a '0' as their index.
    #
    # current curryreader cannot cope with this - loads the list of channels solely
    # based on their order, so can be false. fix it here!
    if not len(ch_types) == len(units) == len(ch_names) == n_ch:
        # read relevant info
        fname_lbl = _check_curry_labels_filename(fname)
        lbl = fname_lbl.read_text().split("START_LIST")
        ch_names_full = []
        for i in range(1, len(lbl)):
            if "LABELS" in lbl[i - 1].split()[-1]:
                for ll in lbl[i].split("\n")[1:]:
                    if "LABELS" not in ll:
                        ch_names_full.append(ll.strip())
                    else:
                        break
        hdr = fname_hdr.read_text().split("START_LIST")
        chaninfile_full = []
        for i in range(1, len(hdr)):
            if "CHAN_IN_FILE" in hdr[i - 1].split()[-1]:
                for ll in hdr[i].split("\n")[1:]:
                    if "CHAN_IN_FILE" not in ll:
                        chaninfile_full.append(int(ll.strip()))
                    else:
                        break
        # drop channels with chan_in_file==0, account for order
        i_drop = [i for i, ich in enumerate(chaninfile_full) if ich == 0]
        ch_names = [
            ch_names_full[i] for i in np.argsort(chaninfile_full) if i not in i_drop
        ]
        ch_pos = np.array(
            [
                ch_pos[i]
                for i in np.argsort(chaninfile_full)
                if (i not in i_drop) and (i < len(ch_pos))
            ]
        )
        ch_types = [ch_types[i] for i in np.argsort(chaninfile_full) if i not in i_drop]
        units = [units[i] for i in np.argsort(chaninfile_full) if i not in i_drop]

    assert len(ch_types) == len(units) == len(ch_names) == n_ch
    assert len(ch_pos) == ch_types.count("eeg") + ch_types.count("mag")

    # finetune channel types (e.g. stim, eog etc might be identified by name)
    # TODO - FUTURE ENHANCEMENT

    # scale data to SI units
    orig_units = dict(zip(ch_names, units))
    cals = [
        1.0 / 1e15 if (u == "fT") else 1.0 / 1e6 if (u == "uV") else 1.0 for u in units
    ]

    return (
        sfreq,
        n_samples,
        ch_names,
        ch_types,
        ch_pos,
        landmarks,
        landmarkslabels,
        hpimatrix,
        events,
        event_desc,
        orig_format,
        orig_units,
        cals,
    )


def _read_annotations_curry(fname, sfreq="auto"):
    r"""Read events from Curry event files.

    Parameters
    ----------
    fname : path-like
        The filename.
    sfreq : float | 'auto'
        The sampling frequency in the file. If set to 'auto' then the
        ``sfreq`` is taken from the fileheader.

    Returns
    -------
    annot : instance of Annotations | None
        The annotations.
    """
    fname = _check_curry_filename(fname)

    (sfreq_fromfile, _, _, _, _, _, _, _, events, event_desc, _, _, _) = (
        _extract_curry_info(fname)
    )
    if sfreq == "auto":
        sfreq = sfreq_fromfile
    elif np.isreal(sfreq):
        if float(sfreq) != float(sfreq_fromfile):
            warn(
                f"provided sfreq ({sfreq} Hz) does not match freq from fileheader "
                "({sfreq_fromfile} Hz)!"
            )
    else:
        raise ValueError("'sfreq' must be numeric or 'auto'")

    if isinstance(events, np.ndarray):  # if there are events
        events = events.astype("int")
        events = np.insert(events, 1, np.diff(events[:, 2:]).flatten(), axis=1)[:, :3]
        return annotations_from_events(events, sfreq, event_desc=event_desc)
    else:
        warn("no event annotations found")
        return None


def _set_chanloc_curry(
    inst, ch_types, ch_pos, landmarks, landmarkslabels, hpimatrix, on_bad_hpi_match
):
    ch_names = inst.info["ch_names"]

    # scale ch_pos to m?!
    ch_pos /= 1000.0
    landmarks /= 1000.0
    # channel locations
    # what about misc without pos? can they mess things up if unordered?
    assert len(ch_pos) >= (ch_types.count("mag") + ch_types.count("eeg"))
    assert len(ch_pos) == (ch_types.count("mag") + ch_types.count("eeg"))
    ch_pos_meg = {
        ch_names[i]: ch_pos[i, :3] for i, t in enumerate(ch_types) if t == "mag"
    }
    ch_pos_eeg = {
        ch_names[i]: ch_pos[i, :3] for i, t in enumerate(ch_types) if t == "eeg"
    }

    # landmarks and headshape
    # FIX: one of the test files (c,rfDC*.cdt) names landmarks differently:
    NAS_NAMES = ["nasion", "nas"]
    LPA_NAMES = ["left ear", "lpa"]
    RPA_NAMES = ["right ear", "rpa"]
    landmarkslabels = [
        "Nas"
        if (ll.lower() in NAS_NAMES)
        else "LPA"
        if (ll.lower() in LPA_NAMES)
        else "RPA"
        if (ll.lower() in RPA_NAMES)
        else ll
        for ll in landmarkslabels
    ]
    landmark_dict = dict(zip(landmarkslabels, landmarks))
    for k in ["Nas", "RPA", "LPA"]:
        if k not in landmark_dict.keys():
            landmark_dict[k] = None
    if len(landmarkslabels) > 0:
        hpi_pos = landmarks[
            [i for i, n in enumerate(landmarkslabels) if re.match("HPI.?[1-99]", n)],
            :,
        ]
    else:
        hpi_pos = None
    if len(landmarkslabels) > 0:
        hsp_pos = landmarks[
            [i for i, n in enumerate(landmarkslabels) if re.match("H.?[1-99]", n)], :
        ]
    else:
        hsp_pos = None

    has_cards = (
        False
        if (
            isinstance(landmark_dict["Nas"], type(None))
            and isinstance(landmark_dict["LPA"], type(None))
            and isinstance(landmark_dict["RPA"], type(None))
        )
        else True
    )
    has_hpi = True if isinstance(hpi_pos, np.ndarray) else False

    add_missing_fiducials = not has_cards  # raises otherwise
    dig = _make_dig_points(
        nasion=landmark_dict["Nas"],
        lpa=landmark_dict["LPA"],
        rpa=landmark_dict["RPA"],
        hpi=hpi_pos,
        extra_points=hsp_pos,
        dig_ch_pos=ch_pos_eeg,
        coord_frame="head",
        add_missing_fiducials=add_missing_fiducials,
    )
    with inst.info._unlock():
        inst.info["dig"] = dig

    # loc transformation for meg sensors (taken from previous version)
    if len(ch_pos_meg) > 0:
        R = np.eye(4)
        R[[0, 1], [0, 1]] = -1  # rotate 180 deg
        # shift down and back
        # (chosen by eyeballing to make the helmet look roughly correct)
        R[:3, 3] = [0.0, -0.015, -0.12]
        curry_dev_dev_t = Transform("ctf_meg", "meg", R)

        ch_normals_meg = _get_curry_meg_normals(inst.filenames[0])
        assert len(ch_normals_meg) == len(ch_pos_meg)
    else:
        curry_dev_dev_t, ch_normals_meg = None, None
    # fill up chanlocs
    assert len(ch_names) == len(ch_types) >= len(ch_pos)
    for i, (ch_name, ch_type, ch_loc) in enumerate(zip(ch_names, ch_types, ch_pos)):
        assert inst.info["ch_names"][i] == ch_name
        ch = inst.info["chs"][i]
        if ch_type == "eeg":
            with inst.info._unlock():
                ch["loc"][:3] = ch_loc[:3]
                ch["coord_frame"] = FIFF.FIFFV_COORD_HEAD
        elif ch_type == "mag":
            # transform mode
            pos = ch_loc[:3]  # just the inner coil for MEG
            pos = apply_trans(curry_dev_dev_t, pos)
            nn = ch_normals_meg[i]
            assert np.isclose(np.linalg.norm(nn), 1.0, atol=1e-4)
            nn /= np.linalg.norm(nn)
            nn = apply_trans(curry_dev_dev_t, nn, move=False)
            trans = np.eye(4)
            trans[:3, 3] = pos
            trans[:3, :3] = _normal_orth(nn).T
            with inst.info._unlock():
                ch["loc"] = _coil_trans_to_loc(trans)
                # TODO: We should figure out if all files are Compumedics,
                # and even then figure out if it's adult or child
                ch["coil_type"] = FIFF.FIFFV_COIL_COMPUMEDICS_ADULT_GRAD
                ch["coord_frame"] = FIFF.FIFFV_COORD_DEVICE
        elif ch_type == "misc":
            pass
        else:
            raise NotImplementedError

    # TODO - REVIEW NEEDED
    # do we need further transformations for MEG channel positions?
    # the testfiles i got look good to me..
    _make_trans_dig(
        inst.info,
        curry_dev_dev_t,
        landmark_dict,
        has_cards,
        has_hpi,
        hpimatrix,
        on_bad_hpi_match,
    )


def _make_trans_dig(
    info,
    curry_dev_dev_t,
    landmark_dict,
    has_cards,
    has_hpi,
    chpidata,
    on_bad_hpi_match,
):
    cards = {
        FIFF.FIFFV_POINT_LPA: landmark_dict["LPA"],
        FIFF.FIFFV_POINT_NASION: landmark_dict["Nas"],
        FIFF.FIFFV_POINT_RPA: landmark_dict["RPA"],
    }

    # Coordinate frame transformations and definitions
    no_msg = "Leaving device<->head transform as None"
    info["dev_head_t"] = None
    lm = [v for v in landmark_dict.values() if isinstance(v, np.ndarray)]
    if len(lm) == 0:
        # no dig
        logger.info(no_msg + " (no landmarks found)")
        return

    if has_cards and has_hpi:  # have all three
        logger.info("Composing device<->head transformation from dig points")
        hpi_u = np.array(
            [d["r"] for d in info["dig"] if d["kind"] == FIFF.FIFFV_POINT_HPI], float
        )
        hpi_c = np.ascontiguousarray(chpidata[0][: len(hpi_u), 1:4])
        bad_hpi_match = False
        try:
            with catch_logging() as log:
                unknown_curry_t = _quaternion_align(
                    "unknown",
                    "ctf_meg",
                    hpi_u.astype("float64"),
                    hpi_c.astype("float64"),
                    1e-2,
                )
        except RuntimeError:
            bad_hpi_match = True
            with catch_logging() as log:
                unknown_curry_t = _quaternion_align(
                    "unknown",
                    "ctf_meg",
                    hpi_u.astype("float64"),
                    hpi_c.astype("float64"),
                    1e-1,
                )
        logger.info(log.getvalue())

        angle = np.rad2deg(
            _angle_between_quats(
                np.zeros(3), rot_to_quat(unknown_curry_t["trans"][:3, :3])
            )
        )
        dist = 1000 * np.linalg.norm(unknown_curry_t["trans"][:3, 3])
        logger.info(f"   Fit a {angle:0.1f}° rotation, {dist:0.1f} mm translation")

        if bad_hpi_match:
            _on_missing(
                on_bad_hpi_match,
                "Poor HPI matching (see log above)!",
                name="on_bad_hpi_match",
            )

        unknown_dev_t = combine_transforms(
            unknown_curry_t, curry_dev_dev_t, "unknown", "meg"
        )
        unknown_head_t = Transform(
            "unknown",
            "head",
            get_ras_to_neuromag_trans(
                *(
                    cards[key]
                    for key in (
                        FIFF.FIFFV_POINT_NASION,
                        FIFF.FIFFV_POINT_LPA,
                        FIFF.FIFFV_POINT_RPA,
                    )
                )
            ),
        )
        with info._unlock():
            info["dev_head_t"] = combine_transforms(
                invert_transform(unknown_dev_t), unknown_head_t, "meg", "head"
            )
            for d in info["dig"]:
                d.update(
                    coord_frame=FIFF.FIFFV_COORD_HEAD,
                    r=apply_trans(unknown_head_t, d["r"]),
                )
    else:
        if has_cards:
            no_msg += " (no .hpi file found)"
        elif has_hpi:
            no_msg += " (not all cardinal points found)"
        else:
            no_msg += " (neither cardinal points nor .hpi file found)"
        logger.info(no_msg)


@verbose
def read_raw_curry(
    fname, preload=False, on_bad_hpi_match="warn", verbose=None
) -> "RawCurry":
    """Read raw data from Curry files.

    .. versionchanged:: 1.11
       This function now  requires ``curryreader`` to be installed.

    Parameters
    ----------
    fname : path-like
        Path to a valid curry file.
    %(preload)s
    %(on_bad_hpi_match)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawCurry
        A Raw object containing Curry data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawCurry.
    """
    fname = _check_curry_filename(fname)
    fname_hdr = _check_curry_header_filename(fname)

    _check_curry_sfreq_consistency(fname_hdr)

    rectype = _get_curry_recording_type(fname)

    inst = RawCurry(fname, preload, on_bad_hpi_match, verbose)
    if rectype in ["epochs", "evoked"]:
        curry_epoch_info = _get_curry_epoch_info(fname)
        inst = Epochs(inst, **curry_epoch_info)
        if rectype == "evoked":
            raise NotImplementedError  # not sure this is even supported format
    return inst


class RawCurry(BaseRaw):
    """Raw object from Curry file.

    Parameters
    ----------
    fname : path-like
        Path to a valid curry file.
    %(preload)s
    %(on_bad_hpi_match)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.

    """

    @verbose
    def __init__(self, fname, preload=False, on_bad_hpi_match="warn", verbose=None):
        fname = _check_curry_filename(fname)

        (
            sfreq,
            n_samples,
            ch_names,
            ch_types,
            ch_pos,
            landmarks,
            landmarkslabels,
            hpimatrix,
            events,
            event_desc,
            orig_format,
            orig_units,
            cals,
        ) = _extract_curry_info(fname)

        meas_date, is_ascii, device_info = _get_curry_meas_info(fname)

        # construct info
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        info["device_info"] = device_info

        # create raw object
        last_samps = [n_samples - 1]
        raw_extras = dict(is_ascii=is_ascii)
        super().__init__(
            info,
            preload=False,
            filenames=[fname],
            last_samps=last_samps,
            orig_format=orig_format,
            raw_extras=[raw_extras],
            orig_units=orig_units,
            verbose=verbose,
        )

        # set meas_date
        self.set_meas_date(meas_date)

        # scale data to SI units
        self._cals = np.array(cals)
        if isinstance(preload, bool | np.bool_) and preload:
            self.load_data()

        # set events / annotations
        # format from curryreader: sample, etype, startsample, endsample
        if isinstance(events, np.ndarray):  # if there are events
            events = events.astype("int")
            events = np.insert(events, 1, np.diff(events[:, 2:]).flatten(), axis=1)[
                :, :3
            ]
            annot = annotations_from_events(events, sfreq, event_desc=event_desc)
            self.set_annotations(annot)

        # add HPI data (if present)
        # TODO - FUTURE ENHANCEMENT
        # from curryreader docstring:
        # "HPI-coil measurements matrix (Orion-MEG only) where every row is:
        # [measurementsample, dipolefitflag, x, y, z, deviation]"
        #
        # that's incorrect, though. it ratehr seems to be:
        # [sample, dipole_1, x_1,y_1, z_1, dev_1, ..., dipole_n, x_n, ...]
        # for all n coils.
        #
        # Do not implement cHPI reader for now.
        # Can be used for dev-head transform, though!
        if not isinstance(hpimatrix, list):
            # warn("cHPI data found, but reader not implemented.")
            hpisamples = hpimatrix[:, 0]
            n_coil = int((hpimatrix.shape[1] - 1) / 5)
            hpimatrix = hpimatrix[:, 1:].reshape(hpimatrix.shape[0], n_coil, 5) / 1000
            logger.info(f"found {len(hpisamples)} cHPI samples for {n_coil} coils")

        # add sensor locations
        # TODO - REVIEW NEEDED
        assert len(self.info["ch_names"]) == len(ch_types) >= len(ch_pos)
        _set_chanloc_curry(
            inst=self,
            ch_types=ch_types,
            ch_pos=ch_pos,
            landmarks=landmarks,
            landmarkslabels=landmarkslabels,
            hpimatrix=hpimatrix,
            on_bad_hpi_match=on_bad_hpi_match,
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        if self._raw_extras[fi]["is_ascii"]:
            if isinstance(idx, slice):
                idx = np.arange(idx.start, idx.stop)
            block = np.loadtxt(
                self.filenames[0], skiprows=start, max_rows=stop - start, ndmin=2
            ).T
            _mult_cal_one(data, block, idx, cals, mult)

        else:
            _read_segments_file(
                self, data, idx, fi, start, stop, cals, mult, dtype="<f4"
            )


@verbose
def read_impedances_curry(fname, verbose=None):
    """Read impedance measurements from Curry files.

    Parameters
    ----------
    fname : path-like
        Path to a valid curry file.
    %(verbose)s

    Returns
    -------
    ch_names : list
        A list object containing channel names
    impedances : np.ndarray
        An array containing up to 10 impedance measurements for all recorded channels.

    """
    _soft_import("curryreader", "read impedances")

    import curryreader

    # use curry-python-reader to load data
    fname = _check_curry_filename(fname)
    fname_hdr = _check_curry_header_filename(fname)

    _check_curry_sfreq_consistency(fname_hdr)

    currydata = curryreader.read(str(fname), plotdata=0, verbosity=1)

    impedances = currydata["impedances"]
    ch_names = currydata["labels"]

    # get impedance measurement times
    # TODO - FUTURE ENHANCEMENT
    # info can be in the files (as events or IMPEDANCE_TIMES)
    # inconsistently, though, and low priority

    # print impedances
    logger.info("impedance measurements:")
    for iimp in range(impedances.shape[0]):
        logger.info({ch: float(imp) for ch, imp in zip(ch_names, impedances[iimp])})

    return ch_names, impedances
