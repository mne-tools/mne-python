#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# from ..._fiff._digitization import _make_dig_points
from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one, _read_segments_file
from ...annotations import annotations_from_events
from ...channels import make_dig_montage
from ...epochs import Epochs
from ...utils import _soft_import, verbose, warn
from ..base import BaseRaw

CURRY_SUFFIX_DATA = [".cdt", ".dat"]
CURRY_SUFFIX_HDR = [".cdt.dpa", ".cdt.dpo", ".dap"]
CURRY_SUFFIX_LABELS = [".cdt.dpa", ".cdt.dpo", ".rs3"]


def _check_curry_filename(fname):
    fname_in = Path(fname)
    fname_out = None
    # try suffixes
    if fname_in.suffix in CURRY_SUFFIX_DATA:
        fname_out = fname_in
    else:
        for data_suff in CURRY_SUFFIX_DATA:
            if fname_in.with_suffix(data_suff).exists():
                fname_out = fname_in.with_suffix(data_suff)
                break
    # final check
    if not fname_out or not fname_out.exists():
        raise FileNotFoundError("no curry data file found (.dat or .cdt)")
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
        baseline=(0, 0),
        metadata=epochmetainfo,
        reject_by_annotation=False,
    )


def _extract_curry_info(fname):
    _soft_import("curryreader", "read file header")

    import curryreader

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
    landmarkslabels = currydata["landmarkslabels"]
    hpimatrix = currydata["hpimatrix"]

    # data
    orig_format = "single"  # curryreader.py always reads float32. is this correct?

    # events
    events = currydata["events"]
    # annotations = currydata[
    #    "annotations"
    # ]  # TODO these dont really seem to correspond to events! what is it?

    # impedance measurements
    # moved to standalone def; see read_impedances_curry
    # impedances = currydata["impedances"]

    # get other essential info not provided by curryreader
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

    print(f"meas_date: {meas_date}")

    # read datatype
    byteorder = (
        re.compile(r"DataByteOrder\s*=\s*[A-Z]+")
        .search(content_hdr)
        .group()
        .split()[-1]
    )
    is_ascii = byteorder == "ASCII"

    # amp info
    # TODO
    # amp_info = (
    #    re.compile(r"AmplifierInfo\s*=.*\n").search(content_hdr).group().split("= ")
    # )

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
        ch_types = [ch_types[i] for i in np.argsort(chaninfile_full) if i not in i_drop]
        units = [units[i] for i in np.argsort(chaninfile_full) if i not in i_drop]

    assert len(ch_types) == len(units) == len(ch_names) == n_ch

    # finetune channel types (e.g. stim, eog etc might be identified by name)
    # TODO?

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
        orig_format,
        orig_units,
        is_ascii,
        cals,
        meas_date,
    )


def _read_annotations_curry(fname, sfreq="auto"):
    r"""Read events from Curry event files.

    Parameters
    ----------
    fname : str
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

    (sfreq_fromfile, _, _, _, _, _, _, _, events, _, _, _, _, _) = _extract_curry_info(
        fname
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
        return annotations_from_events(events, sfreq)
    else:
        warn("no event annotations found")
        return None


def _make_curry_montage(ch_names, ch_types, ch_pos, landmarks, landmarkslabels):
    # scale ch_pos to m?!
    ch_pos /= 1000.0
    # channel locations
    # only take inner coil for MEG (ch_pos[i,:3])
    # TODO what about misc without pos? can they mess things up if unordered?
    assert len(ch_pos) >= (ch_types.count("mag") + ch_types.count("eeg"))
    ch_pos_meg = {
        ch_names[i]: ch_pos[i, :3] for i, t in enumerate(ch_types) if t == "mag"
    }
    ch_pos_eeg = {
        ch_names[i]: ch_pos[i, :3] for i, t in enumerate(ch_types) if t == "eeg"
    }
    # landmarks and headshape
    landmark_dict = dict(zip(landmarkslabels, landmarks))
    for k in ["Nas", "RPA", "LPA"]:
        if k not in landmark_dict.keys():
            landmark_dict[k] = None
    if len(landmarkslabels) > 0:
        hpi_pos = landmarks[
            [i for i, n in enumerate(landmarkslabels) if re.match("HPI[1-99]", n)], :
        ]
    else:
        hpi_pos = None
    if len(landmarkslabels) > 0:
        hsp_pos = landmarks[
            [i for i, n in enumerate(landmarkslabels) if re.match("H[1-99]", n)], :
        ]
    else:
        hsp_pos = None
    # make dig montage for eeg
    mont = None
    if ch_pos.shape[1] in [3, 6]:  # eeg xyz space
        mont = make_dig_montage(
            ch_pos=ch_pos_eeg,
            nasion=landmark_dict["Nas"],
            lpa=landmark_dict["LPA"],
            rpa=landmark_dict["RPA"],
            hsp=hsp_pos,
            hpi=hpi_pos,
            coord_frame="unknown",
        )
        # dig = _make_dig_points(
        #    nasion=landmark_dict["Nas"],
        #    lpa=landmark_dict["LPA"],
        #    rpa=landmark_dict["RPA"],
        #    hpi=hpi_pos,
        #    extra_points=hsp_pos,
        #    dig_ch_pos=ch_pos_eeg,
        #    coord_frame="unknown",
        # )
    else:  # not recorded?
        pass

    # collect pos for meg
    if ch_pos_meg != dict():
        warn("reading MEG sensor locations not yet implemented!")

    return mont


@verbose
def read_raw_curry(
    fname, import_epochs_as_events=False, preload=False, verbose=None
) -> "RawCurry":
    """Read raw data from Curry files.

    Parameters
    ----------
    fname : path-like
        Path to a curry file with extensions ``.dat``, ``.dap``, ``.rs3``,
        ``.cdt``, ``.cdt.dpa``, ``.cdt.cef`` or ``.cef``.
    import_epochs_as_events : bool
        Set to ``True`` if you want to import epoched recordings as continuous ``raw``
        object with event annotations. Only do this if you know your data allows it.
    %(preload)s
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
    rectype = _get_curry_recording_type(fname)

    inst = RawCurry(fname, preload, verbose)
    if rectype in ["epochs", "evoked"]:
        curry_epoch_info = _get_curry_epoch_info(fname)
        if import_epochs_as_events:
            epoch_annotations = annotations_from_events(
                events=curry_epoch_info["events"],
                event_desc={v: k for k, v in curry_epoch_info["event_id"].items()},
                sfreq=inst.info["sfreq"],
            )
            inst.set_annotations(inst.annotations + epoch_annotations)
        else:
            inst = Epochs(
                inst, **curry_epoch_info
            )  # TODO seems to rejects flat channel
            if rectype == "evoked":
                raise NotImplementedError
    return inst


class RawCurry(BaseRaw):
    """Raw object from Curry file.

    Parameters
    ----------
    fname : path-like
        Path to a curry file with extensions ``.dat``, ``.dap``, ``.rs3``,
        ``.cdt``, ``.cdt.dpa``, ``.cdt.cef`` or ``.cef``.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.

    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
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
            orig_format,
            orig_units,
            is_ascii,
            cals,
            meas_date,
        ) = _extract_curry_info(fname)

        # construct info
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

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
            annot = annotations_from_events(events, sfreq)
            self.set_annotations(annot)

        # make montage
        self._set_curry_montage(ch_types, ch_pos, landmarks, landmarkslabels)

        # with self.info._unlock():
        #    self.info['dig'] = mont.dig

        # add HPI data (if present)
        # from curryreader docstring:
        # "HPI-coil measurements matrix (Orion-MEG only) where every row is:
        # [measurementsample, dipolefitflag, x, y, z, deviation]"
        # that's incorrect, though. it seems to be:
        # [sample, dipole_1, x_1,y_1, z_1, dev_1, ..., dipole_n, x_n, ...]
        # for all n coils.
        # TODO
        if not isinstance(hpimatrix, list):
            warn("cHPI data found, but reader not implemented.")
            hpisamples = hpimatrix[:, 0]
            n_coil = int((hpimatrix.shape[1] - 1) / 5)
            hpimatrix = hpimatrix[:, 1:].reshape(hpimatrix.shape[0], n_coil, 5)
            print(f"found {len(hpisamples)} cHPI samples for {n_coil} coils")

    def _set_curry_montage(self, ch_types, ch_pos, landmarks, landmarkslabels):
        assert len(self.info["ch_names"]) == len(ch_types) >= len(ch_pos)

        mont = _make_curry_montage(
            self.info["ch_names"], ch_types, ch_pos, landmarks, landmarkslabels
        )

        # hack the montage in (for MEG chans)
        # TODO change this!
        ch_types_tmp = [ct if ct != "mag" else "eeg" for ct in ch_types]
        self.set_channel_types(dict(zip(self.info["ch_names"], ch_types_tmp)))
        self.set_montage(mont, on_missing="ignore")
        self.set_channel_types(dict(zip(self.info["ch_names"], ch_types)))

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
        Path to a curry file with extensions ``.dat``, ``.dap``, ``.rs3``,
        ``.cdt``, ``.cdt.dpa``, ``.cdt.cef`` or ``.cef``.
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
    currydata = curryreader.read(str(fname), plotdata=0, verbosity=1)

    impedances = currydata["impedances"]
    ch_names = currydata["labels"]

    # try get measurement times
    # TODO possible?
    annotations = currydata[
        "annotations"
    ]  # dont really seem to correspond to events!?!
    for anno in set(annotations):
        if "impedance" in anno.lower():
            print("FOUND IMPEDANCE ANNOTATION!")
            print(f"'{anno}' - N={len([a for a in annotations if a == anno])}")

    # print impedances
    print("impedance measurements:")
    for iimp in range(impedances.shape[0]):
        print({ch: float(imp) for ch, imp in zip(ch_names, impedances[iimp])})

    return ch_names, impedances
