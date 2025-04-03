#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
from pathlib import Path

import curryreader
import numpy as np

from ..._fiff.meas_info import create_info
from ...annotations import annotations_from_events
from ...channels import make_dig_montage
from ...utils import verbose
from ..base import BaseRaw


@verbose
def read_raw_curry(fname, preload=False, verbose=None) -> "RawCurry":
    """Read raw data from Curry files.

    Parameters
    ----------
    fname : path-like
        Path to a curry file with extensions ``.dat``, ``.dap``, ``.rs3``,
        ``.cdt``, ``.cdt.dpa``, ``.cdt.cef`` or ``.cef``.
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
    return RawCurry(fname, preload, verbose)


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
    def __init__(self, fname, preload=True, verbose=None):
        fname = Path(fname)

        # use curry-python-reader
        try:
            currydata = curryreader.read(str(fname), plotdata=0, verbosity=1)
        except Exception as e:
            raise ValueError(f"file could not be read - {e}")

        # extract info
        sfreq = currydata["info"]["samplingfreq"]
        n_samples = currydata["info"]["samples"]
        n_ch = currydata["info"]["channels"]
        ch_names = currydata["labels"]
        ch_pos = currydata["sensorpos"]
        landmarks = currydata["landmarks"]
        landmarkslabels = currydata["landmarkslabels"]

        # extract data
        orig_format = "single"  # curryreader.py always reads float32. is this correct?

        preload = currydata["data"].T.astype(
            "float64"
        )  # curryreader returns float32, but mne seems to need float64
        events = currydata["events"]
        # annotations = currydata[
        #    "annotations"
        # ]  # dont always seem to correspond to events?!
        # impedances = currydata["impedances"]  # see read_impedances_curry
        # epochinfo = currydata["epochinfo"]  # TODO
        # epochlabels = currydata["epochlabels"]  # TODO
        # hpimatrix = currydata["hpimatrix"]  # TODO

        # extract other essential info not provided by curryreader
        fname_hdr = None
        for hdr_suff in [".cdt.dpa", ".cdt.dpo", ".dap"]:
            if fname.with_suffix(hdr_suff).exists():
                fname_hdr = fname.with_suffix(hdr_suff)

        ch_types, units = [], []
        if fname_hdr:
            ch_groups = fname_hdr.read_text().split("DEVICE_PARAMETERS")[1::2]
            for ch_group in ch_groups:
                ch_group = re.compile(r"\s+").sub(" ", ch_group).strip()
                groupid = ch_group.split()[0]
                unit = ch_group.split("DataUnit = ")[1].split()[0]
                n_ch_group = int(ch_group.split("NumChanThisGroup = ")[1].split()[0])
                ch_type = (
                    "mag"
                    if ("MAG" in groupid)
                    else "misc"
                    if ("OTHER" in groupid)
                    else "eeg"
                )
                # combine info
                ch_types += [ch_type] * n_ch_group
                units += [unit] * n_ch_group

            assert len(ch_types) == len(units) == len(ch_names) == n_ch

        else:
            raise NotImplementedError

        # finetune channel types (e.g. stim, eog etc might be identified by name)
        # TODO?

        # scale data to SI units
        orig_units = dict(zip(ch_names, units))
        for i_ch, unit in enumerate(units):
            if unit == "fT":  # femtoTesla
                preload[i_ch, :] /= 1e15
            elif unit == "uV":  # microVolt
                preload[i_ch, :] /= 1e6
            else:  # leave as is
                pass

        # construct info
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        last_samps = [n_samples - 1]

        # create raw object
        super().__init__(
            info,
            preload,
            filenames=[fname],
            last_samps=last_samps,
            orig_format=orig_format,
            orig_units=orig_units,
            verbose=verbose,
        )

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
        mont = _make_curry_montage(ch_names, ch_pos, landmarks, landmarkslabels)
        self.set_montage(mont, on_missing="ignore")

        # add HPI data (if present)
        # TODO


def _make_curry_montage(ch_names, ch_pos, landmarks, landmarkslabels):
    ch_pos_dict = dict(zip(ch_names, ch_pos))
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

    mont = None
    if ch_pos.shape[1] == 3:  # eeg xyz space
        mont = make_dig_montage(
            ch_pos=ch_pos_dict,
            nasion=landmark_dict["Nas"],
            lpa=landmark_dict["LPA"],
            rpa=landmark_dict["RPA"],
            hsp=hsp_pos,
            hpi=hpi_pos,
            coord_frame="unknown",
        )
    elif ch_pos.shape[1] == 6:  # meg?
        # TODO
        pass
    else:  # not recorded?
        pass

    return mont


def read_impedances_curry(fname):
    """Read impedance measurements from Curry files.

    Parameters
    ----------
    fname : path-like
        Path to a curry file with extensions ``.dat``, ``.dap``, ``.rs3``,
        ``.cdt``, ``.cdt.dpa``, ``.cdt.cef`` or ``.cef``.

    Returns
    -------
    ch_names : list
        A list object containing channel names
    impedances : np.ndarray
        An array containing up to 10 impedance measurements for all recorded channels.

    """
    # use curry-python-reader to load data
    try:
        currydata = curryreader.read(str(fname), plotdata=0, verbosity=1)
    except Exception as e:
        raise ValueError(f"file could not be read - {e}")

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
