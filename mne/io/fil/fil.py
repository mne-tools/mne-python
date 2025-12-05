# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import json
import pathlib

import numpy as np

from ..._fiff._digitization import _make_dig_points
from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _empty_info
from ..._fiff.utils import _read_segments_file
from ..._fiff.write import get_new_file_id
from ...transforms import Transform, apply_trans, get_ras_to_neuromag_trans
from ...utils import _check_fname, fill_doc, verbose, warn
from ..base import BaseRaw
from .sensors import (
    _get_plane_vectors,
    _get_pos_units,
    _refine_sensor_orientation,
    _size2units,
)


@verbose
def read_raw_fil(
    binfile, precision="single", preload=False, *, verbose=None
) -> "RawFIL":
    """Raw object from FIL-OPMEG formatted data.

    Parameters
    ----------
    binfile : path-like
        Path to the MEG data binary (ending in ``'_meg.bin'``).
    precision : str, optional
        How is the data represented? ``'single'`` if 32-bit or ``'double'`` if
        64-bit (default is single).
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawFIL
        The raw data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawFIL.
    """
    return RawFIL(binfile, precision=precision, preload=preload)


@fill_doc
class RawFIL(BaseRaw):
    """Raw object from FIL-OPMEG formatted data.

    Parameters
    ----------
    binfile : path-like
        Path to the MEG data binary (ending in ``'_meg.bin'``).
    precision : str, optional
        How is the data represented? ``'single'`` if 32-bit or
        ``'double'`` if 64-bit (default is single).
    %(preload)s

    Returns
    -------
    raw : instance of RawFIL
        The raw data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawFIL.
    """

    def __init__(self, binfile, precision="single", preload=False):
        if precision == "single":
            dt = np.dtype(">f")
            bps = 4
        else:
            dt = np.dtype(">d")
            bps = 8

        sample_info = dict()
        sample_info["dt"] = dt
        sample_info["bps"] = bps

        files = _get_file_names(binfile)

        chans = _from_tsv(files["chans"])
        nchans = len(chans["name"])
        nsamples = _determine_nsamples(files["bin"], nchans, precision) - 1
        sample_info["nsamples"] = nsamples

        raw_extras = list()
        raw_extras.append(sample_info)

        chans["pos"] = [None] * nchans
        chans["ori"] = [None] * nchans
        if files["positions"].is_file():
            chanpos = _from_tsv(files["positions"])
            nlocs = len(chanpos["name"])
            for ii in range(0, nlocs):
                idx = chans["name"].index(chanpos["name"][ii])
                tmp = np.array(
                    [chanpos["Px"][ii], chanpos["Py"][ii], chanpos["Pz"][ii]]
                )
                chans["pos"][idx] = tmp.astype(np.float64)
                tmp = np.array(
                    [chanpos["Ox"][ii], chanpos["Oy"][ii], chanpos["Oz"][ii]]
                )
                chans["ori"][idx] = tmp.astype(np.float64)
        else:
            warn("No sensor position information found.")

        with open(files["meg"]) as fid:
            meg = json.load(fid)
        info = _compose_meas_info(meg, chans)

        super().__init__(
            info,
            preload,
            filenames=[files["bin"]],
            raw_extras=raw_extras,
            last_samps=[nsamples],
            orig_format=precision,
        )

        if files["coordsystem"].is_file():
            with open(files["coordsystem"]) as fid:
                csys = json.load(fid)
            hc = csys["HeadCoilCoordinates"]

            for key in hc:
                if key.lower() == "lpa":
                    lpa = np.asarray(hc[key])
                elif key.lower() == "rpa":
                    rpa = np.asarray(hc[key])
                elif key.lower().startswith("nas"):
                    nas = np.asarray(hc[key])
                else:
                    warn(f"{key} is not a valid fiducial name!")

            size = np.linalg.norm(nas - rpa)
            unit, sf = _size2units(size)
            # TODO: These are not guaranteed to exist and could lead to a
            # confusing error message, should fix later
            lpa /= sf
            rpa /= sf
            nas /= sf

            t = get_ras_to_neuromag_trans(nas, lpa, rpa)

            # transform fiducial points
            nas = apply_trans(t, nas)
            lpa = apply_trans(t, lpa)
            rpa = apply_trans(t, rpa)

            with self.info._unlock():
                self.info["dig"] = _make_dig_points(
                    nasion=nas, lpa=lpa, rpa=rpa, coord_frame="meg"
                )
        else:
            warn(
                "No fiducials found in files, defaulting sensor array to "
                "FIFFV_COORD_DEVICE, this may cause problems later!"
            )
            t = np.eye(4)

        with self.info._unlock():
            self.info["dev_head_t"] = Transform(
                FIFF.FIFFV_COORD_DEVICE, FIFF.FIFFV_COORD_HEAD, t
            )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        si = self._raw_extras[fi]
        _read_segments_file(
            self, data, idx, fi, start, stop, cals, mult, dtype=si["dt"]
        )


def _convert_channel_info(chans):
    """Convert the imported _channels.tsv into the chs element of raw.info."""
    nmeg = nstim = nmisc = nref = 0

    if not all(p is None for p in chans["pos"]):
        _, sf = _get_pos_units(chans["pos"])

    chs = list()
    for ii in range(len(chans["name"])):
        ch = dict(
            scanno=ii + 1,
            range=1.0,
            cal=1.0,
            loc=np.full(12, np.nan),
            unit_mul=FIFF.FIFF_UNITM_NONE,
            ch_name=chans["name"][ii],
            coil_type=FIFF.FIFFV_COIL_NONE,
        )
        chs.append(ch)

        # create the channel information
        if chans["pos"][ii] is not None:
            r0 = chans["pos"][ii].copy() / sf  # mm to m
            ez = chans["ori"][ii].copy()
            ez = ez / np.linalg.norm(ez)
            ex, ey = _get_plane_vectors(ez)
            ch["loc"] = np.concatenate([r0, ex, ey, ez])

        if chans["type"][ii] == "MEGMAG":
            nmeg += 1
            ch.update(
                logno=nmeg,
                coord_frame=FIFF.FIFFV_COORD_DEVICE,
                kind=FIFF.FIFFV_MEG_CH,
                unit=FIFF.FIFF_UNIT_T,
                coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
            )
        elif chans["type"][ii] == "MEGREFMAG":
            nref += 1
            ch.update(
                logno=nref,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                kind=FIFF.FIFFV_REF_MEG_CH,
                unit=FIFF.FIFF_UNIT_T,
                coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
            )
        elif chans["type"][ii] == "TRIG":
            nstim += 1
            ch.update(
                logno=nstim,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                kind=FIFF.FIFFV_STIM_CH,
                unit=FIFF.FIFF_UNIT_V,
            )
        else:
            nmisc += 1
            ch.update(
                logno=nmisc,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                kind=FIFF.FIFFV_MISC_CH,
                unit=FIFF.FIFF_UNIT_NONE,
            )

        # set the calibration based on the units - MNE expects T units for meg
        # and V for eeg
        if chans["units"][ii] == "fT":
            ch.update(cal=1e-15)
        elif chans["units"][ii] == "pT":
            ch.update(cal=1e-12)
        elif chans["units"][ii] == "nT":
            ch.update(cal=1e-9)
        elif chans["units"][ii] == "mV":
            ch.update(cal=1e3)
        elif chans["units"][ii] == "uV":
            ch.update(cal=1e6)

    return chs


def _compose_meas_info(meg, chans):
    """Create info structure."""
    info = _empty_info(meg["SamplingFrequency"])
    # Collect all the necessary data from the structures read
    info["meas_id"] = get_new_file_id()
    tmp = _convert_channel_info(chans)
    info["chs"] = _refine_sensor_orientation(tmp)
    info["line_freq"] = meg["PowerLineFrequency"]
    info._update_redundant()
    info["bads"] = _read_bad_channels(chans)
    info._unlocked = False
    return info


def _determine_nsamples(bin_fname, nchans, precision):
    """Identify how many temporal samples in a dataset."""
    bsize = bin_fname.stat().st_size
    if precision == "single":
        bps = 4
    else:
        bps = 8
    nsamples = int(bsize / (nchans * bps))
    return nsamples


def _read_bad_channels(chans):
    """Check _channels.tsv file to look for premarked bad channels."""
    bads = list()
    for ii in range(0, len(chans["status"])):
        if chans["status"][ii] == "bad":
            bads.append(chans["name"][ii])
    return bads


def _from_tsv(fname, dtypes=None):
    """Read a tsv file into a dict (which we know is ordered)."""
    data = np.loadtxt(
        fname, dtype=str, delimiter="\t", ndmin=2, comments=None, encoding="utf-8-sig"
    )
    column_names = data[0, :]
    info = data[1:, :]
    data_dict = dict()
    if dtypes is None:
        dtypes = [str] * info.shape[1]
    if not isinstance(dtypes, list | tuple):
        dtypes = [dtypes] * info.shape[1]
    if not len(dtypes) == info.shape[1]:
        raise ValueError(
            f"dtypes length mismatch. Provided: {len(dtypes)}, "
            f"Expected: {info.shape[1]}"
        )
    for i, name in enumerate(column_names):
        data_dict[name] = info[:, i].astype(dtypes[i]).tolist()
    return data_dict


def _get_file_names(binfile):
    """Guess the filenames based on predicted suffixes."""
    binfile = pathlib.Path(
        _check_fname(binfile, overwrite="read", must_exist=True, name="fname")
    )
    if not (binfile.suffix == ".bin" and binfile.stem.endswith("_meg")):
        raise ValueError(f"File must be a filename ending in _meg.bin, got {binfile}")
    files = dict()
    dir_ = binfile.parent
    root = binfile.stem[:-4]  # no _meg
    files["bin"] = dir_ / (root + "_meg.bin")
    files["meg"] = dir_ / (root + "_meg.json")
    files["chans"] = dir_ / (root + "_channels.tsv")
    files["positions"] = dir_ / (root + "_positions.tsv")
    files["coordsystem"] = dir_ / (root + "_coordsystem.json")
    return files
