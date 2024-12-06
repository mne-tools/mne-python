"""Conversion tool from BrainVision EEG to FIF."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import configparser
import os
import os.path as op
import re
from datetime import datetime, timezone
from io import StringIO

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _empty_info
from ..._fiff.utils import _mult_cal_one, _read_segments_file
from ...annotations import Annotations, read_annotations
from ...channels import make_dig_montage
from ...defaults import HEAD_SIZE_DEFAULT
from ...transforms import _sph_to_cart
from ...utils import _DefaultEventParser, fill_doc, logger, verbose, warn
from ..base import BaseRaw


@fill_doc
class RawBrainVision(BaseRaw):
    """Raw object from Brain Vision EEG file.

    Parameters
    ----------
    vhdr_fname : path-like
        Path to the EEG header file.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the header file.
        Default is ``('HEOGL', 'HEOGR', 'VEOGb')``.
    misc : list or tuple of str | ``'auto'``
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the header file. If ``'auto'``, units in header file are used for
        inferring misc channels. Default is ``'auto'``.
    scale : float
        The scaling factor for EEG data. Unless specified otherwise by
        header file, units are in microvolts. Default scale factor is 1.
    ignore_marker_types : bool
        If ``True``, ignore marker types and only use marker descriptions. Default is
        ``False``.

        .. versionadded:: 1.8
    %(preload)s
    %(verbose)s

    Attributes
    ----------
    impedances : dict
        A dictionary of all electrodes and their impedances.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.

    Notes
    -----
    If the BrainVision header file contains impedance measurements, these may be
    accessed using ``raw.impedances`` after reading using this function. However,
    this attribute will NOT be available after a save and re-load of the data.
    That is, it is only available when reading data directly from the BrainVision
    header file.

    BrainVision markers consist of a type and a description (in addition to other fields
    like onset and duration). In contrast, annotations in MNE only have a description.
    Therefore, a BrainVision marker of type "Stimulus" and description "S  1" will be
    converted to an annotation "Stimulus/S  1" by default. If you want to ignore the
    type and instead only use the description, set ``ignore_marker_types=True``, which
    will convert the same marker to an annotation "S  1".
    """

    _extra_attributes = ("impedances",)

    @verbose
    def __init__(
        self,
        vhdr_fname,
        eog=("HEOGL", "HEOGR", "VEOGb"),
        misc="auto",
        scale=1.0,
        ignore_marker_types=False,
        preload=False,
        verbose=None,
    ):  # noqa: D107
        # Channel info and events
        logger.info(f"Extracting parameters from {vhdr_fname}...")
        hdr_fname = op.abspath(vhdr_fname)
        ext = op.splitext(hdr_fname)[-1]
        ahdr_format = True if ext == ".ahdr" else False
        (
            info,
            data_fname,
            fmt,
            order,
            n_samples,
            mrk_fname,
            montage,
            orig_units,
        ) = _get_hdr_info(hdr_fname, eog, misc, scale)

        with open(data_fname, "rb") as f:
            if isinstance(fmt, dict):  # ASCII, this will be slow :(
                if order == "F":  # multiplexed, channels in columns
                    n_skip = 0
                    for ii in range(int(fmt["skiplines"])):
                        n_skip += len(f.readline())
                    offsets = np.cumsum([n_skip] + [len(line) for line in f])
                    n_samples = len(offsets) - 1
                elif order == "C":  # vectorized, channels, in rows
                    raise NotImplementedError()
            else:
                n_data_ch = int(info["nchan"])
                f.seek(0, os.SEEK_END)
                n_samples = f.tell()
                dtype_bytes = _fmt_byte_dict[fmt]
                offsets = None
                n_samples = n_samples // (dtype_bytes * n_data_ch)

        orig_format = "single" if isinstance(fmt, dict) else fmt
        raw_extras = dict(offsets=offsets, fmt=fmt, order=order, n_samples=n_samples)
        super().__init__(
            info,
            last_samps=[n_samples - 1],
            filenames=[data_fname],
            orig_format=orig_format,
            preload=preload,
            verbose=verbose,
            raw_extras=[raw_extras],
            orig_units=orig_units,
        )

        self.set_montage(montage)

        settings, _, _, _ = _aux_hdr_info(hdr_fname)
        split_settings = settings.splitlines()
        self.impedances = _parse_impedance(split_settings, self.info["meas_date"])

        # Get annotations from marker file
        annots = read_annotations(
            mrk_fname, info["sfreq"], ignore_marker_types=ignore_marker_types
        )
        self.set_annotations(annots)

        # Drop the fake ahdr channel if needed
        if ahdr_format:
            self.drop_channels(_AHDR_CHANNEL_NAME)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        # read data
        n_data_ch = self._raw_extras[fi]["orig_nchan"]
        fmt = self._raw_extras[fi]["fmt"]
        if self._raw_extras[fi]["order"] == "C":
            _read_segments_c(self, data, idx, fi, start, stop, cals, mult)
        elif isinstance(fmt, str):
            dtype = _fmt_dtype_dict[fmt]
            _read_segments_file(
                self,
                data,
                idx,
                fi,
                start,
                stop,
                cals,
                mult,
                dtype=dtype,
                n_channels=n_data_ch,
            )
        else:
            offsets = self._raw_extras[fi]["offsets"]
            with open(self.filenames[fi], "rb") as fid:
                fid.seek(offsets[start])
                block = np.empty((n_data_ch, stop - start))
                for ii in range(stop - start):
                    line = fid.readline().decode("ASCII")
                    line = line.strip()

                    # Not sure why we special-handle the "," character here,
                    # but let's just keep this for historical and backward-
                    # compat reasons
                    if (
                        isinstance(fmt, dict)
                        and "decimalsymbol" in fmt
                        and fmt["decimalsymbol"] != "."
                    ):
                        line = line.replace(",", ".")

                    if " " in line:
                        line_data = line.split()
                    elif "," in line:
                        # likely exported from BrainVision Analyzer?
                        line_data = line.split(",")
                    else:
                        raise RuntimeError(
                            "Unknown BrainVision data format encountered. "
                            "Please contact the MNE-Python developers."
                        )

                    block[:n_data_ch, ii] = [float(part) for part in line_data]
            _mult_cal_one(data, block, idx, cals, mult)


def _read_segments_c(raw, data, idx, fi, start, stop, cals, mult):
    """Read chunk of vectorized raw data."""
    n_samples = raw._raw_extras[fi]["n_samples"]
    fmt = raw._raw_extras[fi]["fmt"]
    dtype = _fmt_dtype_dict[fmt]
    n_bytes = _fmt_byte_dict[fmt]
    n_channels = raw._raw_extras[fi]["orig_nchan"]
    block = np.zeros((n_channels, stop - start))
    with open(raw.filenames[fi], "rb", buffering=0) as fid:
        ids = np.arange(idx.start, idx.stop) if isinstance(idx, slice) else idx
        for ch_id in ids:
            fid.seek(start * n_bytes + ch_id * n_bytes * n_samples)
            block[ch_id] = np.fromfile(fid, dtype, stop - start)
    _mult_cal_one(data, block, idx, cals, mult)


def _read_mrk(fname, ignore_marker_types=False):
    """Read annotations from a vmrk/amrk file.

    Parameters
    ----------
    fname : str
        vmrk/amrk file to be read.
    ignore_marker_types : bool
        If True, ignore marker types and only use marker descriptions. Default is False.

    Returns
    -------
    onset : array, shape (n_annots,)
        The onsets in seconds.
    duration : array, shape (n_annots,)
        The onsets in seconds.
    description : array, shape (n_annots,)
        The description of each annotation.
    date_str : str
        The recording time as a string. Defaults to empty string if no
        recording time is found.
    """
    # read marker file
    with open(fname, "rb") as fid:
        txt = fid.read()

    # we don't actually need to know the coding for the header line.
    # the characters in it all belong to ASCII and are thus the
    # same in Latin-1 and UTF-8
    header = txt.decode("ascii", "ignore").split("\n")[0].strip()
    _check_bv_version(header, "marker")

    # although the markers themselves are guaranteed to be ASCII (they
    # consist of numbers and a few reserved words), we should still
    # decode the file properly here because other (currently unused)
    # blocks, such as that the filename are specifying are not
    # guaranteed to be ASCII.

    try:
        # if there is an explicit codepage set, use it
        # we pretend like it's ascii when searching for the codepage
        cp_setting = re.search(
            "Codepage=(.+)", txt.decode("ascii", "ignore"), re.IGNORECASE & re.MULTILINE
        )
        codepage = "utf-8"
        if cp_setting:
            codepage = cp_setting.group(1).strip()
        # BrainAmp Recorder also uses ANSI codepage
        # an ANSI codepage raises a LookupError exception
        # python recognize ANSI decoding as cp1252
        if codepage == "ANSI":
            codepage = "cp1252"
        txt = txt.decode(codepage)
    except UnicodeDecodeError:
        # if UTF-8 (new standard) or explicit codepage setting fails,
        # fallback to Latin-1, which is Windows default and implicit
        # standard in older recordings
        txt = txt.decode("latin-1")

    # extract Marker Infos block
    m = re.search(r"\[Marker Infos\]", txt, re.IGNORECASE)
    if not m:
        return np.array(list()), np.array(list()), np.array(list()), ""

    mk_txt = txt[m.end() :]
    m = re.search(r"^\[.*\]$", mk_txt)
    if m:
        mk_txt = mk_txt[: m.start()]

    # extract event information
    items = re.findall(r"^Mk\d+=(.*)", mk_txt, re.MULTILINE)
    onset, duration, description = list(), list(), list()
    date_str = ""
    for info in items:
        info_data = info.split(",")
        mtype, mdesc, this_onset, this_duration = info_data[:4]
        # commas in mtype and mdesc are handled as "\1". convert back to comma
        mtype = mtype.replace(r"\1", ",")
        mdesc = mdesc.replace(r"\1", ",")
        if date_str == "" and len(info_data) == 5 and mtype == "New Segment":
            # to handle the origin of time and handle the presence of multiple
            # New Segment annotations. We only keep the first one that is
            # different from an empty string for date_str.
            date_str = info_data[-1]

        this_duration = int(this_duration) if this_duration.isdigit() else 0
        duration.append(this_duration)
        onset.append(int(this_onset) - 1)  # BV is 1-indexed, not 0-indexed
        if not ignore_marker_types:
            description.append(mtype + "/" + mdesc)
        else:
            description.append(mdesc)

    return np.array(onset), np.array(duration), np.array(description), date_str


def _read_annotations_brainvision(fname, sfreq="auto", ignore_marker_types=False):
    """Create Annotations from BrainVision vmrk/amrk.

    This function reads a .vmrk or .amrk file and creates an :class:`mne.Annotations`
    object.

    Parameters
    ----------
    fname : str | object
        The path to the .vmrk/.amrk file.
    sfreq : float | 'auto'
        The sampling frequency in the file. This is necessary because Annotations are
        expressed in seconds and vmrk/amrk files are in samples. If set to 'auto' then
        the sfreq is taken from the .vhdr/.ahdr file with the same name (without file
        extension). So data.vmrk/amrk looks for sfreq in data.vhdr or, if it does not
        exist, in data.ahdr.
    ignore_marker_types : bool
        If True, ignore marker types and only use marker descriptions. Default is False.

    Returns
    -------
    annotations : instance of Annotations
        The annotations present in the file.
    """
    onset, duration, description, date_str = _read_mrk(
        fname, ignore_marker_types=ignore_marker_types
    )
    orig_time = _str_to_meas_date(date_str)

    if sfreq == "auto":
        hdr_fname = op.splitext(fname)[0] + ".vhdr"
        # if vhdr file does not exist assume that the format is ahdr
        if not op.exists(hdr_fname):
            hdr_fname = op.splitext(fname)[0] + ".ahdr"
        logger.info(f"Finding 'sfreq' from header file: {hdr_fname}")
        _, _, _, info = _aux_hdr_info(hdr_fname)
        sfreq = info["sfreq"]

    onset = np.array(onset, dtype=float) / sfreq
    duration = np.array(duration, dtype=float) / sfreq
    annotations = Annotations(
        onset=onset, duration=duration, description=description, orig_time=orig_time
    )
    return annotations


def _check_bv_version(header, kind):
    """Check the header version."""
    _data_err = (
        "MNE-Python currently only supports %s versions 1.0 and 2.0, got unparsable "
        "%r. Contact MNE-Python developers for support."
    )
    # optional space, optional Core or V-Amp, optional Exchange,
    # Version/Header, optional comma, 1/2
    _data_re = r"Brain ?Vision( Core| V-Amp)? Data( Exchange)? %s File,? Version %s\.0"

    assert kind in ("header", "marker")

    for version in range(1, 3):
        this_re = _data_re % (kind.capitalize(), version)
        if re.search(this_re, header) is not None:
            return version
    else:
        if header == "":
            warn(f"Missing header in {kind} file.")
        else:
            warn(_data_err % (kind, header))


_orientation_dict = dict(MULTIPLEXED="F", VECTORIZED="C")
_fmt_dict = dict(INT_16="short", INT_32="int", IEEE_FLOAT_32="single")
_fmt_byte_dict = dict(short=2, int=4, single=4)
_fmt_dtype_dict = dict(short="<i2", int="<i4", single="<f4")
_unit_dict = {
    "V": 1.0,  # V stands for Volt
    "µV": 1e-6,
    "uV": 1e-6,
    "mV": 1e-3,
    "nV": 1e-9,
    "C": 1,  # C stands for Celsius
    "°C": 1,  # degrees Celsius
    "n/a": 1,  # unit "not available" (or applicable)
    "µS": 1e-6,  # S stands for Siemens
    "uS": 1e-6,
    "ARU": 1,  # ARU is the unity for the breathing data
    "S": 1,
    "N": 1,
}  # Newton


def _str_to_meas_date(date_str):
    date_str = date_str.strip()

    if date_str in ["", "0", "00000000000000000000"]:
        return None

    # these calculations are in naive time but should be okay since
    # they are relative (subtraction below)
    try:
        meas_date = datetime.strptime(date_str, "%Y%m%d%H%M%S%f")
    except ValueError as e:
        if "does not match format" in str(e):
            return None
        else:
            raise

    meas_date = meas_date.replace(tzinfo=timezone.utc)
    return meas_date


def _aux_hdr_info(hdr_fname):
    """Aux function for _get_hdr_info."""
    with open(hdr_fname, "rb") as f:
        # extract the first section to resemble a cfg
        header = f.readline()
        codepage = "utf-8"
        # we don't actually need to know the coding for the header line.
        # the characters in it all belong to ASCII and are thus the
        # same in Latin-1 and UTF-8
        header = header.decode("ascii", "ignore").strip()
        _check_bv_version(header, "header")

        settings = f.read()
        try:
            # if there is an explicit codepage set, use it
            # we pretend like it's ascii when searching for the codepage
            cp_setting = re.search(
                "Codepage=(.+)",
                settings.decode("ascii", "ignore"),
                re.IGNORECASE & re.MULTILINE,
            )
            if cp_setting:
                codepage = cp_setting.group(1).strip()
            # BrainAmp Recorder also uses ANSI codepage
            # an ANSI codepage raises a LookupError exception
            # python recognize ANSI decoding as cp1252
            if codepage == "ANSI":
                codepage = "cp1252"
            settings = settings.decode(codepage)
        except UnicodeDecodeError:
            # if UTF-8 (new standard) or explicit codepage setting fails,
            # fallback to Latin-1, which is Windows default and implicit
            # standard in older recordings
            settings = settings.decode("latin-1")

    if settings.find("[Comment]") != -1:
        params, settings = settings.split("[Comment]")
    else:
        params, settings = settings, ""
    cfg = configparser.ConfigParser(interpolation=None)
    with StringIO(params) as fid:
        cfg.read_file(fid)

    # get sampling info
    # Sampling interval is given in microsec
    cinfostr = "Common Infos"
    if not cfg.has_section(cinfostr):
        cinfostr = "Common infos"  # NeurOne BrainVision export workaround

    # get sampling info
    # Sampling interval is given in microsec
    sfreq = 1e6 / cfg.getfloat(cinfostr, "SamplingInterval")
    info = _empty_info(sfreq)
    info._unlocked = False
    return settings, cfg, cinfostr, info


@fill_doc
def _get_hdr_info(hdr_fname, eog, misc, scale):
    """Extract all the information from the header file.

    Parameters
    ----------
    hdr_fname : str
        Raw EEG header to be read.
    eog : list of str
        Names of channels that should be designated EOG channels. Names should
        correspond to the header file.
    misc : list or tuple of str | 'auto'
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes in the
        header file. If 'auto', units in header file are used for inferring
        misc channels. Default is ``'auto'``.
    scale : float
        The scaling factor for EEG data. Unless specified otherwise by
        header file, units are in microvolts. Default scale factor is 1.

    Returns
    -------
    %(info_not_none)s
    data_fname : str
        Path to the binary data file.
    fmt : str
        The format of the binary data file.
    order : str
        Orientation of the binary data.
    n_samples : int
        Number of data points in the binary data file.
    mrk_fname : str
        Path to the marker file.
    montage : DigMontage
        Coordinates of the channels, if present in the header file.
    orig_units : dict
        Dictionary mapping channel names to their units as specified in
        the header file. Example: {'FC1': 'nV'}
    """
    scale = float(scale)
    ext = op.splitext(hdr_fname)[-1]
    ahdr_format = ext == ".ahdr"
    if ext not in (".vhdr", ".ahdr"):
        raise OSError(
            "The header file must be given to read the data, "
            f"not a file with extension '{ext}'."
        )

    settings, cfg, cinfostr, info = _aux_hdr_info(hdr_fname)
    info._unlocked = True

    order = cfg.get(cinfostr, "DataOrientation")
    if order not in _orientation_dict:
        raise NotImplementedError(f"Data Orientation {order} is not supported")
    order = _orientation_dict[order]

    data_format = cfg.get(cinfostr, "DataFormat")
    if data_format == "BINARY":
        fmt = cfg.get("Binary Infos", "BinaryFormat")
        if fmt not in _fmt_dict:
            raise NotImplementedError(f"Datatype {fmt} is not supported")
        fmt = _fmt_dict[fmt]
    else:
        if order == "C":  # channels in rows
            raise NotImplementedError(
                "BrainVision files with ASCII data in "
                "vectorized order (i.e. channels in rows"
                ") are not supported yet."
            )
        fmt = {key: cfg.get("ASCII Infos", key) for key in cfg.options("ASCII Infos")}

    # locate EEG binary file and marker file for the stim channel
    path = op.dirname(hdr_fname)
    data_fname = op.join(path, cfg.get(cinfostr, "DataFile"))
    mrk_fname = op.join(path, cfg.get(cinfostr, "MarkerFile"))

    # Try to get measurement date from marker file
    # Usually saved with a marker "New Segment", see BrainVision documentation
    regexp = r"^Mk\d+=New Segment,.*,\d+,\d+,-?\d+,(\d{20})$"
    with open(mrk_fname) as tmp_mrk_f:
        lines = tmp_mrk_f.readlines()

    for line in lines:
        match = re.findall(regexp, line.strip())
        # Always take first measurement date we find
        if match:
            date_str = match[0]
            info["meas_date"] = _str_to_meas_date(date_str)
            break
    else:
        info["meas_date"] = None

    # load channel labels
    nchan = cfg.getint(cinfostr, "NumberOfChannels")
    if ahdr_format:
        # add one fake channel for ahdr format
        nchan += 1
    n_samples = None
    if order == "C":
        try:
            n_samples = cfg.getint(cinfostr, "DataPoints")
        except configparser.NoOptionError:
            warn(
                "No info on DataPoints found. Inferring number of "
                "samples from the data file size."
            )
            with open(data_fname, "rb") as fid:
                fid.seek(0, 2)
                n_bytes = fid.tell()
                n_samples = n_bytes // _fmt_byte_dict[fmt] // nchan

    ch_names = [""] * nchan
    cals = np.empty(nchan)
    ranges = np.empty(nchan)
    cals.fill(np.nan)
    ch_dict = dict()
    misc_chs = dict()
    orig_units = dict()
    for chan, props in cfg.items("Channel Infos"):
        n = int(re.findall(r"ch(\d+)", chan)[0]) - 1
        props = props.split(",")

        # default to µV, following the BV specs; the unit is only allowed to be
        # something else if explicitly stated (cf. EEGLAB export below)
        if len(props) < 4:
            # deal with older files, which have no unit property
            props += ("µV",)
        elif props[3] == "":
            # deal with files where the unit property is simply empty, which
            # are created e.g. by PyCorder
            props[3] = "µV"

        name, _, resolution, unit = props[:4]
        # in BrainVision, commas in channel names are encoded as "\1"
        name = name.replace(r"\1", ",")
        ch_dict[chan] = name
        ch_names[n] = name
        if resolution == "":
            if not unit:  # For truncated vhdrs (e.g. EEGLAB export)
                resolution = 0.000001
            else:
                resolution = 1.0  # for files with units specified, but not res
        unit = unit.replace("\xc2", "")  # Remove unwanted control characters
        orig_units[name] = unit  # Save the original units to expose later
        cals[n] = float(resolution)
        ranges[n] = _unit_dict.get(unit, 1) * scale
        if unit not in ("V", "mV", "µV", "uV", "nV"):
            misc_chs[name] = FIFF.FIFF_UNIT_CEL if unit == "C" else FIFF.FIFF_UNIT_NONE
    if ahdr_format:
        ch_dict[_AHDR_CHANNEL_NAME] = _AHDR_CHANNEL_NAME
        ch_names[-1] = _AHDR_CHANNEL_NAME
        orig_units[_AHDR_CHANNEL_NAME] = "V"
        cals[-1] = 1.0
        ranges[-1] = 1

    misc = list(misc_chs.keys()) if misc == "auto" else misc

    # create montage: 'Coordinates' section in VHDR/AHDR file corresponds to
    # "BVEF" BrainVision Electrode File. The data are based on BrainVision
    # Analyzer coordinate system: Defined between standard electrode positions:
    # X-axis from T7 to T8, Y-axis from Oz to Fpz, Z-axis orthogonal from
    # XY-plane through Cz, fit to a sphere if idealized (when radius=1),
    # specified in mm
    montage = None
    if cfg.has_section("Coordinates"):
        montage_pos = list()
        montage_names = list()
        to_misc = list()
        # Go through channels
        for ch in cfg.items("Coordinates"):
            ch_name = ch_dict[ch[0]]
            montage_names.append(ch_name)
            # 1: radius, 2: theta, 3: phi
            rad, theta, phi = (float(c) for c in ch[1].split(","))
            pol = np.deg2rad(theta)
            az = np.deg2rad(phi)
            # Coordinates could be "idealized" (spherical head model)
            if rad == 1:
                # scale up to realistic head radius: *1000 to convert m to mm
                rad *= HEAD_SIZE_DEFAULT * 1000
            pos = _sph_to_cart(np.array([[rad, az, pol]]))[0]
            if (pos == 0).all() and ch_name not in list(eog) + misc:
                to_misc.append(ch_name)
            montage_pos.append(pos)
        # Make a montage, normalizing from BrainVision units "mm" to "m", the
        # unit used for montages in MNE
        montage_pos = np.array(montage_pos) / 1e3
        montage = make_dig_montage(
            ch_pos=dict(zip(montage_names, montage_pos)), coord_frame="head"
        )
        if len(to_misc) > 0:
            misc += to_misc
            warn(
                f"No coordinate information found for channels {to_misc}. Setting "
                "channel types to misc. To avoid this warning, set channel types "
                "explicitly."
            )

    if np.isnan(cals).any():
        raise RuntimeError("Missing channel units")

    # Attempts to extract filtering info from header. If not found, both are
    # set to zero.
    settings = settings.splitlines()
    idx = None

    if "Channels" in settings:
        idx = settings.index("Channels")
        settings = settings[idx + 1 :]
        hp_col, lp_col = 4, 5
        for idx, setting in enumerate(settings):
            if re.match(r"#\s+Name", setting):
                break
            else:
                idx = None

    # If software filters are active, then they override the hardware setup
    # But we still want to be able to double check the channel names
    # for alignment purposes, we keep track of the hardware setting idx
    idx_amp = idx
    filter_list_has_ch_name = True

    if "S o f t w a r e  F i l t e r s" in settings:
        idx = settings.index("S o f t w a r e  F i l t e r s")
        for idx, setting in enumerate(settings[idx + 1 :], idx + 1):
            if re.match(r"#\s+Low Cutoff", setting):
                hp_col, lp_col = 1, 2
                filter_list_has_ch_name = False
                warn(
                    "Online software filter detected. Using software "
                    "filter settings and ignoring hardware values"
                )
                break
            else:
                idx = idx_amp

    if idx:
        lowpass = []
        highpass = []

        # for newer BV files, the unit is specified for every channel
        # separated by a single space, while for older files, the unit is
        # specified in the column headers
        divider = r"\s+"
        if "Resolution / Unit" in settings[idx]:
            shift = 1  # shift for unit
        else:
            shift = 0

        # Extract filter units and convert from seconds to Hz if necessary.
        # this cannot be done as post-processing as the inverse t-f
        # relationship means that the min/max comparisons don't make sense
        # unless we know the units.
        #
        # For reasoning about the s to Hz conversion, see this reference:
        # `Ebersole, J. S., & Pedley, T. A. (Eds.). (2003).
        # Current practice of clinical electroencephalography.
        # Lippincott Williams & Wilkins.`, page 40-41
        header = re.split(r"\s\s+", settings[idx])
        hp_s = "[s]" in header[hp_col]
        lp_s = "[s]" in header[lp_col]

        for i, ch in enumerate(ch_names, 1):
            if ahdr_format and i == len(ch_names) and ch == _AHDR_CHANNEL_NAME:
                break
            # double check alignment with channel by using the hw settings
            if idx == idx_amp:
                line_amp = settings[idx + i]
            else:
                line_amp = settings[idx_amp + i]
            assert line_amp.find(ch) > -1

            # Correct shift for channel names with spaces
            # Header already gives 1 therefore has to be subtracted
            if filter_list_has_ch_name:
                ch_name_parts = re.split(divider, ch)
                real_shift = shift + len(ch_name_parts) - 1
            else:
                real_shift = shift

            line = re.split(divider, settings[idx + i])
            highpass.append(line[hp_col + real_shift])
            lowpass.append(line[lp_col + real_shift])

        if len(highpass) == 0:
            pass
        elif len(set(highpass)) == 1:
            if highpass[0] in ("NaN", "Off"):
                pass  # Placeholder for future use. Highpass set in _empty_info
            elif highpass[0] == "DC":
                info["highpass"] = 0.0
            else:
                info["highpass"] = float(highpass[0])
                if hp_s:
                    # filter time constant t [secs] to Hz conversion: 1/2*pi*t
                    info["highpass"] = 1.0 / (2 * np.pi * info["highpass"])

        else:
            heterogeneous_hp_filter = True
            if hp_s:
                # We convert channels with disabled filters to having
                # highpass relaxed / no filters
                highpass = [
                    float(filt) if filt not in ("NaN", "Off", "DC") else np.inf
                    for filt in highpass
                ]
                info["highpass"] = np.max(np.array(highpass, dtype=np.float64))
                # Conveniently enough 1 / np.inf = 0.0, so this works for
                # DC / no highpass filter
                # filter time constant t [secs] to Hz conversion: 1/2*pi*t
                info["highpass"] = 1.0 / (2 * np.pi * info["highpass"])

                # not exactly the cleanest use of FP, but this makes us
                # more conservative in *not* warning.
                if info["highpass"] == 0.0 and len(set(highpass)) == 1:
                    # not actually heterogeneous in effect
                    # ... just heterogeneously disabled
                    heterogeneous_hp_filter = False
            else:
                highpass = [
                    float(filt) if filt not in ("NaN", "Off", "DC") else 0.0
                    for filt in highpass
                ]
                info["highpass"] = np.min(np.array(highpass, dtype=np.float64))
                if info["highpass"] == 0.0 and len(set(highpass)) == 1:
                    # not actually heterogeneous in effect
                    # ... just heterogeneously disabled
                    heterogeneous_hp_filter = False

            if heterogeneous_hp_filter:
                warn(
                    "Channels contain different highpass filters. "
                    f"Lowest (weakest) filter setting ({info['highpass']:0.2f} Hz) "
                    "will be stored."
                )

        if len(lowpass) == 0:
            pass
        elif len(set(lowpass)) == 1:
            if lowpass[0] in ("NaN", "Off", "0"):
                pass  # Placeholder for future use. Lowpass set in _empty_info
            else:
                info["lowpass"] = float(lowpass[0])
                if lp_s:
                    # filter time constant t [secs] to Hz conversion: 1/2*pi*t
                    info["lowpass"] = 1.0 / (2 * np.pi * info["lowpass"])

        else:
            heterogeneous_lp_filter = True
            if lp_s:
                # We convert channels with disabled filters to having
                # infinitely relaxed / no filters
                lowpass = [
                    float(filt) if filt not in ("NaN", "Off", "0") else 0.0
                    for filt in lowpass
                ]
                info["lowpass"] = np.min(np.array(lowpass, dtype=np.float64))
                try:
                    # filter time constant t [secs] to Hz conversion: 1/2*pi*t
                    info["lowpass"] = 1.0 / (2 * np.pi * info["lowpass"])

                except ZeroDivisionError:
                    if len(set(lowpass)) == 1:
                        # No lowpass actually set for the weakest setting
                        # so we set lowpass to the Nyquist frequency
                        info["lowpass"] = info["sfreq"] / 2.0
                        # not actually heterogeneous in effect
                        # ... just heterogeneously disabled
                        heterogeneous_lp_filter = False
                    else:
                        # no lowpass filter is the weakest filter,
                        # but it wasn't the only filter
                        pass
            else:
                # We convert channels with disabled filters to having
                # infinitely relaxed / no filters
                lowpass = [
                    float(filt) if filt not in ("NaN", "Off", "0") else np.inf
                    for filt in lowpass
                ]
                info["lowpass"] = np.max(np.array(lowpass, dtype=np.float64))

                if np.isinf(info["lowpass"]):
                    # No lowpass actually set for the weakest setting
                    # so we set lowpass to the Nyquist frequency
                    info["lowpass"] = info["sfreq"] / 2.0
                    if len(set(lowpass)) == 1:
                        # not actually heterogeneous in effect
                        # ... just heterogeneously disabled
                        heterogeneous_lp_filter = False

            if heterogeneous_lp_filter:
                # this isn't clean FP, but then again, we only want to provide
                # the Nyquist hint when the lowpass filter was actually
                # calculated from dividing the sampling frequency by 2, so the
                # exact/direct comparison (instead of tolerance) makes sense
                if info["lowpass"] == info["sfreq"] / 2.0:
                    nyquist = ", Nyquist limit"
                else:
                    nyquist = ""
                warn(
                    "Channels contain different lowpass filters. "
                    f"Highest (weakest) filter setting ({info['lowpass']:0.2f} "
                    f"Hz{nyquist}) will be stored."
                )

    # Creates a list of dicts of eeg channels for raw.info
    logger.info("Setting channel info structure...")
    info["chs"] = []
    for idx, ch_name in enumerate(ch_names):
        if ch_name in eog or idx in eog or idx - nchan in eog:
            kind = FIFF.FIFFV_EOG_CH
            coil_type = FIFF.FIFFV_COIL_NONE
            unit = FIFF.FIFF_UNIT_V
        elif ch_name in misc or idx in misc or idx - nchan in misc:
            kind = FIFF.FIFFV_MISC_CH
            coil_type = FIFF.FIFFV_COIL_NONE
            if ch_name in misc_chs:
                unit = misc_chs[ch_name]
            else:
                unit = FIFF.FIFF_UNIT_NONE
        elif ch_name == "STI 014":
            kind = FIFF.FIFFV_STIM_CH
            coil_type = FIFF.FIFFV_COIL_NONE
            unit = FIFF.FIFF_UNIT_NONE
        else:
            kind = FIFF.FIFFV_EEG_CH
            coil_type = FIFF.FIFFV_COIL_EEG
            unit = FIFF.FIFF_UNIT_V
        info["chs"].append(
            dict(
                ch_name=ch_name,
                coil_type=coil_type,
                kind=kind,
                logno=idx + 1,
                scanno=idx + 1,
                cal=cals[idx],
                range=ranges[idx],
                loc=np.full(12, np.nan),
                unit=unit,
                unit_mul=FIFF.FIFF_UNITM_NONE,
                coord_frame=FIFF.FIFFV_COORD_HEAD,
            )
        )

    info._unlocked = False
    info._update_redundant()
    return (info, data_fname, fmt, order, n_samples, mrk_fname, montage, orig_units)


@fill_doc
def read_raw_brainvision(
    vhdr_fname,
    eog=("HEOGL", "HEOGR", "VEOGb"),
    misc="auto",
    scale=1.0,
    ignore_marker_types=False,
    preload=False,
    verbose=None,
) -> RawBrainVision:
    """Reader for Brain Vision EEG file.

    Parameters
    ----------
    vhdr_fname : path-like
        Path to the EEG header file.
    eog : list or tuple of str
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the header file
        Default is ``('HEOGL', 'HEOGR', 'VEOGb')``.
    misc : list or tuple of str | ``'auto'``
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes in the
        header file. If ``'auto'``, units in header file are used for inferring
        misc channels. Default is ``'auto'``.
    scale : float
        The scaling factor for EEG data. Unless specified otherwise by
        header file, units are in microvolts. Default scale factor is 1.
    ignore_marker_types : bool
        If ``True``, ignore marker types and only use marker descriptions. Default is
        ``False``.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawBrainVision
        A Raw object containing BrainVision data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawBrainVision.

    Notes
    -----
    If the BrainVision header file contains impedance measurements, these may be
    accessed using ``raw.impedances`` after reading using this function. However,
    this attribute will NOT be available after a save and re-load of the data.
    That is, it is only available when reading data directly from the BrainVision
    header file.

    BrainVision markers consist of a type and a description (in addition to other fields
    like onset and duration). In contrast, annotations in MNE only have a description.
    Therefore, a BrainVision marker of type "Stimulus" and description "S  1" will be
    converted to an annotation "Stimulus/S  1" by default. If you want to ignore the
    type and instead only use the description, set ``ignore_marker_types=True``, which
    will convert the same marker to an annotation "S  1".
    """
    return RawBrainVision(
        vhdr_fname=vhdr_fname,
        eog=eog,
        misc=misc,
        scale=scale,
        ignore_marker_types=ignore_marker_types,
        preload=preload,
        verbose=verbose,
    )


_BV_EVENT_IO_OFFSETS = {
    "Event/": 0,
    "Stimulus/S": 0,
    "Response/R": 1000,
    "Optic/O": 2000,
}
_OTHER_ACCEPTED_MARKERS = {"New Segment/": 99999, "SyncStatus/Sync On": 99998}
_OTHER_OFFSET = 10001  # where to start "unknown" event_ids
_AHDR_CHANNEL_NAME = "AHDR_CHANNEL"


class _BVEventParser(_DefaultEventParser):
    """Parse standard brainvision events, accounting for non-standard ones."""

    def __call__(self, description):
        """Parse BrainVision event codes (like `Stimulus/S 11`) to ints."""
        offsets = _BV_EVENT_IO_OFFSETS

        maybe_digit = description[-3:].strip()
        kind = description[:-3]
        if maybe_digit.isdigit() and kind in offsets:
            code = int(maybe_digit) + offsets[kind]
        elif description in _OTHER_ACCEPTED_MARKERS:
            code = _OTHER_ACCEPTED_MARKERS[description]
        else:
            code = super().__call__(description, offset=_OTHER_OFFSET)
        return code


def _check_bv_annot(descriptions):
    markers_basename = set([dd.rstrip("0123456789 ") for dd in descriptions])
    bv_markers = set(_BV_EVENT_IO_OFFSETS.keys()).union(
        set(_OTHER_ACCEPTED_MARKERS.keys())
    )
    return len(markers_basename - bv_markers) == 0


def _parse_impedance(settings, recording_date=None):
    """Parse impedances from the header file.

    Parameters
    ----------
    settings : list
        The header settings lines from the VHDR/AHDR file.
    recording_date : datetime.datetime | None
        The date of the recording as extracted from the VMRK/AMRK file.

    Returns
    -------
    impedances : dict
        A dictionary of all electrodes and their impedances.
    """
    ranges = _parse_impedance_ranges(settings)
    impedance_setting_lines = [
        i for i in settings if i.startswith("Impedance [") and i.endswith(" :")
    ]
    impedances = dict()
    if len(impedance_setting_lines) > 0:
        idx = settings.index(impedance_setting_lines[0])
        impedance_setting = impedance_setting_lines[0].split()
        impedance_unit = impedance_setting[1].lstrip("[").rstrip("]")
        impedance_time = None

        # If we have a recording date, we can update it with the time of
        # impedance measurement
        if recording_date is not None:
            meas_time = [int(i) for i in impedance_setting[3].split(":")]
            impedance_time = recording_date.replace(
                hour=meas_time[0],
                minute=meas_time[1],
                second=meas_time[2],
                microsecond=0,
            )
        for setting in settings[idx + 1 :]:
            # Parse channel impedances until we find a line that doesn't start
            # with a channel name and optional +/- polarity for passive elecs
            match = re.match(r"[ a-zA-Z0-9_+-]+:", setting)
            if match:
                channel_name = match.group().rstrip(":")
                channel_imp_line = setting.split()
                imp_as_number = re.findall(r"[-+]?\d*\.\d+|\d+", channel_imp_line[-1])
                channel_impedance = dict(
                    imp=float(imp_as_number[0]) if imp_as_number else np.nan,
                    imp_unit=impedance_unit,
                )
                if impedance_time is not None:
                    channel_impedance.update({"imp_meas_time": impedance_time})

                if channel_name == "Ref" and "Reference" in ranges:
                    channel_impedance.update(ranges["Reference"])
                elif channel_name == "Gnd" and "Ground" in ranges:
                    channel_impedance.update(ranges["Ground"])
                elif "Data" in ranges:
                    channel_impedance.update(ranges["Data"])
                impedances[channel_name] = channel_impedance
            else:
                break
    return impedances


def _parse_impedance_ranges(settings):
    """Parse the selected electrode impedance ranges from the header.

    Parameters
    ----------
    settings : list
        The header settings lines from the VHDR/AHDR file.

    Returns
    -------
    electrode_imp_ranges : dict
        A dictionary of impedance ranges for each type of electrode.
    """
    impedance_ranges = [
        item for item in settings if "Selected Impedance Measurement Range" in item
    ]
    electrode_imp_ranges = dict()
    if impedance_ranges:
        if len(impedance_ranges) == 1:
            img_range = impedance_ranges[0].split()
            for electrode_type in ["Data", "Reference", "Ground"]:
                electrode_imp_ranges[electrode_type] = {
                    "imp_lower_bound": float(img_range[-4]),
                    "imp_upper_bound": float(img_range[-2]),
                    "imp_range_unit": img_range[-1],
                }
        else:
            for electrode_range in impedance_ranges:
                electrode_range = electrode_range.split()
                electrode_imp_ranges[electrode_range[0]] = {
                    "imp_lower_bound": float(electrode_range[6]),
                    "imp_upper_bound": float(electrode_range[8]),
                    "imp_range_unit": electrode_range[9],
                }
    return electrode_imp_ranges
