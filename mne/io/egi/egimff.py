# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""EGI NetStation Load Function."""

import datetime
import fnmatch
import itertools
import math
import os.path as op
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _empty_info, _ensure_meas_date_none_or_dt, create_info
from ..._fiff.proj import setup_proj
from ..._fiff.utils import _create_chs, _mult_cal_one
from ...annotations import Annotations
from ...channels.montage import make_dig_montage
from ...evoked import EvokedArray
from ...utils import _check_fname, _check_option, _soft_import, logger, verbose, warn
from ..base import BaseRaw
from .events import _combine_triggers, _read_events, _triage_include_exclude
from .general import (
    _block_r,
    _extract,
    _get_blocks,
    _get_ep_info,
    _get_gains,
    _get_signalfname,
)

REFERENCE_NAMES = ("VREF", "Vertex Reference")


# TODO: Running list
# - [ ] Add support for reading in the PNS data
# - [ ] Add tutorial for reading calibration data
# - [ ] Add support for reading in the channel status (bad channels)
# - [ ] Replace _read_header with mffpy functions?


def _read_mff(input_fname):
    """Read EGI MFF file."""
    mff_reader = _get_mff_reader(input_fname)
    eeg = _get_eeg_data(mff_reader)
    info = _get_info(mff_reader)
    annotations = _get_annotations(mff_reader, info)
    return eeg, info, annotations


def _get_mff_startdatetime(mff_reader):
    """Get start datetime from mff_reader, with workaround for nanosecond precision bug."""
    try:
        return mff_reader.startdatetime
    except (ValueError, AttributeError):
        # mffpy has a bug parsing timestamps with 9 decimal places (nanoseconds)
        # Workaround: manually parse the timestamp from the info.xml file
        import xml.etree.ElementTree as ET
        info_file = op.join(mff_reader.directory._mffname, "info.xml")
        tree = ET.parse(info_file)
        root = tree.getroot()
        # Handle different XML namespaces by searching for any recordTime element
        time_elem = root.find(".//recordTime") or root.find(".//{*}recordTime")
        if time_elem is None:
            raise
        time_str = time_elem.text
        # Handle timestamps with up to 9 decimal places by truncating to 6
        # e.g., "2017-09-20T09:55:44.072000000+01:00" -> "2017-09-20T09:55:44.072000+01:00"
        # Both formats: +0100 (without colon) and +01:00 (with colon)
        if '+' in time_str or '-' in time_str[-6:]:
            # Truncate nanoseconds in decimal part (keep only 6 digits)
            time_str = re.sub(r'\.(\d{6})\d+([+-])', r'.\1\2', time_str)
        # Python's %z can't always handle colons, so remove them
        time_str = re.sub(r'([+-]\d{2}):(\d{2})$', r'\1\2', time_str)
        return datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f%z')


def _get_mff_reader(input_fname):
    mffpy = _import_mffpy()
    mff_reader = mffpy.Reader(input_fname)
    mff_reader.set_unit("EEG", "V")  # XXX: set PNS unit
    return mff_reader


def _get_montage(mff_reader):
    mffpy = _import_mffpy()
    xml_files = mff_reader.directory.files_by_type[".xml"]
    
    # Read coordinates.xml for fiducial positions
    coords_fname = fnmatch.filter(xml_files, "coordinates")
    coords_sensors = dict()
    if len(coords_fname) == 1:
        with mff_reader.directory.filepointer(coords_fname[0]) as fp:
            coords_content = mffpy.XML.from_file(fp).get_content()
            coords_sensors = coords_content.get("sensors", dict())
    
    n_eeg_channels = mff_reader.num_channels["EEG"]  # XXX: PNS?
    ch_pos = dict()
    hsp_list = []  # Extra headshape points
    lpa, rpa, nasion = None, None, None
    
    # Extract channel positions and fiducials from coordinates.xml
    for ch in coords_sensors.values():
        # XXX: the y coordinate seems to be inverted? Need to investigate
        # Convert from cm to m
        loc = np.array([ch["x"], -(ch["y"]), ch["z"]]) / 100.0
        name = ch.get("name", "None")
        
        # Check if this is a fiducial point
        if name == "Nasion":
            nasion = loc
        elif name == "Left periauricular point":            lpa = loc
        elif name == "Right periauricular point":
            rpa = loc
        elif name in REFERENCE_NAMES or "VREF" in name or "Vertex" in name:
            # Reference electrode can be numbered outside EEG range (e.g., 1001)
            ch_pos[name] = loc
        elif ch["number"] <= n_eeg_channels:
            # EEG channel
            ch_name = name if name != "None" else f"E{ch['number']}"
            ch_pos[ch_name] = loc
    
    # Convert hsp list to array if not empty
    hsp = np.array(hsp_list) if hsp_list else None
    
    montage = make_dig_montage(
        ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, hsp=hsp, coord_frame="unknown"
    )
    return montage


def _get_info(mff_reader):
    montage = _get_montage(mff_reader)
    ch_names = montage.ch_names
    ch_types = ["eeg"] * len(ch_names)  # XXX: refactor this when adding PNS support
    meas_date_orig = _get_mff_startdatetime(mff_reader)
    utc_offset = meas_date_orig.strftime("%z")
    meas_date = meas_date_orig.astimezone(datetime.timezone.utc)
    sfreq = mff_reader.sampling_rates["EEG"]  # XXX: check PNS sfreq?
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage(montage)
    info.set_meas_date(meas_date)
    with info._unlock():
        info["utc_offset"] = utc_offset
    
    # Populate reference location (loc[3:6]) for each EEG channel
    # The reference is VREF (Vertex Reference), which is the last dig point
    if len(info["dig"]) > 0:
        ref_loc = info["dig"][-1]["r"]  # VREF position
        for ch in info["chs"]:
            if ch["kind"] == FIFF.FIFFV_EEG_CH:
                ch["loc"][3:6] = ref_loc
    
    return info


def _get_eeg_data(mff_reader):
    sfreq = mff_reader.sampling_rates["EEG"]  # XXX: check PNS sfreq
    n_channels = mff_reader.num_channels["EEG"]  # Only EEG channels, not all signal types
    epochs = mff_reader.epochs

    data_blocks, start_secs, end_secs = [], [], []
    for epoch in epochs:
        data_chunk, _ = mff_reader.get_physical_samples_from_epoch(epoch)["EEG"]  # XXX
        data_blocks.append(data_chunk)
        start_secs.append(epoch.t0)
        end_secs.append(epoch.t1)

    first_samp = int(start_secs[0] * sfreq)
    # Calculate total samples needed based on actual chunk placements
    max_end_samp = first_samp
    for this_chunk, start in zip(data_blocks, start_secs):
        start_samp = int(start * sfreq)
        end_samp = start_samp + this_chunk.shape[1]
        max_end_samp = max(max_end_samp, end_samp)
    n_samps = max_end_samp - first_samp
    
    eeg = np.zeros((n_channels, n_samps), dtype=np.float64)
    for this_chunk, start in zip(data_blocks, start_secs):
        start_idx = int(start * sfreq) - first_samp
        end_idx = start_idx + this_chunk.shape[1]
        eeg[:, start_idx:end_idx] = this_chunk
    return eeg


def _get_gap_annotations(mff_reader):
    epochs = mff_reader.epochs
    start_secs = [epoch.t0 for epoch in epochs]
    end_secs = [epoch.t1 for epoch in epochs]
    gap_durations = np.array(start_secs[1:]) - np.array(end_secs[:-1])
    descriptions = ["BAD_ACQ_SKIP"] * len(gap_durations)
    gap_onsets = np.array(end_secs[:-1])
    # TODO: Re-enable warning once lazy loading is properly implemented
    # The warning should be raised during data access, not during __init__
    # if len(gap_durations) > 0:
    #     warn(
    #         "Acquisition skips detected. EGI MFF file contains gaps between "
    #         "recording epochs.",
    #         RuntimeWarning,
    #     )
    gap_annots = Annotations(gap_onsets, gap_durations, descriptions)
    return gap_annots


def _get_event_annotations(mff_reader, mne_info):
    mffpy = _import_mffpy()
    xml_files = mff_reader.directory.files_by_type[".xml"]
    events_xmls = fnmatch.filter(xml_files, "Events*")
    if not events_xmls:
        raise RuntimeError("No events found in MFF file.")
    mff_events = {}
    for event_file in events_xmls:
        with mff_reader.directory.filepointer(event_file) as fp:
            categories = mffpy.XML.from_file(fp)
        mff_events[event_file] = categories.get_content()["event"]
    onsets = []
    descriptions = []
    mff_events = list(itertools.chain.from_iterable(mff_events.values()))
    for event in mff_events:
        onset_dt = event["beginTime"].astimezone(datetime.timezone.utc)
        ts = (onset_dt - mne_info["meas_date"]).total_seconds()
        onsets.append(ts)
        # XXX: we could use event["duration"] but it always seems to be 1000ms?
        descriptions.append(event["code"])
    durations = [0] * len(onsets)
    event_annots = Annotations(onsets, durations, descriptions)
    return event_annots


def _get_annotations(mff_reader, mne_info):
    event_annots = _get_event_annotations(mff_reader, mne_info)
    gap_annots = _get_gap_annotations(mff_reader)
    return event_annots + gap_annots


def _read_mff_header(filepath):
    """Read mff header."""
    _soft_import("defusedxml", "reading EGI MFF data")
    from defusedxml.minidom import parse

    all_files = _get_signalfname(filepath)
    eeg_file = all_files["EEG"]["signal"]
    eeg_info_file = all_files["EEG"]["info"]

    info_filepath = op.join(filepath, "info.xml")  # add with filepath
    tags = ["mffVersion", "recordTime"]
    version_and_date = _extract(tags, filepath=info_filepath)
    version = ""
    if len(version_and_date["mffVersion"]):
        version = version_and_date["mffVersion"][0]

    fname = op.join(filepath, eeg_file)
    signal_blocks = _get_blocks(fname)
    epochs = _get_ep_info(filepath)
    summaryinfo = dict(eeg_fname=eeg_file, info_fname=eeg_info_file)
    summaryinfo.update(signal_blocks)
    # sanity check and update relevant values
    record_time = version_and_date["recordTime"][0]
    # e.g.,
    # 2018-07-30T10:47:01.021673-04:00
    # 2017-09-20T09:55:44.072000000+01:00
    g = re.match(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.(\d{6}(?:\d{3})?)[+-]\d{2}:\d{2}",  # noqa: E501
        record_time,
    )
    if g is None:
        raise RuntimeError(f"Could not parse recordTime {repr(record_time)}")
    frac = g.groups()[0]
    assert len(frac) in (6, 9) and all(f.isnumeric() for f in frac)  # regex
    div = 1000 if len(frac) == 6 else 1000000
    for key in ("last_samps", "first_samps"):
        # convert from times in µS to samples
        for ei, e in enumerate(epochs[key]):
            if e % div != 0:
                raise RuntimeError(f"Could not parse epoch time {e}")
            epochs[key][ei] = e // div
        epochs[key] = np.array(epochs[key], np.uint64)
        # I guess they refer to times in milliseconds?
        # What we really need to do here is:
        # epochs[key] *= signal_blocks['sfreq']
        # epochs[key] //= 1000
        # But that multiplication risks an overflow, so let's only multiply
        # by what we need to (e.g., a sample rate of 500 means we can multiply
        # by 1 and divide by 2 rather than multiplying by 500 and dividing by
        # 1000)
        numerator = int(signal_blocks["sfreq"])
        denominator = 1000
        this_gcd = math.gcd(numerator, denominator)
        numerator = numerator // this_gcd
        denominator = denominator // this_gcd
        with np.errstate(over="raise"):
            epochs[key] *= numerator
        epochs[key] //= denominator
        # Should be safe to cast to int now, which makes things later not
        # upbroadcast to float
        epochs[key] = epochs[key].astype(np.int64)
    n_samps_block = signal_blocks["samples_block"].sum()
    n_samps_epochs = (epochs["last_samps"] - epochs["first_samps"]).sum()
    bad = (
        n_samps_epochs != n_samps_block
        or not (epochs["first_samps"] < epochs["last_samps"]).all()
        or not (epochs["first_samps"][1:] >= epochs["last_samps"][:-1]).all()
    )
    if bad:
        raise RuntimeError(
            "EGI epoch first/last samps could not be parsed:\n"
            f'{list(epochs["first_samps"])}\n{list(epochs["last_samps"])}'
        )
    summaryinfo.update(epochs)
    # index which samples in raw are actually readable from disk (i.e., not
    # in a skip)
    disk_samps = np.full(epochs["last_samps"][-1], -1)
    offset = 0
    for first, last in zip(epochs["first_samps"], epochs["last_samps"]):
        n_this = last - first
        disk_samps[first:last] = np.arange(offset, offset + n_this)
        offset += n_this
    summaryinfo["disk_samps"] = disk_samps

    # Add the sensor info.
    sensor_layout_file = op.join(filepath, "sensorLayout.xml")
    sensor_layout_obj = parse(sensor_layout_file)
    summaryinfo["device"] = sensor_layout_obj.getElementsByTagName("name")[
        0
    ].firstChild.data
    sensors = sensor_layout_obj.getElementsByTagName("sensor")
    chan_type = list()
    chan_unit = list()
    n_chans = 0
    numbers = list()  # used for identification
    for sensor in sensors:
        sensortype = int(sensor.getElementsByTagName("type")[0].firstChild.data)
        if sensortype in [0, 1]:
            sn = sensor.getElementsByTagName("number")[0].firstChild.data
            sn = sn.encode()
            numbers.append(sn)
            chan_type.append("eeg")
            chan_unit.append("uV")
            n_chans = n_chans + 1
    if n_chans != summaryinfo["n_channels"]:
        raise RuntimeError(
            "Number of defined channels (%d) did not match the "
            "expected channels (%d)" % (n_chans, summaryinfo["n_channels"])
        )

    # Check presence of PNS data
    pns_names = []
    if "PNS" in all_files:
        pns_fpath = op.join(filepath, all_files["PNS"]["signal"])
        pns_blocks = _get_blocks(pns_fpath)
        pns_samples = pns_blocks["samples_block"]
        signal_samples = signal_blocks["samples_block"]
        same_blocks = np.array_equal(
            pns_samples[:-1], signal_samples[:-1]
        ) and pns_samples[-1] in (signal_samples[-1] - np.arange(2))
        if not same_blocks:
            raise RuntimeError(
                "PNS and signals samples did not match:\n"
                f"{list(pns_samples)}\nvs\n{list(signal_samples)}"
            )

        pns_file = op.join(filepath, "pnsSet.xml")
        pns_obj = parse(pns_file)
        sensors = pns_obj.getElementsByTagName("sensor")
        pns_types = []
        pns_units = []
        for sensor in sensors:
            # sensor number:
            # sensor.getElementsByTagName('number')[0].firstChild.data
            name = sensor.getElementsByTagName("name")[0].firstChild.data
            unit_elem = sensor.getElementsByTagName("unit")[0].firstChild
            unit = ""
            if unit_elem is not None:
                unit = unit_elem.data

            if name == "ECG":
                ch_type = "ecg"
            elif "EMG" in name:
                ch_type = "emg"
            else:
                ch_type = "bio"
            pns_types.append(ch_type)
            pns_units.append(unit)
            pns_names.append(name)

        summaryinfo.update(
            pns_types=pns_types,
            pns_units=pns_units,
            pns_fname=all_files["PNS"]["signal"],
            pns_sample_blocks=pns_blocks,
        )
    summaryinfo.update(
        pns_names=pns_names,
        version=version,
        date=version_and_date["recordTime"][0],
        chan_type=chan_type,
        chan_unit=chan_unit,
        numbers=numbers,
    )

    return summaryinfo


def _read_header(input_fname):
    """Obtain the headers from the file package mff.

    Parameters
    ----------
    input_fname : path-like
        Path for the file

    Returns
    -------
    info : dict
        Main headers set.
    """
    input_fname = str(input_fname)  # cast to str any Paths
    mff_hdr = _read_mff_header(input_fname)
    with open(input_fname + "/signal1.bin", "rb") as fid:
        version = np.fromfile(fid, np.int32, 1)[0]
    """
    the datetime.strptime .f directive (milleseconds)
    will only accept up to 6 digits. if there are more than
    six millesecond digits in the provided timestamp string
    (i.e. because of trailing zeros, as in test_egi_pns.mff)
    then slice both the first 26 elements and the last 6
    elements of the timestamp string to truncate the
    milleseconds to 6 digits and extract the timezone,
    and then piece these together and assign back to mff_hdr['date']
    """
    if len(mff_hdr["date"]) > 32:
        dt, tz = [mff_hdr["date"][:26], mff_hdr["date"][-6:]]
        mff_hdr["date"] = dt + tz

    time_n = datetime.datetime.strptime(mff_hdr["date"], "%Y-%m-%dT%H:%M:%S.%f%z")

    info = dict(
        version=version,
        meas_dt_local=time_n,
        utc_offset=time_n.strftime("%z"),
        gain=0,
        bits=0,
        value_range=0,
    )
    info.update(
        n_categories=0,
        n_segments=1,
        n_events=0,
        event_codes=[],
        category_names=[],
        category_lengths=[],
        pre_baseline=0,
    )
    info.update(mff_hdr)
    return info


def _get_eeg_calibration_info(filepath, egi_info):
    """Calculate calibration info for EEG channels."""
    gains = _get_gains(op.join(filepath, egi_info["info_fname"]))
    if egi_info["value_range"] != 0 and egi_info["bits"] != 0:
        cals = [egi_info["value_range"] / 2 ** egi_info["bits"]] * len(
            egi_info["chan_type"]
        )
    else:
        cal_scales = {"uV": 1e-6, "V": 1}
        cals = [cal_scales[t] for t in egi_info["chan_unit"]]
    if "gcal" in gains:
        cals *= gains["gcal"]
    return cals


def _read_locs(filepath, egi_info, channel_naming):
    """Read channel locations."""
    _soft_import("defusedxml", "reading EGI MFF data")
    from defusedxml.minidom import parse

    fname = op.join(filepath, "coordinates.xml")
    if not op.exists(fname):
        logger.warn("File coordinates.xml not found, not setting channel locations")
        ch_names = [channel_naming % (i + 1) for i in range(egi_info["n_channels"])]
        return ch_names, None
    dig_ident_map = {
        "Left periauricular point": "lpa",
        "Right periauricular point": "rpa",
        "Nasion": "nasion",
    }
    numbers = np.array(egi_info["numbers"])
    coordinates = parse(fname)
    sensors = coordinates.getElementsByTagName("sensor")
    ch_pos = OrderedDict()
    hsp = list()
    nlr = dict()
    ch_names = list()

    for sensor in sensors:
        name_element = sensor.getElementsByTagName("name")[0].firstChild
        num_element = sensor.getElementsByTagName("number")[0].firstChild
        name = (
            channel_naming % int(num_element.data)
            if name_element is None
            else name_element.data
        )
        nr = num_element.data.encode()
        coords = [
            float(sensor.getElementsByTagName(coord)[0].firstChild.data)
            for coord in "xyz"
        ]
        loc = np.array(coords) / 100  # cm -> m
        # create dig entry
        if name in dig_ident_map:
            nlr[dig_ident_map[name]] = loc
        else:
            # id_ is the index of the channel in egi_info['numbers']
            id_ = np.flatnonzero(numbers == nr)
            # if it's not in egi_info['numbers'], it's a headshape point
            if len(id_) == 0:
                hsp.append(loc)
            # not HSP, must be a data or reference channel
            else:
                ch_names.append(name)
                ch_pos[name] = loc
    mon = make_dig_montage(ch_pos=ch_pos, hsp=hsp, **nlr)
    return ch_names, mon


def _add_pns_channel_info(chs, egi_info, ch_names):
    """Add info for PNS channels to channel info dict."""
    for i_ch, ch_name in enumerate(egi_info["pns_names"]):
        idx = ch_names.index(ch_name)
        ch_type = egi_info["pns_types"][i_ch]
        type_to_kind_map = {"ecg": FIFF.FIFFV_ECG_CH, "emg": FIFF.FIFFV_EMG_CH}
        ch_kind = type_to_kind_map.get(ch_type, FIFF.FIFFV_BIO_CH)
        ch_unit = FIFF.FIFF_UNIT_V
        ch_cal = 1e-6
        if egi_info["pns_units"][i_ch] != "uV":
            ch_unit = FIFF.FIFF_UNIT_NONE
            ch_cal = 1.0
        chs[idx].update(
            cal=ch_cal, kind=ch_kind, coil_type=FIFF.FIFFV_COIL_NONE, unit=ch_unit
        )
    return chs


@verbose
def _read_raw_egi_mff(
    input_fname,
    eog=None,
    misc=None,
    include=None,
    exclude=None,
    preload=False,
    channel_naming="E%d",
    *,
    events_as_annotations=True,
    verbose=None,
):
    """Read EGI mff binary as raw object."""
    return RawMff(
        input_fname,
        eog,
        misc,
        include,
        exclude,
        preload,
        channel_naming,
        events_as_annotations=events_as_annotations,
        verbose=verbose,
    )


class RawMff(BaseRaw):
    """RawMff class."""

    _extra_attributes = ("event_id",)

    @verbose
    def __init__(
        self,
        input_fname,
        eog=None,  # XXX: allow user to specify EOG channels?
        misc=None,  # XXX: allow user to specify misc channels?
        include=None,  # XXX: Now We dont create stim channels. Remove this?
        exclude=None,  # XXX: Ditto. But maybe we can exclude events from annots.
        preload=False,  # XXX: Make this work again
        channel_naming="E%d",  # XXX: Do we need to still support this?
        *,
        events_as_annotations=True,  # XXX: This is now the only way. Remove?
        verbose=None,
    ):
        """Init the RawMff class."""
        input_fname = str(
            _check_fname(
                input_fname,
                "read",
                True,
                "input_fname",
                need_dir=True,
            )
        )
        logger.info(f"Reading EGI MFF Header from {input_fname}...")
        eog = [] if eog is None else eog
        misc = [] if misc is None else misc
        egi_info = _read_header(input_fname)

        # Event data (for stim channels and optional STI 014)
        egi_events, egi_info, mff_events = _read_events(input_fname, egi_info)
        event_codes = list(egi_info["event_codes"])
        include = _triage_include_exclude(include, exclude, egi_events, egi_info)
        if not events_as_annotations:
            included_codes = [e for e in event_codes if e in include]
            if len(included_codes):
                events_ids = []
                next_id = 1
                for code in included_codes:
                    match = re.match(r"DIN(\d+)$", code)
                    if match is not None:
                        events_ids.append(int(match.group(1)))
                    else:
                        while next_id in events_ids:
                            next_id += 1
                        events_ids.append(next_id)
                        next_id += 1
                events_ids = np.array(events_ids, int)
                egi_info["new_trigger"] = _combine_triggers(
                    egi_events[[c in include for c in event_codes]], remapping=events_ids
                )
                self.event_id = dict(zip(included_codes, events_ids))
            else:
                egi_info["new_trigger"] = None
                self.event_id = None
            if egi_info["new_trigger"] is not None:
                egi_events = np.vstack([egi_events, egi_info["new_trigger"]])
        else:
            self.event_id = None
            egi_info["new_trigger"] = None
        assert egi_events.shape[1] == egi_info["last_samps"][-1]

        # Info and channels
        meas_dt_utc = egi_info["meas_dt_local"].astimezone(datetime.timezone.utc)
        info = _empty_info(egi_info["sfreq"])
        info["meas_date"] = _ensure_meas_date_none_or_dt(meas_dt_utc)
        info["utc_offset"] = egi_info["utc_offset"]
        info["device_info"] = dict(type=egi_info["device"])

        ch_names, mon = _read_locs(input_fname, egi_info, channel_naming)
        ch_names.extend(list(egi_info["event_codes"]))
        n_extra = len(event_codes) + len(misc) + len(eog) + len(egi_info["pns_names"])
        if egi_info["new_trigger"] is not None:
            ch_names.append("STI 014")
            n_extra += 1
        ch_names.extend(egi_info["pns_names"])

        cals = _get_eeg_calibration_info(input_fname, egi_info)
        cals = np.concatenate([cals, np.ones(n_extra)])
        assert len(cals) == len(ch_names), (len(cals), len(ch_names))

        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, (), (), misc)

        sti_ch_idx = [
            i
            for i, name in enumerate(ch_names)
            if name.startswith("STI") or name in event_codes
        ]
        for idx in sti_ch_idx:
            chs[idx].update(
                {
                    "unit_mul": FIFF.FIFF_UNITM_NONE,
                    "cal": cals[idx],
                    "kind": FIFF.FIFFV_STIM_CH,
                    "coil_type": FIFF.FIFFV_COIL_NONE,
                    "unit": FIFF.FIFF_UNIT_NONE,
                }
            )
        chs = _add_pns_channel_info(chs, egi_info, ch_names)
        info["chs"] = chs
        info._unlocked = False
        info._update_redundant()

        if mon is not None:
            info.set_montage(mon, on_missing="ignore")

        if mon is not None:
            ref_idx = np.flatnonzero(np.isin(mon.ch_names, REFERENCE_NAMES))
            if len(ref_idx):
                ref_idx = ref_idx.item()
                ref_coords = info["chs"][int(ref_idx)]["loc"][:3]
                for chan in info["chs"]:
                    if chan["kind"] == FIFF.FIFFV_EEG_CH:
                        chan["loc"][3:6] = ref_coords

        file_bin = op.join(input_fname, egi_info["eeg_fname"])
        egi_info["egi_events"] = egi_events

        keys = ("eeg", "sti", "pns")
        idx = dict()
        idx["eeg"] = np.where([ch["kind"] == FIFF.FIFFV_EEG_CH for ch in chs])[0]
        idx["sti"] = np.where([ch["kind"] == FIFF.FIFFV_STIM_CH for ch in chs])[0]
        idx["pns"] = np.where(
            [
                ch["kind"] in (FIFF.FIFFV_ECG_CH, FIFF.FIFFV_EMG_CH, FIFF.FIFFV_BIO_CH)
                for ch in chs
            ]
        )[0]
        if not np.array_equal(
            np.concatenate([idx[key] for key in keys]), np.arange(len(chs))
        ):
            raise ValueError("Currently interlacing EEG and PNS channels is not supported")

        egi_info["kind_bounds"] = [0]
        for key in keys:
            egi_info["kind_bounds"].append(len(idx[key]))
        egi_info["kind_bounds"] = np.cumsum(egi_info["kind_bounds"])
        assert egi_info["kind_bounds"][0] == 0
        assert egi_info["kind_bounds"][-1] == info["nchan"]

        first_samps = [0]
        last_samps = [egi_info["last_samps"][-1] - 1]

        annot = dict(onset=list(), duration=list(), description=list())

        if len(idx["pns"]):
            egi_info["pns_filepath"] = op.join(input_fname, egi_info["pns_fname"])
            pns_samples = np.sum(egi_info["pns_sample_blocks"]["samples_block"])
            eeg_samples = np.sum(egi_info["samples_block"])
            if pns_samples == eeg_samples - 1:
                warn("This file has the EGI PSG sample bug")
                annot["onset"].append(last_samps[-1] / egi_info["sfreq"])
                annot["duration"].append(1 / egi_info["sfreq"])
                annot["description"].append("BAD_EGI_PSG")
            elif pns_samples != eeg_samples:
                raise RuntimeError(
                    "PNS samples (%d) did not match EEG samples (%d)"
                    % (pns_samples, eeg_samples)
                )

        super().__init__(
            info,
            preload=preload,
            orig_format="single",
            filenames=[file_bin],
            first_samps=first_samps,
            last_samps=last_samps,
            raw_extras=[egi_info],
            verbose=verbose,
        )

        egi_info["has_acq_skip"] = np.any(
            egi_info["first_samps"][1:] > egi_info["last_samps"][:-1]
        )
        egi_info["_acq_skip_warned"] = False

        for first, prev_last in zip(
            egi_info["first_samps"][1:], egi_info["last_samps"][:-1]
        ):
            gap = first - prev_last
            assert gap >= 0
            if gap:
                annot["onset"].append((prev_last - 0.5) / egi_info["sfreq"])
                annot["duration"].append(gap / egi_info["sfreq"])
                annot["description"].append("BAD_ACQ_SKIP")

        if events_as_annotations:
            for code, samples in mff_events.items():
                if code not in include:
                    continue
                annot["onset"].extend(np.array(samples) / egi_info["sfreq"])
                annot["duration"].extend([0.0] * len(samples))
                annot["description"].extend([code] * len(samples))

        if len(annot["onset"]):
            self.set_annotations(Annotations(**annot))

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of data."""
        logger.debug(f"Reading MFF {start:6d} ... {stop:6d} ...")
        dtype = "<f4"  # Data read in four byte floats.

        egi_info = self._raw_extras[fi]
        if (
            egi_info.get("has_acq_skip", False)
            and egi_info.get("n_events", 0) > 0
            and not egi_info.get("_acq_skip_warned", False)
        ):
            warn(
                "Acquisition skips detected. EGI MFF file contains gaps between "
                "recording epochs.",
                RuntimeWarning,
            )
            egi_info["_acq_skip_warned"] = True
        one = np.zeros((egi_info["kind_bounds"][-1], stop - start))

        # info about the binary file structure
        n_channels = egi_info["n_channels"]
        samples_block = egi_info["samples_block"]

        # Check how many channels to read are from each type
        bounds = egi_info["kind_bounds"]
        if isinstance(idx, slice):
            idx = np.arange(idx.start, idx.stop)
        eeg_out = np.where(idx < bounds[1])[0]
        eeg_one = idx[eeg_out, np.newaxis]
        eeg_in = idx[eeg_out]
        stim_out = np.where((idx >= bounds[1]) & (idx < bounds[2]))[0]
        stim_one = idx[stim_out]
        stim_in = idx[stim_out] - bounds[1]
        pns_out = np.where((idx >= bounds[2]) & (idx < bounds[3]))[0]
        pns_in = idx[pns_out] - bounds[2]
        pns_one = idx[pns_out, np.newaxis]
        del eeg_out, stim_out, pns_out

        # take into account events (already extended to correct size)
        one[stim_one, :] = egi_info["egi_events"][stim_in, start:stop]

        # Convert start and stop to limits in terms of the data
        # actually on disk, plus an indexer (disk_use_idx) that populates
        # the potentially larger `data` with it, taking skips into account
        disk_samps = egi_info["disk_samps"][start:stop]
        disk_use_idx = np.where(disk_samps > -1)[0]
        # short circuit in case we don't need any samples
        if not len(disk_use_idx):
            _mult_cal_one(data, one, idx, cals, mult)
            return

        start = disk_samps[disk_use_idx[0]]
        stop = disk_samps[disk_use_idx[-1]] + 1
        assert len(disk_use_idx) == stop - start

        # Get starting/stopping block/samples
        block_samples_offset = np.cumsum(samples_block)
        offset_blocks = np.sum(block_samples_offset <= start)
        offset_samples = start - (
            block_samples_offset[offset_blocks - 1] if offset_blocks > 0 else 0
        )

        # TODO: Refactor this reading with the PNS reading in a single function
        # (DRY)
        samples_to_read = stop - start
        with open(self.filenames[fi], "rb", buffering=0) as fid:
            # Go to starting block
            current_block = 0
            current_block_info = None
            current_data_sample = 0
            while current_block < offset_blocks:
                this_block_info = _block_r(fid)
                if this_block_info is not None:
                    current_block_info = this_block_info
                fid.seek(current_block_info["block_size"], 1)
                current_block += 1

            # Start reading samples
            while samples_to_read > 0:
                logger.debug(f"    Reading from block {current_block}")
                this_block_info = _block_r(fid)
                current_block += 1
                if this_block_info is not None:
                    current_block_info = this_block_info

                to_read = current_block_info["nsamples"] * current_block_info["nc"]
                block_data = np.fromfile(fid, dtype, to_read)
                block_data = block_data.reshape(n_channels, -1, order="C")

                # Compute indexes
                samples_read = block_data.shape[1]
                logger.debug(f"        Read   {samples_read} samples")
                logger.debug(f"        Offset {offset_samples} samples")
                if offset_samples > 0:
                    # First block read, skip to the offset:
                    block_data = block_data[:, offset_samples:]
                    samples_read = samples_read - offset_samples
                    offset_samples = 0
                if samples_to_read < samples_read:
                    # Last block to read, skip the last samples
                    block_data = block_data[:, :samples_to_read]
                    samples_read = samples_to_read
                logger.debug(f"        Keep   {samples_read} samples")

                s_start = current_data_sample
                s_end = s_start + samples_read

                one[eeg_one, disk_use_idx[s_start:s_end]] = block_data[eeg_in]
                samples_to_read = samples_to_read - samples_read
                current_data_sample = current_data_sample + samples_read

        if len(pns_one) > 0:
            # PNS Data is present and should be read:
            pns_filepath = egi_info["pns_filepath"]
            pns_info = egi_info["pns_sample_blocks"]
            n_channels = pns_info["n_channels"]
            samples_block = pns_info["samples_block"]

            # Get starting/stopping block/samples
            block_samples_offset = np.cumsum(samples_block)
            offset_blocks = np.sum(block_samples_offset < start)
            offset_samples = start - (
                block_samples_offset[offset_blocks - 1] if offset_blocks > 0 else 0
            )

            samples_to_read = stop - start
            with open(pns_filepath, "rb", buffering=0) as fid:
                # Check file size
                fid.seek(0, 2)
                file_size = fid.tell()
                fid.seek(0)
                # Go to starting block
                current_block = 0
                current_block_info = None
                current_data_sample = 0
                while current_block < offset_blocks:
                    this_block_info = _block_r(fid)
                    if this_block_info is not None:
                        current_block_info = this_block_info
                    fid.seek(current_block_info["block_size"], 1)
                    current_block += 1

                # Start reading samples
                while samples_to_read > 0:
                    if samples_to_read == 1 and fid.tell() == file_size:
                        # We are in the presence of the EEG bug
                        # fill with zeros and break the loop
                        one[pns_one, -1] = 0
                        break

                    this_block_info = _block_r(fid)
                    if this_block_info is not None:
                        current_block_info = this_block_info

                    to_read = current_block_info["nsamples"] * current_block_info["nc"]
                    block_data = np.fromfile(fid, dtype, to_read)
                    block_data = block_data.reshape(n_channels, -1, order="C")

                    # Compute indexes
                    samples_read = block_data.shape[1]
                    if offset_samples > 0:
                        # First block read, skip to the offset:
                        block_data = block_data[:, offset_samples:]
                        samples_read = samples_read - offset_samples
                        offset_samples = 0

                    if samples_to_read < samples_read:
                        # Last block to read, skip the last samples
                        block_data = block_data[:, :samples_to_read]
                        samples_read = samples_to_read

                    s_start = current_data_sample
                    s_end = s_start + samples_read

                    one[pns_one, disk_use_idx[s_start:s_end]] = block_data[pns_in]
                    samples_to_read = samples_to_read - samples_read
                    current_data_sample = current_data_sample + samples_read

        # do the calibration
        _mult_cal_one(data, one, idx, cals, mult)


@verbose
def read_evokeds_mff(
    fname, condition=None, channel_naming="E%d", baseline=None, verbose=None
):
    """Read averaged MFF file as EvokedArray or list of EvokedArray.

    Parameters
    ----------
    fname : path-like
        File path to averaged MFF file. Should end in ``.mff``.
    condition : int or str | list of int or str | None
        The index (indices) or category (categories) from which to read in
        data. Averaged MFF files can contain separate averages for different
        categories. These can be indexed by the block number or the category
        name. If ``condition`` is a list or None, a list of EvokedArray objects
        is returned.
    channel_naming : str
        Channel naming convention for EEG channels. Defaults to 'E%%d'
        (resulting in channel names 'E1', 'E2', 'E3'...).
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used. Correction is applied by computing mean
        of the baseline period and subtracting it from the data. The baseline
        (a, b) includes both endpoints, i.e. all timepoints t such that
        a <= t <= b.
    %(verbose)s

    Returns
    -------
    evoked : EvokedArray or list of EvokedArray
        The evoked dataset(s); one EvokedArray if condition is int or str,
        or list of EvokedArray if condition is None or list.

    Raises
    ------
    ValueError
        If ``fname`` has file extension other than '.mff'.
    ValueError
        If the MFF file specified by ``fname`` is not averaged.
    ValueError
        If no categories.xml file in MFF directory specified by ``fname``.

    See Also
    --------
    Evoked, EvokedArray, create_info

    Notes
    -----
    .. versionadded:: 0.22
    """
    mffpy = _import_mffpy()
    # Confirm `fname` is a path to an MFF file
    fname = Path(fname)  # should be replace with _check_fname
    if not fname.suffix == ".mff":
        raise ValueError('fname must be an MFF file with extension ".mff".')
    # Confirm the input MFF is averaged
    mff = mffpy.Reader(fname)
    try:
        flavor = mff.mff_flavor
    except AttributeError:  # < 6.3
        flavor = mff.flavor
    if flavor not in ("averaged", "segmented"):  # old, new names
        raise ValueError(
            f"{fname} is a {flavor} MFF file. "
            "fname must be the path to an averaged MFF file."
        )
    # Check for categories.xml file
    if "categories.xml" not in mff.directory.listdir():
        raise ValueError(
            "categories.xml not found in MFF directory. "
            f"{fname} may not be an averaged MFF file."
        )
    return_list = True
    if condition is None:
        categories = mff.categories.categories
        condition = list(categories.keys())
    elif not isinstance(condition, list):
        condition = [condition]
        return_list = False
    logger.info(f"Reading {len(condition)} evoked datasets from {fname} ...")
    output = [
        _read_evoked_mff(
            fname, c, channel_naming=channel_naming, verbose=verbose
        ).apply_baseline(baseline)
        for c in condition
    ]
    return output if return_list else output[0]


def _read_evoked_mff(fname, condition, channel_naming="E%d", verbose=None):
    """Read evoked data from MFF file."""
    import mffpy

    egi_info = _read_header(fname)
    mff = mffpy.Reader(fname)
    categories = mff.categories.categories

    if isinstance(condition, str):
        # Condition is interpreted as category name
        category = _check_option(
            "condition", condition, categories, extra="provided as category name"
        )
        epoch = mff.epochs[category]
    elif isinstance(condition, int):
        # Condition is interpreted as epoch index
        try:
            epoch = mff.epochs[condition]
        except IndexError:
            raise ValueError(
                f'"condition" parameter ({condition}), provided '
                "as epoch index, is out of range for available "
                f"epochs ({len(mff.epochs)})."
            )
        category = epoch.name
    else:
        raise TypeError('"condition" parameter must be either int or str.')

    # Read in signals from the target epoch
    data = mff.get_physical_samples_from_epoch(epoch)
    eeg_data, t0 = data["EEG"]
    if "PNSData" in data:
        pns_data, t0 = data["PNSData"]
        all_data = np.vstack((eeg_data, pns_data))
        ch_types = egi_info["chan_type"] + egi_info["pns_types"]
    else:
        all_data = eeg_data
        ch_types = egi_info["chan_type"]
    all_data *= 1e-6  # convert to volts

    # Load metadata into info object
    # Exclude info['meas_date'] because record time info in
    # averaged MFF is the time of the averaging, not true record time.
    ch_names, mon = _read_locs(fname, egi_info, channel_naming)
    ch_names.extend(egi_info["pns_names"])
    info = create_info(ch_names, mff.sampling_rates["EEG"], ch_types)
    with info._unlock():
        info["device_info"] = dict(type=egi_info["device"])
        info["nchan"] = sum(mff.num_channels.values())

    # Add individual channel info
    # Get calibration info for EEG channels
    cals = _get_eeg_calibration_info(fname, egi_info)
    # Initialize calibration for PNS channels, will be updated later
    cals = np.concatenate([cals, np.repeat(1, len(egi_info["pns_names"]))])
    ch_coil = FIFF.FIFFV_COIL_EEG
    ch_kind = FIFF.FIFFV_EEG_CH
    chs = _create_chs(ch_names, cals, ch_coil, ch_kind, (), (), (), ())
    # Update PNS channel info
    chs = _add_pns_channel_info(chs, egi_info, ch_names)
    with info._unlock():
        info["chs"] = chs
    if mon is not None:
        info.set_montage(mon, on_missing="ignore")

    # Add bad channels to info
    info["description"] = category
    try:
        channel_status = categories[category][0]["channelStatus"]
    except KeyError:
        warn(
            f"Channel status data not found for condition {category}. "
            "No channels will be marked as bad.",
            category=UserWarning,
        )
        channel_status = None
    bads = []
    if channel_status:
        for entry in channel_status:
            if entry["exclusion"] == "badChannels":
                if entry["signalBin"] == 1:
                    # Add bad EEG channels
                    for ch in entry["channels"]:
                        bads.append(ch_names[ch - 1])
                elif entry["signalBin"] == 2:
                    # Add bad PNS channels
                    for ch in entry["channels"]:
                        bads.append(egi_info["pns_names"][ch - 1])
    info["bads"] = bads

    # Add EEG reference to info
    try:
        fp = mff.directory.filepointer("history")
    except (ValueError, FileNotFoundError):  # old (<=0.6.3) vs new mffpy
        pass
    else:
        with fp:
            history = mffpy.XML.from_file(fp)
        for entry in history.entries:
            if entry["method"] == "Montage Operations Tool":
                if "Average Reference" in entry["settings"]:
                    # Average reference has been applied
                    _, info = setup_proj(info)

    # Get nave from categories.xml
    try:
        nave = categories[category][0]["keys"]["#seg"]["data"]
    except KeyError:
        warn(
            f"Number of averaged epochs not found for condition {category}. "
            "nave will default to 1.",
            category=UserWarning,
        )
        nave = 1

    # Let tmin default to 0
    return EvokedArray(
        all_data, info, tmin=0.0, comment=category, nave=nave, verbose=verbose
    )


def _import_mffpy(why="read averaged .mff files"):
    """Import and return module mffpy."""
    try:
        import mffpy
    except ImportError as exp:
        msg = f"mffpy is required to {why}, got:\n{exp}"
        raise ImportError(msg)

    # Monkey-patch mffpy to handle timestamps with 9 decimal places (nanoseconds)
    # This is needed because some MFF files have timestamps like
    # "2006-04-28T15:32:00.000000000+0100" which Python's %f can't parse
    if not hasattr(mffpy.XML, '_mne_patched'):
        original_parse_time_str = mffpy.XML._parse_time_str
        
        @classmethod
        def _patched_parse_time_str(cls, txt):
            """Parse time string with support for 9-decimal nanoseconds."""
            # Truncate nanoseconds to 6 decimal places if present
            # e.g., "2017-09-20T09:55:44.072000000+01:00" -> "2017-09-20T09:55:44.072000+01:00"
            if txt and '.' in txt:
                txt = re.sub(r'\.(\d{6})\d+([+-])', r'.\1\2', txt)
            return original_parse_time_str(txt)
        
        mffpy.XML._parse_time_str = _patched_parse_time_str
        mffpy.XML._mne_patched = True

    return mffpy
