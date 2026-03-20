# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import tarfile
from collections import Counter, namedtuple
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from defusedxml import ElementTree as ET

from ... import Annotations
from ..._fiff.meas_info import create_info
from ...utils import _check_fname, fill_doc, logger, verbose, warn
from ..base import BaseRaw

_CONTROL_CHS = ("buffer", "ramp")  # mapped to `stim`
_AUX_CHS = ("loadcell", "aux")  # mapped to `misc`
# Quaternion channels are handled separately (and are mapped to `chpi`)


def _signal_conversion_factors(device, adapter):
    # values extracted from
    # https://github.com/OTBioelettronica/OTB-Matlab/blob/main/MATLAB%20Open%20and%20Processing%20OTBFiles/OpenOTBFiles/OpenOTBfilesConvFact.m
    convfact = namedtuple("OTBAdapterProperties", ("adc_range", "bit_depth", "gain"))
    if adapter == "AdapterControl":
        return convfact(adc_range=1.0, bit_depth=1, gain=1.0)
    if adapter == "AdapterQuaternions":
        # NB: original MATLAB code sets bit_depth=1 here ↓
        return convfact(adc_range=1.0, bit_depth=16, gain=1.0)
    elif adapter in ("Direct connection", "Direct connection to Syncstation Input"):
        return convfact(adc_range=5.0, bit_depth=16, gain=0.5)
    elif device in ("QUATTRO", "QUATTROCENTO"):
        return convfact(adc_range=5.0, bit_depth=16, gain=150.0)
    elif device == "DUE":
        return convfact(adc_range=5.0, bit_depth=16, gain=200.0)
    elif device in ("DUE+", "QUATTRO+"):
        return convfact(adc_range=3.3, bit_depth=16, gain=202.0)
    # elif device == "MUOVI":
    #     return convfact(adc_range=..., bit_depth=16, gain=1.0)
    elif device == "SYNCSTATION":
        if adapter in ("Due+", "Quattro+"):
            return convfact(adc_range=3.3, bit_depth=16, gain=202.0)
        # elif adapter == "Sessantaquattro":
        #     return convfact(adc_range=..., bit_depth=16, gain=150.0)
        elif adapter == "AdapterLoadCell":
            # NB: original MATLAB code sets gain=205.0 here ↓
            return convfact(adc_range=5.0, bit_depth=16, gain=0.5)
        else:
            return convfact(adc_range=4.8, bit_depth=16, gain=256.0)
    else:
        return convfact(adc_range=4.8, bit_depth=16, gain=256.0)


def _get(node, attr, _type=str, **replacements):
    val = node.get(attr)
    # filter freqs may be "Unknown", can't blindly parse as floats
    return replacements[val] if val in replacements else _type(val)


def _find(node, tag, _type=str, **replacements):
    val = node.find(tag)
    if val is not None:
        val = val.text
    # filter freqs may be "Unknown", can't blindly parse as floats
    return replacements[val] if val in replacements else _type(val)


def _parse_otb_plus_metadata(metadata, extras_metadata):
    assert metadata.tag == "Device"
    # device-level metadata
    device_name = _get(metadata, "Name")
    device_bit_depth = _get(metadata, "ad_bits", int)
    n_chan = _get(metadata, "DeviceTotalChannels", int)
    sfreq = _get(metadata, "SampleFrequency", float)
    # containers
    adc_ranges = np.full(n_chan, np.nan)
    bit_depths = np.full(n_chan, np.nan)
    gains = np.full(n_chan, np.nan)
    ch_names = list()
    ch_types = list()
    highpass = list()
    lowpass = list()
    # check in advance where we'll need to append indices to uniquify ch_names
    n_ch_by_type = Counter([ch.get("ID") for ch in metadata.iter("Channel")])
    dupl_ids = [k for k, v in n_ch_by_type.items() if v > 1]
    ch_ix = {key: iter(range(val)) for key, val in n_ch_by_type.items()}
    if "Quaternion" in ch_ix:
        if (n_quat := n_ch_by_type["Quaternion"]) != 4:
            raise ValueError(
                f"Quaternions present, but there are {n_quat} of them (expected 4)"
            )
        ch_ix["Quaternion"] = iter(list("wxyz"))
    # iterate over adapters & channels to extract gain, filters, names, etc
    for adapter in metadata.iter("Adapter"):
        adapter_id = _get(adapter, "ID")
        adapter_adc_range, adapter_bit_depth, adapter_gain = _signal_conversion_factors(
            device_name, adapter_id
        )
        if (adapter_gain_xml := _get(adapter, "Gain", float)) not in (1, adapter_gain):
            warn(
                f"{device_name}, {adapter_id}, {adapter_gain_xml} from XML, "
                f"{adapter_gain} from LUT"
            )
        ch_offset = _get(adapter, "ChannelStartIndex", int)
        # we only really care about lowpass/highpass on the data channels
        if not any(adapter_id.startswith(t) for t in ("Adapter", "Direct connection")):
            highpass.append(_get(adapter, "HighPassFilter", float, Unknown=None))
            lowpass.append(_get(adapter, "LowPassFilter", float, Unknown=None))
        for ch in adapter.iter("Channel"):
            ch_id = _get(ch, "ID")
            if ch_id in dupl_ids:
                ch_names.append(f"{ch_id}_{next(ch_ix[ch_id])}")
            else:
                ch_names.append(ch_id)
            # store gains
            ix = _get(ch, "Index", int)
            gain_ix = ix + ch_offset
            gains[gain_ix] = _get(ch, "Gain", float) * adapter_gain
            adc_ranges[gain_ix] = adapter_adc_range
            bit_depths[gain_ix] = adapter_bit_depth
            # quats should maybe be FIFF.FIFFV_QUAT_{N} (N from 0-6), but need to verify
            # what quats should be, as there are only 4 quat channels. The FIFF quats:
            # 0: obsolete
            # 1-3: rotations
            # 4-6: translations
            if ch_id.startswith("Quaternion"):
                ch_type = "chpi"
                bit_depths[gain_ix] = device_bit_depth  # override TODO SAY WHY
            # ramp and buffer
            elif any(ch_id.lower().startswith(_ch.lower()) for _ch in _CONTROL_CHS):
                ch_type = "stim"
            # loadcell and AUX
            elif any(ch_id.lower().startswith(_ch.lower()) for _ch in _AUX_CHS):
                ch_type = "misc"
            else:
                ch_type = "emg"
                assert bit_depths[gain_ix] == device_bit_depth
            ch_types.append(ch_type)

    # parse subject info
    def parse_date(dt):
        return datetime.fromisoformat(dt).date()

    def parse_sex(sex):
        # For devices that generate `.otb+` files, the recording GUI only has M or F
        # options and choosing one is mandatory.
        return dict(m=1, f=2)[sex.lower()[0]] if sex else 0  # 0 means "unknown"

    subj_info_mapping = (
        ("family_name", "last_name", str),
        ("first_name", "first_name", str),
        ("weight", "weight", float),
        ("height", "height", float),
        ("sex", "sex", parse_sex),
        ("birth_date", "birthday", parse_date),
    )
    subject_info = dict()
    for source, target, func in subj_info_mapping:
        value = _find(extras_metadata, source)
        if value is not None:
            subject_info[target] = func(value)

    meas_date = _find(extras_metadata, "time")
    duration = _find(extras_metadata, "duration", float)
    site = _find(extras_metadata, "place")

    return dict(
        adc_range=adc_ranges,
        bit_depth=device_bit_depth,
        bit_depths=bit_depths,
        ch_names=ch_names,
        ch_types=ch_types,
        device_name=device_name,
        duration=duration,
        gains=gains,
        highpass=highpass,
        lowpass=lowpass,
        meas_date=meas_date,
        n_chan=n_chan,
        sfreq=sfreq,
        signal_paths=None,
        site=site,
        subject_info=subject_info,
        units=None,
    )


def _parse_otb_four_metadata(metadata, extras_metadata):
    assert metadata.tag == "DeviceParameters"
    # device-level metadata
    bit_depth = _find(metadata, "AdBits", int)  # TODO use `SampleSize * 8` instead?
    sfreq = _find(metadata, "SamplingFrequency", float)
    device_gain = _find(metadata, "Gain", float)
    # containers
    bit_depths = list()
    gains = list()
    ch_names = list()
    ch_types = list()
    highpass = list()
    lowpass = list()
    n_chans = list()
    paths = list()
    units = list()
    adc_ranges = list()
    # stored per-adapter, but should be uniform
    device_names = set()
    durations = set()
    meas_dates = set()
    sfreqs = set()
    # adapter-level metadata
    for adapter in extras_metadata.iter("TrackInfo"):
        strings = adapter.find("StringsDescriptions")
        # expected to be same for all adapters
        device_names.add(_find(adapter, "Device"))
        sfreqs.add(_find(adapter, "SamplingFrequency", int))
        durations.add(_find(adapter, "TimeDuration", float))
        # may be different for each adapter
        adapter_adc_range = _find(adapter, "ADC_Range", float)
        adapter_bit_depth = _find(adapter, "ADC_Nbits", int)
        adapter_id = _find(adapter, "SubTitle")
        adapter_gain = _find(adapter, "Gain", float)
        # adapter_scaling = 1.0 / _find(adapter, "UnitOfMeasurementFactor", float)
        # ch_offset = _find(adapter, "AcquisitionChannel", int)
        adapter_n_chans = _find(adapter, "NumberOfChannels", int)
        n_chans.append(adapter_n_chans)
        paths.append(_find(adapter, "SignalStreamPath"))
        units.append(_find(adapter, "UnitOfMeasurement"))
        # we only really care about lowpass/highpass on the data channels
        if adapter_id not in ("Quaternion", "Buffer", "Ramp"):
            hp = _find(strings, "HighPassFilter", float, Unknown=None)
            lp = _find(strings, "LowPassFilter", float, Unknown=None)
            if hp is not None:
                highpass.append(hp)
            if lp is not None:
                lowpass.append(lp)
        # meas_date
        meas_date = strings.find("StartDate")
        if meas_date is not None:
            meas_dates.add(meas_date.text)
        # # range (TODO maybe not needed; might be just for mfg's GUI?)
        # # FWIW in the example file: range for Buffer is 1-100,
        # # Ramp and Control are -32767-32768, and
        # # EMG chs are ±2.1237507098703645E-05
        # rmin = _find(adapter, "RangeMin", float)
        # rmax = _find(adapter, "RangeMax", float)
        # if rmin.is_integer() and rmax.is_integer():
        #     rmin = int(rmin)
        #     rmax = int(rmax)

        # extract channel-specific info                  ↓ not a typo
        for ch in adapter.find("Channels").iter("ChannelRapresentation"):
            # channel names
            if adapter_n_chans == 1:
                ch_name = adapter_id
            else:
                ix = int(ch.find("Index").text)
                ch_name = ch.find("Label").text
                try:
                    _ = int(ch_name)
                except ValueError:
                    pass
                else:
                    ch_name = f"{adapter_id}_{ix}"
            ch_names.append(ch_name)
            # signal properties
            adc_ranges.append(adapter_adc_range)
            bit_depths.append(adapter_bit_depth)
            gains.append(adapter_gain * device_gain)
            # channel types
            # TODO verify for quats & buffer channel
            # ramp and control channels maybe "MISC", arguably "STIM"?
            # quats should maybe be FIFF.FIFFV_QUAT_{N} (N from 0-6), but need to verify
            # what quats should be, as there are only 4 quat channels. The FIFF quats:
            # 0: obsolete (?)
            # 1-3: rotations
            # 4-6: translations
            if adapter_id.startswith("Quaternion"):
                ch_type = "chpi"  # TODO verify
            elif any(adapter_id.lower().startswith(_ch) for _ch in _CONTROL_CHS):
                ch_type = "stim"
            elif any(adapter_id.lower().startswith(_ch) for _ch in _AUX_CHS):
                ch_type = "misc"
            else:
                ch_type = "emg"
            ch_types.append(ch_type)

    # validate the fields stored at adapter level, but that ought to be uniform:
    def check_uniform(name, adapter_values, device_value=None):
        if len(adapter_values) > 1:
            vals = sorted(map(str, adapter_values))
            raise RuntimeError(
                f"multiple {name}s found ({', '.join(vals)}), this is not yet supported"
            )
        adapter_value = adapter_values.pop()
        if device_value is not None and device_value != adapter_value:
            raise RuntimeError(
                f"mismatch between device-level {name} ({device_value}) and "
                f"adapter-level {name} ({adapter_value})"
            )
        return adapter_value

    device_name = check_uniform("device name", device_names)
    duration = check_uniform("duration", durations)
    sfreq = check_uniform("sampling frequency", sfreqs, sfreq)
    meas_date = check_uniform("meas date", meas_dates)

    return dict(
        adc_range=adc_ranges,
        bit_depth=bit_depth,
        bit_depths=bit_depths,
        ch_names=ch_names,
        ch_types=ch_types,
        device_name=device_name,
        duration=duration,
        gains=gains,
        highpass=highpass,
        lowpass=lowpass,
        meas_date=meas_date,
        n_chan=sum(n_chans),
        sfreq=sfreq,
        signal_paths=paths,
        site=None,
        subject_info=dict(),
        units=units,
    )


def _parse_annots(tree_list):
    anns = set()  # avoids duplicate annots
    for tree in tree_list:
        for marker in tree.iter("Marker"):
            # TODO is it always "Milliseconds"?
            ons = _find(marker, "Milliseconds", float) / 1e3
            # TODO will markers ever have duration? is duration
            # encoded as onset/offset with 2 markers?
            dur = 0.0
            # simplify descriptions
            desc = _find(marker, "Description").strip()
            if desc.startswith(sync := "Sync Pulse with code: "):
                desc = int(desc.replace(sync, ""))
            # add to containers
            anns.add((ons, dur, desc))
    if anns:
        onset, duration, description = zip(*sorted(anns))
        return Annotations(
            onset=onset,
            duration=duration,
            description=description,
        )


@fill_doc
class RawOTB(BaseRaw):
    """Raw object from an OTB file.

    Parameters
    ----------
    fname : path-like
        Path to the OTB file.
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @verbose
    def __init__(self, fname, *, verbose=None):
        # Adapted from the MATLAB code at:
        # https://github.com/OTBioelettronica/OTB-Matlab/tree/main/MATLAB%20Open%20and%20Processing%20OTBFiles
        # with permission to relicense as BSD-3 granted here:
        # https://github.com/OTBioelettronica/OTB-Python/issues/2#issuecomment-2979135882
        fname = str(_check_fname(fname, "read", True, "fname"))
        v4_format = fname.endswith(".otb4")
        logger.info(f"Loading {fname}")

        self.preload = True  # lazy loading not supported

        with tarfile.open(fname, "r") as fid:
            fnames = fid.getnames()
            # the .sig file(s) are the binary channel data.
            sig_fnames = [_fname for _fname in fnames if _fname.endswith(".sig")]
            # the markers_NN.xml are the annotations
            ann_fnames = [_fname for _fname in fnames if _fname.startswith("marker")]
            # TODO ↓↓↓↓↓↓↓↓ this may be wrong for Novecento+ devices
            #               (MATLAB code appears to skip the first sig_fname)
            data_size_bytes = sum(fid.getmember(_fname).size for _fname in sig_fnames)
            # triage the file format versions
            if v4_format:
                metadata_fname = "DeviceParameters.xml"
                extras_fname = "Tracks_000.xml"
                parse_func = _parse_otb_four_metadata
            else:
                # .otb4 format may legitimately have multiple .sig files, but
                # .otb+ should not (if it's truly raw data)
                if len(sig_fnames) > 1:
                    raise NotImplementedError(
                        "multiple .sig files found in the OTB+ archive. Probably this "
                        "means that an acquisition was imported into another session. "
                        "This is not yet supported; please open an issue at "
                        "https://github.com/mne-tools/mne-emg/issues if you want us to "
                        "add such support."
                    )
                # the .xml file with the matching basename contains signal metadata
                metadata_fname = str(Path(sig_fnames[0]).with_suffix(".xml"))
                extras_fname = "patient.xml"
                parse_func = _parse_otb_plus_metadata
            # parse the XML into a tree
            metadata_tree = ET.fromstring(fid.extractfile(metadata_fname).read())
            extras_tree = ET.fromstring(fid.extractfile(extras_fname).read())
            ann_trees = [
                ET.fromstring(fid.extractfile(ann_fname).read())
                for ann_fname in ann_fnames
            ]
        # extract what we need from the tree
        metadata = parse_func(metadata_tree, extras_tree)
        annots = _parse_annots(ann_trees)
        adc_range = metadata["adc_range"]
        bit_depth = metadata["bit_depth"]
        bit_depths = metadata["bit_depths"]
        ch_names = metadata["ch_names"]
        ch_types = metadata["ch_types"]
        device_name = metadata["device_name"]
        duration = metadata["duration"]
        gains = metadata["gains"]
        highpass = metadata["highpass"]
        lowpass = metadata["lowpass"]
        meas_date = metadata["meas_date"]
        n_chan = metadata["n_chan"]
        sfreq = metadata["sfreq"]
        signal_paths = metadata["signal_paths"]
        site = metadata["site"]
        subject_info = metadata["subject_info"]
        # units = metadata["units"]  # TODO needed for orig_units maybe

        # bit_depth seems to be unreliable for some OTB4 files, so let's check:
        if duration is not None:  # None for OTB+ files
            expected_n_samp = int(duration * sfreq * n_chan)
            expected_bit_depth = int(np.rint(8 * data_size_bytes / expected_n_samp))
            if bit_depth != expected_bit_depth:
                warn(
                    f"mismatch between file metadata `AdBits` ({bit_depth} bit) and "
                    "computed sample size based on reported duration, sampling "
                    f"frequency, and number of channels ({expected_bit_depth} bit). "
                    "Using the computed bit depth."
                )
                bit_depth = expected_bit_depth
        if bit_depth == 16:
            _dtype = np.int16
        elif bit_depth == 24:  # EEG data recorded on OTB devices may be like this
            _dtype = np.uint8  # hack, see `_preload_data` method
        else:
            raise NotImplementedError(
                f"expected 16- or 24-bit data, but file metadata says {bit_depth}-bit. "
                "If this file can be successfully read with other software (i.e. it is "
                "not corrupted), please open an issue at "
                "https://github.com/mne-tools/mne-emg/issues so we can add support for "
                "your file."
            )
        # compute number of samples
        n_samples, extra = divmod(data_size_bytes, (bit_depth // 8) * n_chan)
        if extra != 0:
            warn(
                f"Number of bytes in file ({data_size_bytes}) not evenly divided by "
                f"number of channels ({n_chan}). File may be corrupted or truncated."
            )

        # validate
        assert np.isfinite(gains).all()
        assert np.isfinite(adc_range).all()
        assert np.isfinite(bit_depths).all()

        # check filter freqs. Can vary by adapter, so in theory we might get different
        # filters for different *data* channels (not just different between data and
        # misc/aux/whatever).
        def check_filter_freqs(name, freqs):
            if not len(freqs):
                return None
            else:
                extreme = dict(highpass="lowest", lowpass="highest")[name]
                func = dict(highpass=min, lowpass=max)[name]
                extremum = func(freqs)
                if len(freqs) > 1:
                    warn(
                        f"More than one {name} frequency found in file; choosing "
                        f"{extreme} ({extremum} Hz)"
                    )
                return extremum

        highpass = check_filter_freqs("highpass", highpass)
        lowpass = check_filter_freqs("lowpass", lowpass)

        # create info
        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        device_info = dict(type="OTB", model=device_name)  # other allowed keys: serial
        if site is not None:
            device_info.update(site=site)
        info.update(subject_info=subject_info, device_info=device_info)
        with info._unlock():
            info["highpass"] = highpass
            info["lowpass"] = lowpass
            for ix, _ch in enumerate(info["chs"]):
                cal = 1 / 2 ** bit_depths[ix] / gains[ix]
                _ch.update(cal=cal, range=adc_range[ix])
            if meas_date is not None:
                info["meas_date"] = datetime.fromisoformat(meas_date).astimezone(
                    timezone.utc
                )

        # verify duration from metadata matches n_samples/sfreq
        if duration is not None:
            np.testing.assert_almost_equal(duration, n_samples / sfreq, decimal=3)

        # TODO other fields in extras_tree for otb+ format:
        # protocol_code, pathology, commentsPatient, comments

        # TODO parse files markers_0.xml, markers_1.xml as annotations?

        # populate raw_extras
        raw_extras = dict(
            device_name=device_name,
            dtype=_dtype,
            sig_fnames=sig_fnames,
            signal_paths=signal_paths,
        )
        FORMAT_MAPPING = dict(
            d="double",
            f="single",
            i="int",
            h="short",
            B="int",  # hack, we read 24-bit as uint8 and upcast to 32-bit signed int
        )
        orig_format = FORMAT_MAPPING[_dtype().dtype.char]

        super().__init__(
            info,
            preload=True,
            last_samps=(n_samples - 1,),
            filenames=[fname],
            orig_format=orig_format,
            # orig_units=dict(...),  # TODO needed?
            raw_extras=[raw_extras],
            verbose=verbose,
        )
        if annots:
            self.set_annotations(annots)

    def _preload_data(self, preload):
        """Load raw data from an OTB+ file."""
        _extras = self._raw_extras[0]
        sig_fnames = _extras["sig_fnames"]
        # if device_name=="Novecento+" then we may need these:
        # sig_paths = _extras["signal_paths"]
        # device_name = _extras["device_name"]

        with tarfile.open(self.filenames[0], "r") as fid:
            _data = list()
            for sig_fname in sig_fnames:
                this_data = np.frombuffer(
                    fid.extractfile(sig_fname).read(), _extras["dtype"]
                )
                if _extras["dtype"] is np.uint8:  # hack to handle 24-bit data
                    # adapted from wavio._wav2array © 2015-2022 Warren Weckesser (BSD-2)
                    a = np.empty((this_data.size // 3, 4), dtype=np.uint8)
                    a[..., :3] = this_data.reshape(-1, 3)
                    # we read in 24-bit data as unsigned ints, but assume that it was
                    # actually *signed* data. So we check the most significant bit:
                    msb = a[..., 2:3] >> 7
                    # Where it was 1, the 24-bit number was negative, so the added byte
                    # should be 11111111; otherwise it should be 00000000.
                    a[..., 3:] = msb * np.iinfo(np.uint8).max
                    # now we upcast to signed 32-bit and remove the extra dimension
                    this_data = np.squeeze(a.view(np.int32), axis=-1)
                _data.append(this_data.reshape(-1, self.info["nchan"]).T)
            _data = np.concatenate(_data, axis=0)  # no-op if len(_data) == 1
        cals = np.array([_ch["cal"] * _ch["range"] for _ch in self.info["chs"]])
        self._data = _data * cals[:, np.newaxis]


@fill_doc
def read_raw_otb(fname, verbose=None) -> RawOTB:
    """Reader for an OTB (.otb/.otb+/.otb4) recording.

    Parameters
    ----------
    fname : path-like
        Path to the OTB file.
    %(verbose)s

    Returns
    -------
    raw : instance of RawOTB
        A Raw object containing OTB data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawPersyst.

    Notes
    -----
    ``preload=False`` is not supported for this file format.
    """
    return RawOTB(fname, verbose=verbose)
