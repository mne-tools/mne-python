# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from defusedxml import ElementTree as ET

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ...utils import _check_fname, fill_doc, logger, verbose, warn
from ..base import BaseRaw

# these will all get mapped to `misc`. Quaternion channels are handled separately.
_NON_DATA_CHS = ("buffer", "ramp", "loadcell", "aux")


def _parse_date(dt):
    return datetime.fromisoformat(dt).date()


def _parse_patient_xml(tree):
    """Convert an ElementTree to a dict."""

    def _parse_sex(sex):
        # For devices that generate `.otb+` files, the recording GUI only has M or F
        # options and choosing one is mandatory. For `.otb4` the field is optional.
        return dict(m=1, f=2)[sex.lower()[0]] if sex else 0  # 0 means "unknown"

    subj_info_mapping = (
        ("family_name", "last_name", str),
        ("first_name", "first_name", str),
        ("weight", "weight", float),
        ("height", "height", float),
        ("sex", "sex", _parse_sex),
        ("birth_date", "birthday", _parse_date),
    )
    subject_info = dict()
    for source, target, func in subj_info_mapping:
        value = tree.find(source)
        if value is not None:
            subject_info[target] = func(value.text)
    return subject_info


def _parse_otb_plus_metadata(metadata, extras_metadata):
    assert metadata.tag == "Device"
    # device-level metadata
    adc_range = 0.0033  # 3.3 mV (TODO VERIFY)
    bit_depth = int(metadata.attrib["ad_bits"])
    device_name = metadata.attrib["Name"]
    n_chan = int(metadata.attrib["DeviceTotalChannels"])
    sfreq = float(metadata.attrib["SampleFrequency"])
    # containers
    gains = np.full(n_chan, np.nan)
    scalings = np.full(n_chan, np.nan)
    ch_names = list()
    ch_types = list()
    highpass = list()
    lowpass = list()
    # check in advance where we'll need to append indices to uniquify ch_names
    n_ch_by_type = Counter([ch.get("ID") for ch in metadata.iter("Channel")])
    dupl_ids = [k for k, v in n_ch_by_type.items() if v > 1]
    # iterate over adapters & channels to extract gain, filters, names, etc
    for adapter in metadata.iter("Adapter"):
        adapter_id = adapter.get("ID")
        adapter_gain = float(adapter.get("Gain"))
        ch_offset = int(adapter.get("ChannelStartIndex"))
        # we only really care about lowpass/highpass on the data channels
        if adapter_id not in ("AdapterQuaternions", "AdapterControl"):
            highpass.append(float(adapter.get("HighPassFilter")))
            lowpass.append(float(adapter.get("LowPassFilter")))
        for ch in adapter.iter("Channel"):
            ix = int(ch.get("Index"))
            ch_id = ch.get("ID")
            # # see if we can parse the adapter name to get row,col info
            # pattern = re.compile(
            #     # connector type   inter-elec dist    grid rows    grid cols
            #     r"(?:[a-zA-Z]+)(?:(?P<ied>\d+)MM)(?P<row>\d{2})(?P<col>\d{2})"
            # )
            # if match := pattern.match(ch_id):
            #     col = ix % int(match["col"])
            #     row = ix // int(match["row"])
            #     ch_name = f"EMG_{adapter_ix}({row:02},{col:02})"
            # elif ch_id
            # else:
            #     ch_name = f"EMG_{ix + adapter_ch_offset:03}"
            # ch_names.append(ch_name)
            ch_names.append(f"{ch_id}_{ix}" if ch_id in dupl_ids else ch_id)
            # store gains
            gain_ix = ix + ch_offset
            gains[gain_ix] = float(ch.get("Gain")) * adapter_gain
            # TODO verify ch_type for quats & buffer channel
            # ramp and control channels definitely "MISC"
            # quats should maybe be FIFF.FIFFV_QUAT_{N} (N from 0-6), but need to verify
            # what quats should be, as there are only 4 quat channels. The FIFF quats:
            # 0: obsolete
            # 1-3: rotations
            # 4-6: translations
            if ch_id.startswith("Quaternion"):
                ch_type = "chpi"  # TODO verify
                scalings[gain_ix] = 1e-4
            elif any(ch_id.lower().startswith(_ch.lower()) for _ch in _NON_DATA_CHS):
                ch_type = "misc"
                scalings[gain_ix] = 1.0
            else:
                ch_type = "emg"
                scalings[gain_ix] = 1.0
            ch_types.append(ch_type)
    # parse subject info
    subject_info = _parse_patient_xml(extras_metadata)

    return dict(
        sfreq=sfreq,
        n_chan=n_chan,
        bit_depth=bit_depth,
        device_name=device_name,
        adc_range=adc_range,
        subject_info=subject_info,
        gains=gains,
        ch_names=ch_names,
        ch_types=ch_types,
        highpass=highpass,
        lowpass=lowpass,
        units=None,
        scalings=scalings,
        signal_paths=None,
    )


def _parse_otb_four_metadata(metadata, extras_metadata):
    def get_str(node, tag):
        return node.find(tag).text

    def get_int(node, tag):
        return int(get_str(node, tag))

    def get_float(node, tag, **replacements):
        # filter freqs may be "Unknown", can't blindly parse as floats
        val = get_str(node, tag)
        return replacements.get(val, float(val))

    assert metadata.tag == "DeviceParameters"
    # device-level metadata
    adc_range = float(metadata.find("ADC_Range").text)
    bit_depth = int(metadata.find("AdBits").text)
    n_chan = int(metadata.find("TotalChannelsInFile").text)
    sfreq = float(metadata.find("SamplingFrequency").text)
    # containers
    gains = np.full(n_chan, np.nan)
    ch_names = list()
    ch_types = list()
    highpass = list()
    lowpass = list()
    n_chans = list()
    paths = list()
    scalings = list()
    units = list()
    # stored per-adapter, but should be uniform
    adc_ranges = set()
    bit_depths = set()
    device_names = set()
    meas_dates = set()
    sfreqs = set()
    # adapter-level metadata
    for adapter in extras_metadata.iter("TrackInfo"):
        # expected to be same for all adapters
        adc_ranges.add(get_float(adapter, "ADC_Range"))
        bit_depths.add(get_int(adapter, "ADC_Nbits"))
        device_names.add(get_str(adapter, "Device"))
        sfreqs.add(get_int(adapter, "SamplingFrequency"))
        # may be different for each adapter
        adapter_id = get_str(adapter, "SubTitle")
        adapter_gain = get_float(adapter, "Gain")
        ch_offset = get_int(adapter, "AcquisitionChannel")
        n_chans.append(get_int(adapter, "NumberOfChannels"))
        paths.append(get_str(adapter, "SignalStreamPath"))
        scalings.append(get_str(adapter, "UnitOfMeasurementFactor"))
        units.append(get_str(adapter, "UnitOfMeasurement"))
        # we only really care about lowpass/highpass on the data channels
        if adapter_id not in ("Quaternion", "Buffer", "Ramp"):
            highpass = get_float(adapter, "HighPassFilter", Unknown=None)
            lowpass = get_float(adapter, "LowPassFilter", Unknown=None)
        # meas_date
        meas_date = adapter.find("StringsDescriptions").find("StartDate")
        if meas_date is not None:
            meas_dates.add(_parse_date(meas_date.text).astimezone(timezone.utc))
        # # range (TODO maybe not needed; might be just for mfg's GUI?)
        # # FWIW in the example file: range for Buffer is 1-100,
        # # Ramp and Control are -32767-32768, and
        # # EMG chs are ±2.1237507098703645E-05
        # rmin = get_float(adapter, "RangeMin")
        # rmax = get_float(adapter, "RangeMax")
        # if rmin.is_integer() and rmax.is_integer():
        #     rmin = int(rmin)
        #     rmax = int(rmax)

        # extract channel-specific info                 ↓ not a typo
        for ch in adapter.get("Channels").iter("ChannelRapresentation"):
            # channel names
            ix = int(ch.find("Index").text)
            ch_name = ch.find("Label").text
            try:
                _ = int(ch_name)
            except ValueError:
                pass
            else:
                ch_name = f"{adapter_id}_{ix}"
            ch_names.append(ch_name)
            # gains
            gains[ix + ch_offset] = adapter_gain
            # channel types
            # TODO verify for quats & buffer channel
            # ramp and control channels definitely "MISC"
            # quats should maybe be FIFF.FIFFV_QUAT_{N} (N from 0-6), but need to verify
            # what quats should be, as there are only 4 quat channels. The FIFF quats:
            # 0: obsolete
            # 1-3: rotations
            # 4-6: translations
            if adapter_id.startswith("Quaternion"):
                ch_type = "chpi"  # TODO verify
            elif any(adapter_id.lower().startswith(_ch) for _ch in _NON_DATA_CHS):
                ch_type = "misc"
            else:
                ch_type = "emg"
            ch_types.append(ch_type)

    # validate the fields stored at adapter level, but that ought to be uniform:
    def check_uniform(name, adapter_values, device_value=None):
        if len(adapter_values) > 1:
            raise RuntimeError(
                f"multiple {name}s found ({', '.join(sorted(adapter_values))}), "
                "this is not yet supported"
            )
        adapter_value = adapter_values.pop()
        if device_value is not None and device_value != adapter_value:
            raise RuntimeError(
                f"mismatch between device-level {name} ({device_value}) and "
                f"adapter-level {name} ({adapter_value})"
            )
        return adapter_value

    device_name = check_uniform("device name", device_names)
    adc_range = check_uniform("analog-to-digital range", adc_ranges, adc_range)
    bit_depth = check_uniform("bit depth", bit_depths, bit_depth)
    sfreq = check_uniform("sampling frequency", sfreqs, sfreq)

    # verify number of channels in device metadata matches sum of adapters
    assert sum(n_chans) == n_chan, (
        f"total channels ({n_chan}) doesn't match sum of channels for each adapter "
        f"({sum(n_chans)})"
    )

    return dict(
        adc_range=adc_range,
        bit_depth=bit_depth,
        ch_names=ch_names,
        ch_types=ch_types,
        device_name=device_name,
        gains=gains,
        highpass=highpass,
        lowpass=lowpass,
        n_chan=n_chan,
        scalings=scalings,
        sfreq=sfreq,
        signal_paths=paths,
        subject_info=dict(),
        units=units,
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
        # extract what we need from the tree
        metadata = parse_func(metadata_tree, extras_tree)
        adc_range = metadata["adc_range"]
        bit_depth = metadata["bit_depth"]
        ch_names = metadata["ch_names"]
        ch_types = metadata["ch_types"]
        device_name = metadata["device_name"]
        gains = metadata["gains"]
        highpass = metadata["highpass"]
        lowpass = metadata["lowpass"]
        n_chan = metadata["n_chan"]
        scalings = metadata["scalings"]
        sfreq = metadata["sfreq"]
        signal_paths = metadata["signal_paths"]
        subject_info = metadata["subject_info"]
        # units = metadata["units"]  # TODO needed for orig_units maybe

        if bit_depth == 16:
            _dtype = np.int16
        elif bit_depth == 24:  # EEG data recorded on OTB devices do this
            # this is possible but will be a bit tricky, see:
            # https://stackoverflow.com/a/34128171
            # https://stackoverflow.com/a/11967503
            raise NotImplementedError(
                "OTB files with 24-bit data are not yet supported."
            )
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
        n_samples = int(n_samples)

        # validate gains
        assert np.isfinite(gains).all()

        # check filter freqs. Can vary by adapter, so in theory we might get different
        # filters for different *data* channels (not just different between data and
        # misc/aux/whatever).
        if len(highpass) > 1:
            warn(
                "More than one highpass frequency found in file; choosing lowest "
                f"({min(highpass)} Hz)"
            )
        if len(lowpass) > 1:
            warn(
                "More than one lowpass frequency found in file; choosing highest "
                f"({max(lowpass)} Hz)"
            )
        highpass = min(highpass)
        lowpass = max(lowpass)

        # create info
        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        device_info = dict(type="OTB", model=device_name)  # other allowed keys: serial
        meas_date = extras_tree.find("time")
        site = extras_tree.find("place")
        if site is not None:
            device_info.update(site=site.text)
        info.update(subject_info=subject_info, device_info=device_info)
        with info._unlock():
            info["highpass"] = highpass
            info["lowpass"] = lowpass
            for ix, _ch in enumerate(info["chs"]):
                # divisor = 1.0 if _ch["kind"] == FIFF.FIFFV_MISC_CH else 2**bit_depth
                cal = 1 / 2**bit_depth / gains[ix] * scalings[ix]
                _range = (
                    adc_range
                    if _ch["kind"] in (FIFF.FIFFV_EMG_CH, FIFF.FIFFV_EEG_CH)
                    else 1.0
                )
                _ch.update(cal=cal, range=_range)
            if meas_date is not None:
                info["meas_date"] = datetime.fromisoformat(meas_date.text).astimezone(
                    timezone.utc
                )

        # verify duration from metadata matches n_samples/sfreq
        dur = extras_tree.find("duration")
        if dur is not None:
            np.testing.assert_almost_equal(
                float(dur.text), n_samples / sfreq, decimal=3
            )

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
                _data.append(
                    np.frombuffer(
                        fid.extractfile(sig_fname).read(),
                        dtype=_extras["dtype"],
                    )
                    .reshape(-1, self.info["nchan"])
                    .T
                )
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
