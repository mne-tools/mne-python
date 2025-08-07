# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from defusedxml import ElementTree as ET

from ..._fiff.meas_info import create_info
from ...utils import _check_fname, fill_doc, logger, verbose, warn
from ..base import BaseRaw


def _parse_date(dt):
    return datetime.fromisoformat(dt).date()


def _parse_patient_xml(tree):
    """Convert an ElementTree to a dict."""

    def _parse_sex(sex):
        # TODO For devices that generate `.otb+` files, the recording GUI only has M or
        # F options and choosing one is mandatory. For `.otb4` the field is optional.
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
    sfreq = float(metadata.attrib["SampleFrequency"])
    n_chan = int(metadata.attrib["DeviceTotalChannels"])
    bit_depth = int(metadata.attrib["ad_bits"])
    model = metadata.attrib["Name"]
    adc_range = 3.3
    return dict(
        sfreq=sfreq,
        n_chan=n_chan,
        bit_depth=bit_depth,
        model=model,
        adc_range=adc_range,
    )


def _parse_otb_four_metadata(metadata, extras_metadata):
    pass


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

        highpass = list()
        lowpass = list()
        ch_names = list()
        ch_types = list()

        # these are the only non-data channel IDs (besides "AUX*", handled via glob)
        NON_DATA_CHS = ("Quaternion", "BufferChannel", "RampChannel", "LoadCellChannel")

        with tarfile.open(fname, "r") as fid:
            fnames = fid.getnames()
            # the .sig file(s) are the binary channel data.
            sig_fnames = [_fname for _fname in fnames if _fname.endswith(".sig")]
            # TODO ↓↓↓↓↓↓↓↓ make compatible with multiple sig_fnames
            data_size_bytes = fid.getmember(sig_fnames[0]).size
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
        sfreq = metadata["sfreq"]
        n_chan = metadata["n_chan"]
        bit_depth = metadata["bit_depth"]
        model = metadata["model"]
        adc_range = metadata["adc_range"]

        if bit_depth == 16:
            _dtype = np.int16
        elif bit_depth == 24:  # EEG data recorded on OTB devices do this
            # this is possible but will be a bit tricky, see:
            # https://stackoverflow.com/a/34128171
            # https://stackoverflow.com/a/11967503
            raise NotImplementedError(
                "OTB+ files with 24-bit data are not yet supported."
            )
        else:
            raise NotImplementedError(
                f"expected 16- or 24-bit data, but file metadata says {bit_depth}-bit. "
                "If this file can be successfully read with other software (i.e. it is "
                "not corrupted), please open an issue at "
                "https://github.com/mne-tools/mne-emg/issues so we can add support for "
                "your use case."
            )
        gains = np.full(n_chan, np.nan)
        # check in advance where we'll need to append indices to uniquify ch_names
        n_ch_by_type = Counter([ch.get("ID") for ch in metadata_tree.iter("Channel")])
        dupl_ids = [k for k, v in n_ch_by_type.items() if v > 1]
        # iterate over adapters & channels to extract gain, filters, names, etc
        for adapter_ix, adapter in enumerate(metadata_tree.iter("Adapter")):
            adapter_ch_offset = int(adapter.get("ChannelStartIndex"))
            adapter_gain = float(adapter.get("Gain"))
            # we only care about lowpass/highpass on the data channels
            # TODO verify these two are the only non-data adapter types
            if adapter.get("ID") not in ("AdapterQuaternions", "AdapterControl"):
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
                gains[ix + adapter_ch_offset] = float(ch.get("Gain")) * adapter_gain
                # TODO verify ch_type for quats, buffer channel, and ramp channel
                ch_types.append(
                    "misc"
                    if ch_id in NON_DATA_CHS or ch_id.lower().startswith("aux")
                    else "emg"
                )
        assert np.isfinite(gains).all()

        # compute number of samples
        n_samples, extra = divmod(data_size_bytes, (bit_depth // 8) * n_chan)
        if extra != 0:
            warn(
                f"Number of bytes in file ({data_size_bytes}) not evenly divided by "
                f"number of channels ({n_chan}). File may be corrupted or truncated."
            )
        n_samples = int(n_samples)

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
        subject_info = _parse_patient_xml(extras_tree)
        device_info = dict(type="OTB", model=model)  # other allowed keys: serial
        meas_date = extras_tree.find("time")
        site = extras_tree.find("place")
        if site is not None:
            device_info.update(site=site.text)
        info.update(subject_info=subject_info, device_info=device_info)
        with info._unlock():
            info["highpass"] = highpass
            info["lowpass"] = lowpass
            for _ch in info["chs"]:
                cal = 1 / 2**bit_depth / gains[ix + adapter_ch_offset]
                _ch.update(cal=cal, range=adc_range)
            if meas_date is not None:
                info["meas_date"] = datetime.fromisoformat(meas_date.text).astimezone(
                    timezone.utc
                )

        # sanity check
        dur = extras_tree.find("duration")
        if dur is not None:
            np.testing.assert_almost_equal(
                float(dur.text), n_samples / sfreq, decimal=3
            )

        # TODO other fields in extras_tree:
        # protocol_code, pathology, commentsPatient, comments

        # TODO parse files markers_0.xml, markers_1.xml as annotations?

        # populate raw_extras
        raw_extras = dict(dtype=_dtype, sig_fnames=sig_fnames)
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
            # orig_units="V",  # TODO maybe not needed
            raw_extras=[raw_extras],
            verbose=verbose,
        )

    def _preload_data(self, preload):
        """Load raw data from an OTB+ file."""
        _extras = self._raw_extras[0]
        sig_fnames = _extras["sig_fnames"]

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
            if len(_data) == 1:
                _data = _data[0]
            else:
                _data = np.concatenate(_data, axis=0)

        cals = np.array(
            [
                _ch["cal"] * _ch["range"] * _ch.get("scale", 1.0)
                for _ch in self.info["chs"]
            ]
        )
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
