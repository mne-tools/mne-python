# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from defusedxml import ElementTree as ET

from ..._fiff.meas_info import create_info
from ...utils import _check_fname, fill_doc, logger, verbose, warn
from ..base import BaseRaw

OTB_PLUS_DTYPE = np.int16  # the file I have is int16; TBD if *all* OTB+ are like that
FORMAT_MAPPING = dict(
    d="double",
    f="single",
    i="int",
    h="short",
)
OTB_PLUS_FORMAT = FORMAT_MAPPING[OTB_PLUS_DTYPE().dtype.char]


def _parse_patient_xml(tree):
    """Convert an ElementTree to a dict."""

    def _parse_sex(sex):
        # TODO English-centric; any value of "sex" not starting with "m" or "f" will get
        # classed as "unknown". TBD if non-English-like values in the XML are possible
        # (e.g. maybe the recording GUI is localized but the XML values are not?)
        return dict(m=1, f=2)[sex.lower()[0]] if sex else 0

    def _parse_date(dt):
        return datetime.fromisoformat(dt).date()

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
        if value := tree.find(source):
            subject_info[target] = func(value.text)
    return subject_info


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
        otb_version = "four" if fname.endswith(".otb4") else "plus"
        logger.info(f"Loading {fname}")

        self.preload = True  # lazy loading not supported

        highpass = list()
        lowpass = list()
        ch_names = list()
        ch_types = list()

        # TODO verify these are the only non-data channel IDs
        NON_DATA_CHS = ("Quaternion", "BufferChannel", "RampChannel")

        with tarfile.open(fname, "r") as fid:
            fnames = fid.getnames()
            # the .sig file is the binary channel data
            sig_fname = [_fname for _fname in fnames if _fname.endswith(".sig")]
            assert len(sig_fname) == 1  # TODO is this a valid assumption?
            sig_fname = sig_fname[0]
            data_size_bytes = fid.getmember(sig_fname).size
            # the .xml file with the matching basename contains signal metadata
            metadata_fname = str(Path(sig_fname).with_suffix(".xml"))
            metadata = ET.fromstring(fid.extractfile(metadata_fname).read())
            # patient info
            patient_info_xml = ET.fromstring(fid.extractfile("patient.xml").read())
        # structure of `metadata` is:
        # Device
        # └ Channels
        #   ├ Adapter
        #   | ├ Channel
        #   | ├ ...
        #   | └ Channel
        #   ├ ...
        #   └ Adapter
        #     ├ Channel
        #     ├ ...
        #     └ Channel
        assert metadata.tag == "Device"
        sfreq = float(metadata.attrib["SampleFrequency"])
        n_chan = int(metadata.attrib["DeviceTotalChannels"])
        bit_depth = int(metadata.attrib["ad_bits"])
        assert bit_depth == 16, f"expected 16-bit data, got {bit_depth}"  # TODO verify
        gains = np.full(n_chan, np.nan)
        # check in advance where we'll need to append indices to uniquify ch_names
        n_ch_by_type = Counter([ch.get("ID") for ch in metadata.iter("Channel")])
        dupl_ids = [k for k, v in n_ch_by_type.items() if v > 1]
        # iterate over adapters & channels to extract gain, filters, names, etc
        for adapter in metadata.iter("Adapter"):
            ch_offset = int(adapter.get("ChannelStartIndex"))
            adapter_gain = float(adapter.get("Gain"))
            # we only care about lowpass/highpass on the data channels
            # TODO verify these two are the only non-data adapter types
            if adapter.get("ID") not in ("AdapterQuaternions", "AdapterControl"):
                highpass.append(float(adapter.get("HighPassFilter")))
                lowpass.append(float(adapter.get("LowPassFilter")))

            for ch in adapter.iter("Channel"):
                # uniquify channel names by appending channel index (if needed)
                ix = int(ch.get("Index"))
                ch_id = ch.get("ID")
                # TODO better to call these "emg_1" etc? should we left-zeropad ix?
                ch_names.append(f"{ch_id}_{ix}" if ch_id in dupl_ids else ch_id)
                # store gains
                gains[ix + ch_offset] = float(ch.get("Gain")) * adapter_gain

                # TODO verify ch_type for quats, buffer channel, and ramp channel
                ch_types.append("misc" if ch_id in NON_DATA_CHS else "emg")

        # compute number of samples
        n_samples, extra = divmod(data_size_bytes, (bit_depth / 8) * n_chan)
        if extra != 0:
            warn(
                f"Number of bytes in file ({data_size_bytes}) not evenly divided by "
                f"number of channels ({n_chan}). File may be corrupted or truncated."
            )
        n_samples = int(n_samples)

        # check filter freqs
        if len(highpass) > 1:
            warn(
                "More than one highpass frequency found in file; choosing highest "
                f"({max(highpass)})"
            )
        if len(lowpass) > 1:
            warn(
                "More than one lowpass frequency found in file; choosing lowest "
                f"({min(lowpass)})"
            )
        highpass = max(highpass)
        lowpass = min(lowpass)

        # create info
        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        subject_info = _parse_patient_xml(patient_info_xml)
        info.update(subject_info=subject_info)
        with info._unlock():
            info["highpass"] = highpass
            info["lowpass"] = lowpass
            if meas_date := patient_info_xml.find("time"):
                info["meas_date"] = datetime.fromisoformat(meas_date.text)

        # sanity check
        if dur := patient_info_xml.find("duration"):
            assert float(dur.text) == n_samples / sfreq

        # TODO other fields in patient_info_xml:
        # protocol_code, place, pathology, commentsPatient, comments

        # TODO parse files markers_0.xml, markers_1.xml as annotations

        # populate raw_extras
        raw_extras = dict(
            n_samples=n_samples,
            gains=gains,
            dtype=OTB_PLUS_DTYPE,
            data_size_bytes=data_size_bytes,
            sig_fname=sig_fname,
            bit_depth=bit_depth,
            otb_version=otb_version,
        )

        super().__init__(
            info,
            preload=True,
            last_samps=(n_samples - 1,),
            filenames=[fname],
            orig_format=OTB_PLUS_FORMAT,
            # orig_units="mV",  # TODO verify
            raw_extras=[raw_extras],
            verbose=verbose,
        )

    def _preload_data(self, preload):
        """Load raw data from an OTB+ file."""
        _extras = self._raw_extras[0]
        sig_fname = _extras["sig_fname"]
        power_supply = 3.3 if _extras["otb_version"] == "plus" else None
        bit_depth = _extras["bit_depth"]
        gains = _extras["gains"]

        with tarfile.open(self.filenames[0], "r") as fid:
            _data = np.frombuffer(
                fid.extractfile(sig_fname).read(),
                dtype=_extras["dtype"],
            )
        # TODO is the factor of 1000 (copied from the MATLAB code) to convert Volts
        # milliVolts? If so we should remove it (in MNE we always store in SI units, so
        # we should keep it in V)
        self._data = _data * (1000 * power_supply / 2**bit_depth / gains)[:, np.newaxis]


@fill_doc
def read_raw_otb(fname, verbose=None) -> RawOTB:
    """Reader for an OTB (.otb4/.otb+) recording.

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
