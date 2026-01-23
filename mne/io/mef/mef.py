# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Read MEF3 files."""

import datetime as dt

import numpy as np

from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...annotations import Annotations
from ...utils import _check_fname, _soft_import, fill_doc, logger, verbose, warn
from ..base import BaseRaw
from ._utils import (
    _GMT_OFFSET_NO_ENTRY,
    _UUTC_NO_ENTRY,
    _get_mef_units_scale,
    _mef_get,
    _records_to_annotations,
    _toc_to_gap_annotations,
)


@fill_doc
@verbose
def read_raw_mef(fname, *, password="", preload=False, verbose=None):
    """Read raw data from MEF3 files.

    Parameters
    ----------
    fname : path-like
        Path to the MEF3 ``.mefd`` directory.
    password : str | bytes | None
        Password for encrypted MEF sessions. Use an empty string for
        unencrypted data.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawMEF
        A Raw object containing the MEF3 data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawMEF.

    Notes
    -----
    Data is read using the `pymef package <https://github.com/msel-source/pymef>`__.

    Channel types default to sEEG (stereo-EEG). Use :meth:`raw.set_channel_types()
    <mne.io.Raw.set_channel_types>` to set appropriate types after loading.

    Examples
    --------
    Read a MEF3 file::

        >>> raw = mne.io.read_raw_mef('recording.mefd')  # doctest: +SKIP
    """
    return RawMEF(fname, password=password, preload=preload, verbose=verbose)


@fill_doc
class RawMEF(BaseRaw):
    """Raw object for MEF3 files.

    Parameters
    ----------
    fname : path-like
        Path to the MEF3 ``.mefd`` directory.
    password : str | bytes | None
        Password for encrypted MEF sessions. Use an empty string for
        unencrypted data.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @verbose
    def __init__(self, fname, password="", *, preload=False, verbose=None):
        pymef = _soft_import("pymef", "reading MEF3 files", strict=True)

        # This type of dataset is a directory!
        fname = _check_fname(fname, "read", True, "fname", need_dir=True)
        # The dataset maybe have password
        password = (password or "").decode() if isinstance(password, bytes) else (password or "")
        # Open the dataset
        session = pymef.mef_session.MefSession(str(fname), password)

        # Get the time series channels
        ts_channels = session.session_md["time_series_channels"]
        if not ts_channels:
            raise ValueError("No time series channels found in MEF session.")

        ch_names = list(ts_channels.keys())
        logger.info("Reading %s: %d channels", fname, len(ch_names))

        # Extract per-channel metadata
        sfreqs, n_samples_list, unit_scales, orig_units = [], [], [], {}
        for ch_name in ch_names:
            ch_md = ts_channels[ch_name]["section_2"]
            sfreq = _mef_get(ch_md, "sampling_frequency", kind="float")
            n_samp = _mef_get(ch_md, "number_of_samples", kind="int")
            if not (sfreq and np.isfinite(sfreq)) or n_samp is None:
                raise ValueError(f"MEF metadata incomplete for {ch_name!r}.")
            sfreqs.append(sfreq)
            n_samples_list.append(n_samp)
            scale, unit_desc, *_ = _get_mef_units_scale(
                _mef_get(ch_md, "units_description"),
                _mef_get(ch_md, "units_conversion_factor"),
            )
            unit_scales.append(scale)
            if unit_desc:
                orig_units[ch_name] = unit_desc

        if len(set(sfreqs)) != 1 or len(set(n_samples_list)) != 1:
            raise ValueError("MEF channels have inconsistent sfreq or n_samples, "
                             "it is not supported for MNE the reading of this dataset.")

        sfreq, n_samples = sfreqs[0], n_samples_list[0]     

        # Create info
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
        for scale, ch in zip(unit_scales, info["chs"]):
            ch["cal"] = scale

        # Extract session metadata
        session_md = session.session_md
        ts_meta = session_md.get("time_series_metadata", {})

        # Section 3 is a dictionary with the session metadata
        # Documentation reference: 
        # https://github.com/msel-source/pymef/blob/89e1eb22847320b3585354b15d6361b71315bf33/pymef/mef_file/pymef3_file.h#L52
        sec3 = ts_meta.get("section_3") or next(iter(ts_channels.values()), {}).get("section_3")

        # Get start time
        start_uutc = _mef_get(session_md.get("session_specific_metadata", {}), "earliest_start_time", kind="int")
        if start_uutc in (None, _UUTC_NO_ENTRY):
            starts = [_mef_get(ch.get("channel_specific_metadata", {}), "earliest_start_time", kind="int")
                      for ch in ts_channels.values()]
            starts = [s for s in starts if s is not None and s != _UUTC_NO_ENTRY]
            start_uutc = min(starts) if starts else None

        # Indexing the start time
        meas_date = (dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) +
                     dt.timedelta(microseconds=int(start_uutc))) if start_uutc else None

        # Set info fields
        # following the style of how is make in the other mne io readers
        with info._unlock():
            if meas_date:
                info["meas_date"] = meas_date
            # Indexing the GMT offset
            gmt = _mef_get(sec3, "GMT_offset", kind="int")
            # If the GMT offset is not None and is not the default value and is not too large
            if gmt is not None and gmt != _GMT_OFFSET_NO_ENTRY and abs(gmt) <= 86400:
                info["utc_offset"] = f"{'+' if gmt >= 0 else '-'}{abs(gmt)//3600:02d}{(abs(gmt)%3600)//60:02d}"
            for key, mef_key in [("description", "session_description"), ("subject_info", None)]:
                # Parsing the subject info, and session description (officialt support by mne info)
                if key == "subject_info":
                    subj = {}
                    for k, m in [("his_id", "subject_ID"), ("first_name", "subject_name_1"), ("last_name", "subject_name_2")]:
                        v = _mef_get(sec3, m, kind="text", default="")
                        if v:
                            subj[k] = v
                    if subj:
                        info["subject_info"] = subj
                else:
                    # Parsing the session description (officialt support by mne info)
                    v = _mef_get(ts_meta.get("section_2"), mef_key, kind="text", default="")
                    if v:
                        info[key] = v

        super().__init__(
            info=info,
            last_samps=[n_samples - 1],
            filenames=[str(fname)],
            preload=preload,
            raw_extras=[dict(n_channels=len(ch_names), ch_names=ch_names, password=password)],
            orig_units=orig_units,
        )

        # Build annotations
        annotations_data = []
        try:
            toc = session.get_channel_toc(ch_names[0])
            gap_onsets, gap_durs = _toc_to_gap_annotations(toc, sfreq)
            annotations_data.extend((o, d, "BAD_ACQ_SKIP", [], {"source": "toc"})
                                    for o, d in zip(gap_onsets, gap_durs))
        except Exception:
            pass

        if start_uutc:
            rec_data = _records_to_annotations(session, ts_channels, start_uutc)
            annotations_data.extend((o, d, desc, chs, {**ex, "source": "record"})
                                    for o, d, desc, chs, ex in zip(*rec_data))

        if annotations_data:
            annotations_data.sort(key=lambda x: x[0])
            self.set_annotations(Annotations(
                onset=[a[0] for a in annotations_data],
                duration=[a[1] for a in annotations_data],
                description=[a[2] for a in annotations_data],
                ch_names=[a[3] for a in annotations_data],
                orig_time=meas_date,
            ))

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        from pymef.mef_session import MefSession

        extras = self._raw_extras[fi]
        ch_names = extras["ch_names"]
        session = MefSession(str(self._filenames[fi]), extras.get("password", ""))

        ch_indices = range(*idx.indices(extras["n_channels"])) if isinstance(idx, slice) else idx
        selected = [ch_names[i] for i in ch_indices]

        if stop > start:
            raw_data = np.array(session.read_ts_channels_sample(selected, [start, stop]), dtype=np.float64)
        else:
            raw_data = np.empty((len(selected), 0), dtype=np.float64)

        block_out = np.zeros((extras["n_channels"], stop - start), dtype=data.dtype)
        block_out[idx] = raw_data
        _mult_cal_one(data, block_out, idx, cals, mult)
