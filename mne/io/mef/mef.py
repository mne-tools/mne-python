# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Read MEF3 files."""

import datetime as dt

import numpy as np

from ...annotations import Annotations
from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...utils import (
    _check_fname,
    _soft_import,
    _validate_type,
    fill_doc,
    logger,
    verbose,
    warn,
)
from ..base import BaseRaw
from ._utils import (
    _GMT_OFFSET_NO_ENTRY,
    _UUTC_NO_ENTRY,
    _get_consistent_value,
    _get_mef_units_scale,
    _mef_time_metadata_extras,
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
    <mne.io.Raw.set_channel_types>` to set appropriate types after loading
    (e.g., EEG for scalp electrodes, ECG for cardiac channels).
    Data are scaled to volts using MEF metadata (``units_conversion_factor`` and
    ``units_description``). If the unit description is missing or unsupported,
    the conversion factor is applied without additional unit scaling.
    Filter settings, session description, subject information, and record
    annotations are extracted when available. MEF discontinuities are mapped
    to ``BAD_ACQ_SKIP`` annotations based on the per-channel table of contents.

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

        fname = _check_fname(fname, "read", True, "fname", need_dir=True)
        if password is None:
            password = ""
        if isinstance(password, bytes):
            password = password.decode()
        _validate_type(password, str, "password")
        logger.info("Reading MEF3 file: %s", fname)

        # Open MEF session
        session = pymef.mef_session.MefSession(str(fname), password)

        # Get channel info
        ts_channels = session.session_md["time_series_channels"]
        if not ts_channels:
            raise ValueError("No time series channels found in MEF session.")

        ch_names = list(ts_channels.keys())
        n_channels = len(ch_names)

        logger.info("Found %d channels", n_channels)

        # Validate sampling rate and sample count consistency across channels
        sfreqs = []
        n_samples_list = []
        unit_scales = []
        acq_ch_numbers = []
        lowpass_vals = []
        highpass_vals = []
        notch_vals = []
        line_vals = []
        orig_units = {}
        missing_conv = []
        missing_units = []
        unknown_units = []
        for ch_name in ch_names:
            ch_md = ts_channels[ch_name]["section_2"]
            sfreq = _mef_get(ch_md, "sampling_frequency", kind="float")
            if sfreq is None or not np.isfinite(sfreq):
                raise ValueError(
                    f"MEF metadata missing sampling_frequency for {ch_name!r}."
                )
            n_samples = _mef_get(ch_md, "number_of_samples", kind="int")
            if n_samples is None:
                raise ValueError(
                    f"MEF metadata missing number_of_samples for {ch_name!r}."
                )
            sfreqs.append(float(sfreq))
            n_samples_list.append(int(n_samples))
            acq_ch_numbers.append(
                _mef_get(ch_md, "acquisition_channel_number", kind="int")
            )
            highpass_vals.append(
                _mef_get(
                    ch_md,
                    "low_frequency_filter_setting",
                    kind="float",
                    default=np.nan,
                )
            )
            lowpass_vals.append(
                _mef_get(
                    ch_md,
                    "high_frequency_filter_setting",
                    kind="float",
                    default=np.nan,
                )
            )
            notch_vals.append(
                _mef_get(
                    ch_md,
                    "notch_filter_frequency_setting",
                    kind="float",
                    default=np.nan,
                )
            )
            line_vals.append(
                _mef_get(ch_md, "AC_line_frequency", kind="float", default=np.nan)
            )
            scale, unit_desc, unit_desc_norm, ufact_valid, unit_known = (
                _get_mef_units_scale(
                    _mef_get(ch_md, "units_description"),
                    _mef_get(ch_md, "units_conversion_factor"),
                )
            )
            unit_scales.append(scale)
            if unit_desc:
                orig_units[ch_name] = unit_desc
            if not ufact_valid:
                missing_conv.append(ch_name)
            if not unit_desc_norm:
                missing_units.append(ch_name)
            elif not unit_known:
                unknown_units.append(unit_desc)
        if len(set(sfreqs)) != 1:
            raise ValueError("MEF channels have inconsistent sampling frequencies.")
        if len(set(n_samples_list)) != 1:
            raise ValueError("MEF channels have inconsistent number of samples.")
        sfreq = sfreqs[0]
        n_samples = n_samples_list[0]

        if missing_conv:
            warn(
                "MEF metadata has no units conversion factor for "
                f"{len(missing_conv)} channel(s); assuming a factor of 1."
            )
        if missing_units:
            warn(
                "MEF metadata has no units description for "
                f"{len(missing_units)} channel(s); assuming data are in volts "
                "after applying any conversion factor."
            )
        if unknown_units:
            unknown = ", ".join(sorted(set(unknown_units)))
            warn(
                "MEF metadata reports unsupported units description values "
                f"({unknown}); using the conversion factor without additional "
                "unit scaling."
            )

        logger.info("Sampling rate: %s Hz", sfreq)
        logger.info("Total samples: %d", n_samples)

        # Create info - default to sEEG, users can adjust with set_channel_types()
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
        for scale, ch in zip(unit_scales, info["chs"]):
            ch["cal"] = scale

        session_md = session.session_md
        session_specific = session_md.get("session_specific_metadata", {})
        start_uutc = _mef_get(
            session_specific, "earliest_start_time", kind="int"
        )
        if start_uutc in (None, _UUTC_NO_ENTRY):
            start_uutc = None
            starts = []
            if isinstance(ts_channels, dict):
                for ch_md in ts_channels.values():
                    ch_spec = ch_md.get("channel_specific_metadata", {})
                    ch_start = _mef_get(
                        ch_spec, "earliest_start_time", kind="int"
                    )
                    if ch_start is not None and ch_start != _UUTC_NO_ENTRY:
                        starts.append(ch_start)
            if starts:
                start_uutc = min(starts)
        meas_date = None
        if start_uutc is not None:
            meas_date = dt.datetime(
                1970, 1, 1, tzinfo=dt.timezone.utc
            ) + dt.timedelta(microseconds=int(start_uutc))
        ts_meta = session_md.get("time_series_metadata", {})
        session_section_2 = ts_meta.get("section_2")
        session_section_3 = ts_meta.get("section_3")
        if session_section_3 is None and isinstance(ts_channels, dict):
            first_md = next(iter(ts_channels.values()))
            session_section_3 = first_md.get("section_3")
        session_desc = _mef_get(
            session_section_2, "session_description", kind="text", default=""
        )
        gmt_offset = _mef_get(session_section_3, "GMT_offset", kind="int")
        utc_offset = None
        if gmt_offset is not None:
            if gmt_offset != _GMT_OFFSET_NO_ENTRY and abs(gmt_offset) <= 86400:
                sign = "+" if gmt_offset >= 0 else "-"
                gmt_offset = abs(gmt_offset)
                hours, remainder = divmod(gmt_offset, 3600)
                minutes = remainder // 60
                utc_offset = f"{sign}{hours:02d}{minutes:02d}"
        recording_location = _mef_get(
            session_section_3, "recording_location", kind="text", default=""
        )
        time_meta_extras = _mef_time_metadata_extras(session_section_3)

        subject_info = {}
        subject_id = _mef_get(
            session_section_3, "subject_ID", kind="text", default=""
        )
        if subject_id:
            subject_info["his_id"] = subject_id
        subject_name_1 = _mef_get(
            session_section_3, "subject_name_1", kind="text", default=""
        )
        subject_name_2 = _mef_get(
            session_section_3, "subject_name_2", kind="text", default=""
        )
        if subject_name_1:
            subject_info["first_name"] = subject_name_1
        if subject_name_2:
            subject_info["last_name"] = subject_name_2

        lowpass = _get_consistent_value(lowpass_vals, "lowpass")
        highpass = _get_consistent_value(highpass_vals, "highpass")
        line_freq = _get_consistent_value(line_vals, "line frequency")
        if line_freq is None:
            line_freq = _get_consistent_value(notch_vals, "notch filter frequency")

        with info._unlock():
            if meas_date is not None:
                info["meas_date"] = meas_date
            if utc_offset is not None:
                info["utc_offset"] = utc_offset
            if session_desc:
                info["description"] = session_desc
            if subject_info:
                info["subject_info"] = subject_info
            if recording_location:
                info["device_info"] = dict(site=recording_location)
            if highpass is not None:
                info["highpass"] = highpass
            if lowpass is not None:
                info["lowpass"] = lowpass
            if line_freq is not None:
                info["line_freq"] = line_freq

        if (
            info["highpass"] is not None
            and info["lowpass"] is not None
            and info["highpass"] > info["lowpass"]
        ):
            warn(
                "MEF highpass cutoff frequency is greater than lowpass; "
                "setting values to 0 and Nyquist."
            )
            info["highpass"] = 0.0
            info["lowpass"] = info["sfreq"] / 2.0

        for acq_num, ch in zip(acq_ch_numbers, info["chs"]):
            if acq_num is not None and acq_num > 0:
                ch["scanno"] = acq_num
                ch["logno"] = acq_num

        # Store extras for lazy loading
        raw_extras = dict(
            n_channels=n_channels,
            ch_names=ch_names,
            password=password,
            start_uutc=start_uutc,
        )

        super().__init__(
            info=info,
            last_samps=[n_samples - 1],
            filenames=[str(fname)],
            preload=preload,
            raw_extras=[raw_extras],
            orig_units=orig_units,
        )

        annotations_onset = []
        annotations_duration = []
        annotations_desc = []
        annotations_ch_names = []
        annotations_extras = []

        try:
            toc = session.get_channel_toc(ch_names[0])
        except Exception:
            toc = None
        gap_onsets, gap_durations = _toc_to_gap_annotations(toc, sfreq)
        annotations_onset.extend(gap_onsets)
        annotations_duration.extend(gap_durations)
        annotations_desc.extend(["BAD_ACQ_SKIP"] * len(gap_onsets))
        annotations_ch_names.extend([[]] * len(gap_onsets))
        annotations_extras.extend([{"source": "toc"}] * len(gap_onsets))

        if start_uutc is None:
            warn("MEF metadata has no start time; skipping record annotations.")
        else:
            (
                rec_onsets,
                rec_durations,
                rec_desc,
                rec_ch_names,
                rec_extras,
            ) = _records_to_annotations(session, ts_channels, start_uutc)
            for extras in rec_extras:
                extras["source"] = "record"
            annotations_onset.extend(rec_onsets)
            annotations_duration.extend(rec_durations)
            annotations_desc.extend(rec_desc)
            annotations_ch_names.extend(rec_ch_names)
            annotations_extras.extend(rec_extras)

        if time_meta_extras:
            annotations_onset.append(0.0)
            annotations_duration.append(0.0)
            annotations_desc.append("MEF_METADATA")
            annotations_ch_names.append([])
            time_meta_extras["source"] = "session_metadata"
            annotations_extras.append(time_meta_extras)

        if annotations_onset:
            order = np.argsort(annotations_onset)
            annotations = Annotations(
                onset=np.array(annotations_onset)[order],
                duration=np.array(annotations_duration)[order],
                description=np.array(annotations_desc, dtype=object)[order],
                ch_names=[annotations_ch_names[ii] for ii in order],
                orig_time=meas_date,
                extras=[annotations_extras[ii] for ii in order],
            )
            self.set_annotations(annotations)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        from pymef.mef_session import MefSession

        extras = self._raw_extras[fi]
        n_channels = extras["n_channels"]
        ch_names = extras["ch_names"]

        # Open MEF session
        session = MefSession(str(self._filenames[fi]), extras.get("password", ""))

        # Determine which channels to read
        if isinstance(idx, slice):
            ch_indices = range(*idx.indices(n_channels))
        else:
            ch_indices = idx

        selected_ch_names = [ch_names[i] for i in ch_indices]

        # Read data [start, stop) - pymef expects exclusive stop
        if stop <= start:
            raw_data = np.empty((len(selected_ch_names), 0), dtype=np.float64)
        else:
            raw_data = session.read_ts_channels_sample(
                selected_ch_names, [start, stop]
            )

        # Convert to numpy array
        raw_data = np.array(raw_data, dtype=np.float64)

        # Build output block
        block_out = np.zeros((n_channels, stop - start), dtype=data.dtype)
        block_out[idx] = raw_data

        _mult_cal_one(data, block_out, idx, cals, mult)
