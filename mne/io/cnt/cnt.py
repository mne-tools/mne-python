"""Conversion tool from Neuroscan CNT to FIF."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from os import path

import numpy as np

from ..._fiff._digitization import _make_dig_points
from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _empty_info
from ..._fiff.utils import _create_chs, _find_channels, _mult_cal_one, read_str
from ...annotations import Annotations
from ...channels.layout import _topo_to_sphere
from ...utils import _check_option, _explain_exception, _validate_type, fill_doc, warn
from ..base import BaseRaw
from ._utils import (
    CNTEventType3,
    _compute_robust_event_table_position,
    _get_event_parser,
    _read_teeg,
    _session_date_2_meas_date,
)


def _read_annotations_cnt(fname, data_format="int16"):
    """CNT Annotation File Reader.

    This method opens the .cnt files, searches all the metadata to construct
    the annotations and parses the event table. Notice that CNT files, can
    point to a different file containing the events. This case when the
    event table is separated from the main .cnt is not supported.

    Parameters
    ----------
    fname: path-like
        Path to CNT file containing the annotations.
    data_format : 'int16' | 'int32'
        Defines the data format the data is read in.

    Returns
    -------
    annot : instance of Annotations
        The annotations.
    """
    # Offsets from SETUP structure in http://paulbourke.net/dataformats/eeg/
    SETUP_NCHANNELS_OFFSET = 370
    SETUP_RATE_OFFSET = 376

    def _accept_reject_function(keypad_accept):
        accept_list = []
        for code in keypad_accept:
            if "xd0" in str(code):
                accept_list.append("good")
            elif "xc0" in str(code):
                accept_list.append("bad")
            else:
                accept_list.append("NA")
        return np.array(accept_list)

    def _translating_function(offset, n_channels, event_type, data_format=data_format):
        n_bytes = 2 if data_format == "int16" else 4
        if event_type == CNTEventType3:
            offset *= n_bytes * n_channels
        event_time = offset - 900 - (75 * n_channels)
        event_time //= n_channels * n_bytes
        event_time = event_time - 1
        # Prevent negative event times
        np.clip(event_time, 0, None, out=event_time)
        return event_time

    def _update_bad_span_onset(accept_reject, onset, duration, description):
        accept_reject = accept_reject.tolist()
        onset = onset.tolist()
        duration = duration.tolist()
        description = description.tolist()
        # If there are no bad spans, return original parameters
        if "bad" not in accept_reject:
            return np.array(onset), np.array(duration), np.array(description)
        # Create lists of bad and good span markers and onset
        bad_good_span_markers = [i for i in accept_reject if i in ["bad", "good"]]
        bad_good_onset = [
            onset[i]
            for i, value in enumerate(accept_reject)
            if value in ["bad", "good"]
        ]
        # Calculate duration of bad span
        first_bad_index = bad_good_span_markers.index("bad")
        duration_list = [
            bad_good_onset[i + 1] - bad_good_onset[i]
            for i in range(first_bad_index, len(bad_good_span_markers), 2)
        ]
        # Add bad event marker duration and description
        duration_list_index = 0
        for i in range(len(onset)):
            if accept_reject[i] == "bad":
                duration[i] = duration_list[duration_list_index]
                description[i] = "BAD_" + description[i]
                duration_list_index += 1
        # Remove good span markers
        final_onset, final_duration, final_description = [], [], []
        for i in range(len(accept_reject)):
            if accept_reject[i] != "good":
                final_onset.append(onset[i])
                final_duration.append(duration[i])
                final_description.append(description[i])
        return (
            np.array(final_onset),
            np.array(final_duration),
            np.array(final_description),
        )

    with open(fname, "rb") as fid:
        fid.seek(SETUP_NCHANNELS_OFFSET)
        (n_channels,) = np.frombuffer(fid.read(2), dtype="<u2")

        fid.seek(SETUP_RATE_OFFSET)
        (sfreq,) = np.frombuffer(fid.read(2), dtype="<u2")

        event_table_pos = _compute_robust_event_table_position(
            fid=fid, data_format=data_format
        )

    with open(fname, "rb") as fid:
        teeg = _read_teeg(fid, teeg_offset=event_table_pos)

    event_parser = _get_event_parser(event_type=teeg.event_type)

    with open(fname, "rb") as fid:
        fid.seek(event_table_pos + 9)  # the real table stats at +9
        buffer = fid.read(teeg.total_length)

    my_events = list(event_parser(buffer))

    if not my_events:
        return Annotations(list(), list(), list(), None)
    else:
        onset = _translating_function(
            np.array([e.Offset for e in my_events], dtype=float),
            n_channels=n_channels,
            event_type=type(my_events[0]),
            data_format=data_format,
        )
        # There is a Latency field but it's not useful for durations, see
        # https://github.com/mne-tools/mne-python/pull/11828
        duration = np.zeros(len(my_events), dtype=float)
        accept_reject = _accept_reject_function(
            np.array([e.KeyPad_Accept for e in my_events])
        )

        # Check to see if there are any button presses
        description = []
        for event in my_events:
            # Extract the 4-bit fields
            # Upper nibble (4 bits) currently not used
            # accept = (event.KeyPad_Accept[0] & 0xF0) >> 4
            # Lower nibble (4 bits) keypad button press
            keypad = event.KeyPad_Accept[0] & 0x0F
            if str(keypad) != "0":
                description.append(f"KeyPad Response {keypad}")
            elif event.KeyBoard != 0:
                description.append(f"Keyboard Response {event.KeyBoard}")
            else:
                description.append(str(event.StimType))

        description = np.array(description)

        onset, duration, description = _update_bad_span_onset(
            accept_reject, onset / sfreq, duration, description
        )
        return Annotations(
            onset=onset, duration=duration, description=description, orig_time=None
        )


@fill_doc
def read_raw_cnt(
    input_fname,
    eog=(),
    misc=(),
    ecg=(),
    emg=(),
    data_format="auto",
    date_format="mm/dd/yy",
    *,
    header="auto",
    preload=False,
    verbose=None,
) -> "RawCNT":
    """Read CNT data as raw object.

    .. Note::
        2d spatial coordinates (x, y) for EEG channels are read from the file
        header and fit to a sphere to compute corresponding z-coordinates.
        If channels assigned as EEG channels have locations
        far away from the head (i.e. x and y coordinates don't fit to a
        sphere), all the channel locations will be distorted
        (all channels that are not assigned with keywords ``eog``, ``ecg``,
        ``emg`` and ``misc`` are assigned as EEG channels). If you are not
        sure that the channel locations in the header are correct, it is
        probably safer to replace them with :meth:`mne.io.Raw.set_montage`.
        Montages can be created/imported with:

        - Standard montages with :func:`mne.channels.make_standard_montage`
        - Montages for `Compumedics systems
          <https://compumedicsneuroscan.com>`__ with
          :func:`mne.channels.read_dig_dat`
        - Other reader functions are listed under *See Also* at
          :class:`mne.channels.DigMontage`

    Parameters
    ----------
    input_fname : path-like
        Path to the data file.
    eog : list | tuple | ``'auto'`` | ``'header'``
        Names of channels or list of indices that should be designated
        EOG channels. If 'header', VEOG and HEOG channels assigned in the file
        header are used. If ``'auto'``, channel names containing ``'EOG'`` are
        used. Defaults to empty tuple.
    misc : list | tuple
        Names of channels or list of indices that should be designated
        MISC channels. Defaults to empty tuple.
    ecg : list | tuple | ``'auto'``
        Names of channels or list of indices that should be designated
        ECG channels. If ``'auto'``, the channel names containing ``'ECG'`` are
        used. Defaults to empty tuple.
    emg : list | tuple
        Names of channels or list of indices that should be designated
        EMG channels. If 'auto', the channel names containing 'EMG' are used.
        Defaults to empty tuple.
    data_format : ``'auto'`` | ``'int16'`` | ``'int32'``
        Defines the data format the data is read in. If ``'auto'``, it is
        determined from the file header using ``numsamples`` field.
        Defaults to ``'auto'``.
    date_format : ``'mm/dd/yy'`` | ``'dd/mm/yy'``
        Format of date in the header. Defaults to ``'mm/dd/yy'``.
    header : ``'auto'`` | ``'new'`` | ``'old'``
        Defines the header format. Used to describe how bad channels
        are formatted. If auto, reads using old and new header and
        if either contain a bad channel make channel bad.
        Defaults to ``'auto'``.

        .. versionadded:: 1.6
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawCNT.
        The raw data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawCNT.

    Notes
    -----
    .. versionadded:: 0.12
    """
    return RawCNT(
        input_fname,
        eog=eog,
        misc=misc,
        ecg=ecg,
        emg=emg,
        data_format=data_format,
        date_format=date_format,
        header=header,
        preload=preload,
        verbose=verbose,
    )


def _get_cnt_info(input_fname, eog, ecg, emg, misc, data_format, date_format, header):
    """Read the cnt header."""
    data_offset = 900  # Size of the 'SETUP' header.
    cnt_info = dict()
    # Reading only the fields of interest. Structure of the whole header at
    # http://paulbourke.net/dataformats/eeg/
    with open(input_fname, "rb", buffering=0) as fid:
        fid.seek(21)
        patient_id = read_str(fid, 20)
        patient_id = int(patient_id) if patient_id.isdigit() else 0
        fid.seek(121)
        patient_name = read_str(fid, 20).split()
        last_name = patient_name[0] if len(patient_name) > 0 else ""
        first_name = patient_name[-1] if len(patient_name) > 0 else ""
        fid.seek(2, 1)
        sex = read_str(fid, 1)
        if sex == "M":
            sex = FIFF.FIFFV_SUBJ_SEX_MALE
        elif sex == "F":
            sex = FIFF.FIFFV_SUBJ_SEX_FEMALE
        else:  # can be 'U'
            sex = FIFF.FIFFV_SUBJ_SEX_UNKNOWN
        hand = read_str(fid, 1)
        if hand == "R":
            hand = FIFF.FIFFV_SUBJ_HAND_RIGHT
        elif hand == "L":
            hand = FIFF.FIFFV_SUBJ_HAND_LEFT
        else:  # can be 'M' for mixed or 'U'
            hand = None
        fid.seek(205)
        session_label = read_str(fid, 20)

        session_date = f"{read_str(fid, 10)} {read_str(fid, 12)}"
        meas_date = _session_date_2_meas_date(session_date, date_format)

        fid.seek(370)
        n_channels = np.fromfile(fid, dtype="<u2", count=1).item()
        fid.seek(376)
        sfreq = np.fromfile(fid, dtype="<u2", count=1).item()
        if eog == "header":
            fid.seek(402)
            eog = [idx for idx in np.fromfile(fid, dtype="i2", count=2) if idx >= 0]
        fid.seek(438)
        lowpass_toggle = np.fromfile(fid, "i1", count=1).item()
        highpass_toggle = np.fromfile(fid, "i1", count=1).item()

        # Header has a field for number of samples, but it does not seem to be
        # too reliable. That's why we have option for setting n_bytes manually.
        fid.seek(864)
        n_samples = np.fromfile(fid, dtype="<u4", count=1).item()
        n_samples_header = n_samples
        fid.seek(869)
        lowcutoff = np.fromfile(fid, dtype="f4", count=1).item()
        fid.seek(2, 1)
        highcutoff = np.fromfile(fid, dtype="f4", count=1).item()

        event_offset = _compute_robust_event_table_position(
            fid=fid, data_format=data_format
        )
        fid.seek(890)
        cnt_info["continuous_seconds"] = np.fromfile(fid, dtype="<f4", count=1).item()

        if event_offset < data_offset:  # no events
            data_size = n_samples * n_channels
        else:
            data_size = event_offset - (data_offset + 75 * n_channels)

        _check_option("data_format", data_format, ["auto", "int16", "int32"])
        if data_format == "auto":
            if n_samples == 0 or data_size // (n_samples * n_channels) not in [2, 4]:
                warn(
                    "Could not define the number of bytes automatically. "
                    "Defaulting to 2."
                )
                n_bytes = 2
                n_samples = data_size // (n_bytes * n_channels)
                # See: PR #12393
                annotations = _read_annotations_cnt(input_fname, data_format="int16")
                # See: PR #12986
                if len(annotations) and annotations.onset[-1] * sfreq > n_samples:
                    n_bytes = 4
                    n_samples = n_samples_header
                    warn(
                        "Annotations are outside data range. "
                        "Changing data format to 'int32'."
                    )
            else:
                n_bytes = data_size // (n_samples * n_channels)
        else:
            n_bytes = 2 if data_format == "int16" else 4
            n_samples = data_size // (n_bytes * n_channels)

            # See PR #12393
            if n_samples_header != 0:
                n_samples = n_samples_header
        # Channel offset refers to the size of blocks per channel in the file.
        cnt_info["channel_offset"] = np.fromfile(fid, dtype="<i4", count=1).item()
        if cnt_info["channel_offset"] > 1:
            cnt_info["channel_offset"] //= n_bytes
        else:
            cnt_info["channel_offset"] = 1

        ch_names, cals, baselines, chs, pos = (list(), list(), list(), list(), list())

        bads = list()
        _validate_type(header, str, "header")
        _check_option("header", header, ("auto", "new", "old"))
        for ch_idx in range(n_channels):  # ELECTLOC fields
            fid.seek(data_offset + 75 * ch_idx)
            ch_name = read_str(fid, 10)
            ch_names.append(ch_name)

            # Some files have bad channels marked differently in the header.
            if header in ("new", "auto"):
                fid.seek(data_offset + 75 * ch_idx + 14)
                if np.fromfile(fid, dtype="u1", count=1).item():
                    bads.append(ch_name)
            if header in ("old", "auto"):
                fid.seek(data_offset + 75 * ch_idx + 4)
                if np.fromfile(fid, dtype="u1", count=1).item():
                    bads.append(ch_name)

            fid.seek(data_offset + 75 * ch_idx + 19)
            xy = np.fromfile(fid, dtype="f4", count=2)
            xy[1] *= -1  # invert y-axis
            pos.append(xy)
            fid.seek(data_offset + 75 * ch_idx + 47)
            # Baselines are subtracted before scaling the data.
            baselines.append(np.fromfile(fid, dtype="i2", count=1).item())
            fid.seek(data_offset + 75 * ch_idx + 59)
            sensitivity = np.fromfile(fid, dtype="f4", count=1).item()
            fid.seek(data_offset + 75 * ch_idx + 71)
            cal = np.fromfile(fid, dtype="f4", count=1).item()
            cals.append(cal * sensitivity * 1e-6 / 204.8)

    info = _empty_info(sfreq)
    if lowpass_toggle == 1:
        info["lowpass"] = highcutoff
    if highpass_toggle == 1:
        info["highpass"] = lowcutoff
    subject_info = {
        "hand": hand,
        "id": patient_id,
        "sex": sex,
        "first_name": first_name,
        "last_name": last_name,
    }
    subject_info = {key: val for key, val in subject_info.items() if val is not None}

    if eog == "auto":
        eog = _find_channels(ch_names, "EOG")
    if ecg == "auto":
        ecg = _find_channels(ch_names, "ECG")
    if emg == "auto":
        emg = _find_channels(ch_names, "EMG")

    chs = _create_chs(
        ch_names, cals, FIFF.FIFFV_COIL_EEG, FIFF.FIFFV_EEG_CH, eog, ecg, emg, misc
    )
    eegs = [idx for idx, ch in enumerate(chs) if ch["coil_type"] == FIFF.FIFFV_COIL_EEG]
    coords = _topo_to_sphere(pos, eegs)
    locs = np.full((len(chs), 12), np.nan)
    locs[:, :3] = coords
    dig = _make_dig_points(
        dig_ch_pos=dict(zip(ch_names, coords)),
        coord_frame="head",
        add_missing_fiducials=True,
    )
    for ch, loc in zip(chs, locs):
        ch.update(loc=loc)

    cnt_info.update(baselines=np.array(baselines), n_samples=n_samples, n_bytes=n_bytes)

    session_label = None if str(session_label) == "" else str(session_label)
    info.update(
        meas_date=meas_date,
        dig=dig,
        description=session_label,
        subject_info=subject_info,
        chs=chs,
    )
    info._unlocked = False
    info._update_redundant()
    info["bads"] = bads
    return info, cnt_info


@fill_doc
class RawCNT(BaseRaw):
    """Raw object from Neuroscan CNT file.

    .. note::

        The channel positions are read from the file header. Channels that are
        not assigned with keywords ``eog``, ``ecg``, ``emg`` and ``misc`` are
        assigned as eeg channels. All the eeg channel locations are fit to a
        sphere when computing the z-coordinates for the channels. If channels
        assigned as eeg channels have locations far away from the head (i.e.
        x and y coordinates don't fit to a sphere), all the channel locations
        will be distorted. If you are not sure that the channel locations in
        the header are correct, it is probably safer to use a (standard)
        montage. See :func:`mne.channels.make_standard_montage`

    .. note::

        A CNT file can also come from the EEG manufacturer ANT Neuro, in which case the
        function :func:`mne.io.read_raw_ant` should be used.

    Parameters
    ----------
    input_fname : path-like
        Path to the Neuroscan CNT file.
    eog : list | tuple
        Names of channels or list of indices that should be designated
        EOG channels. If ``'auto'``, the channel names beginning with
        ``EOG`` are used. Defaults to empty tuple.
    misc : list | tuple
        Names of channels or list of indices that should be designated
        MISC channels. Defaults to empty tuple.
    ecg : list | tuple
        Names of channels or list of indices that should be designated
        ECG channels. If ``'auto'``, the channel names beginning with
        ``ECG`` are used. Defaults to empty tuple.
    emg : list | tuple
        Names of channels or list of indices that should be designated
        EMG channels. If ``'auto'``, the channel names beginning with
        ``EMG`` are used. Defaults to empty tuple.
    data_format : ``'auto'`` | ``'int16'`` | ``'int32'``
        Defines the data format the data is read in. If ``'auto'``, it is
        determined from the file header using ``numsamples`` field.
        Defaults to ``'auto'``.
    date_format : ``'mm/dd/yy'`` | ``'dd/mm/yy'``
        Format of date in the header. Defaults to ``'mm/dd/yy'``.
    header : ``'auto'`` | ``'new'`` | ``'old'``
        Defines the header format. Used to describe how bad channels
        are formatted. If auto, reads using old and new header and
        if either contain a bad channel make channel bad.
        Defaults to ``'auto'``.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    def __init__(
        self,
        input_fname,
        eog=(),
        misc=(),
        ecg=(),
        emg=(),
        data_format="auto",
        date_format="mm/dd/yy",
        *,
        header="auto",
        preload=False,
        verbose=None,
    ):
        _check_option("date_format", date_format, ["mm/dd/yy", "dd/mm/yy"])
        if date_format == "dd/mm/yy":
            _date_format = "%d/%m/%y %H:%M:%S"
        else:
            _date_format = "%m/%d/%y %H:%M:%S"

        input_fname = path.abspath(input_fname)
        try:
            info, cnt_info = _get_cnt_info(
                input_fname, eog, ecg, emg, misc, data_format, _date_format, header
            )
        except Exception:
            raise RuntimeError(
                f"{_explain_exception()}\n"
                "WARNING: mne.io.read_raw_cnt "
                "supports Neuroscan CNT files only. If this file is an ANT Neuro CNT, "
                "please use mne.io.read_raw_ant instead."
            )
        last_samps = [cnt_info["n_samples"] - 1]
        super().__init__(
            info,
            preload,
            filenames=[input_fname],
            raw_extras=[cnt_info],
            last_samps=last_samps,
            orig_format="int",
            verbose=verbose,
        )

        data_format = "int32" if cnt_info["n_bytes"] == 4 else "int16"
        self.set_annotations(
            _read_annotations_cnt(input_fname, data_format=data_format)
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Take a chunk of raw data, multiply by mult or cals, and store."""
        n_channels = self._raw_extras[fi]["orig_nchan"]
        if "stim_channel" in self._raw_extras[fi]:
            f_channels = n_channels - 1  # Stim channel already read.
            stim_ch = self._raw_extras[fi]["stim_channel"]
        else:
            f_channels = n_channels
            stim_ch = None

        channel_offset = self._raw_extras[fi]["channel_offset"]
        baselines = self._raw_extras[fi]["baselines"]
        n_bytes = self._raw_extras[fi]["n_bytes"]
        n_samples = self._raw_extras[fi]["n_samples"]
        dtype = "<i4" if n_bytes == 4 else "<i2"
        chunk_size = channel_offset * f_channels  # Size of chunks in file.
        # The data is divided into blocks of samples / channel.
        # channel_offset determines the amount of successive samples.
        # Here we use sample offset to align the data because start can be in
        # the middle of these blocks.
        data_left = (stop - start) * f_channels
        # Read up to 100 MB of data at a time, block_size is in data samples
        block_size = ((int(100e6) // n_bytes) // chunk_size) * chunk_size
        block_size = min(data_left, block_size)
        s_offset = start % channel_offset
        with open(self.filenames[fi], "rb", buffering=0) as fid:
            fid.seek(900 + f_channels * (75 + (start - s_offset) * n_bytes))
            for sample_start in np.arange(0, data_left, block_size) // f_channels:
                # Earlier comment says n_samples is unreliable, but I think it
                # is because it needed to be changed to unsigned int
                # See: PR #12393
                sample_stop = sample_start + min(
                    (
                        n_samples,
                        block_size // f_channels,
                        data_left // f_channels - sample_start,
                    )
                )
                n_samps = sample_stop - sample_start
                one = np.zeros((n_channels, n_samps))

                # In case channel offset and start time do not align perfectly,
                # extra sample sets are read here to cover the desired time
                # window. The whole (up to 100 MB) block is read at once and
                # then reshaped to (n_channels, n_samples).
                extra_samps = (
                    chunk_size
                    if (s_offset != 0 or n_samps % channel_offset != 0)
                    else 0
                )
                if s_offset >= (channel_offset / 2):  # Extend at the end.
                    extra_samps += chunk_size
                count = n_samps // channel_offset * chunk_size + extra_samps
                n_chunks = count // chunk_size
                samps = np.fromfile(fid, dtype=dtype, count=count)
                samps = samps.reshape((n_chunks, f_channels, channel_offset), order="C")

                # Intermediate shaping to chunk sizes.
                block = np.zeros((n_channels, channel_offset * n_chunks))
                for set_idx, row in enumerate(samps):  # Final shape.
                    block_slice = slice(
                        set_idx * channel_offset, (set_idx + 1) * channel_offset
                    )
                    block[:f_channels, block_slice] = row
                if "stim_channel" in self._raw_extras[fi]:
                    _data_start = start + sample_start
                    _data_stop = start + sample_stop
                    block[-1] = stim_ch[_data_start:_data_stop]
                one[idx] = block[idx, s_offset : n_samps + s_offset]

                one[idx] -= baselines[idx][:, None]
                _mult_cal_one(data[:, sample_start:sample_stop], one, idx, cals, mult)
