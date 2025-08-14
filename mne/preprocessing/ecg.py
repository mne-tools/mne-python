# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.meas_info import create_info
from .._fiff.pick import _picks_to_idx, pick_channels, pick_types
from ..annotations import _annotations_starts_stops
from ..epochs import BaseEpochs, Epochs
from ..evoked import Evoked
from ..filter import filter_data
from ..io import BaseRaw, RawArray
from ..utils import int_like, logger, sum_squared, verbose, warn


@verbose
def qrs_detector(
    sfreq,
    ecg,
    thresh_value=0.6,
    levels=2.5,
    n_thresh=3,
    l_freq=5,
    h_freq=35,
    tstart=0,
    filter_length="10s",
    verbose=None,
):
    """Detect QRS component in ECG channels.

    QRS is the main wave on the heart beat.

    Parameters
    ----------
    sfreq : float
        Sampling rate
    ecg : array
        ECG signal
    thresh_value : float | str
        qrs detection threshold. Can also be "auto" for automatic
        selection of threshold.
    levels : float
        number of std from mean to include for detection
    n_thresh : int
        max number of crossings
    l_freq : float
        Low pass frequency
    h_freq : float
        High pass frequency
    %(tstart_ecg)s
    %(filter_length_ecg)s
    %(verbose)s

    Returns
    -------
    events : array
        Indices of ECG peaks.
    """
    win_size = int(round((60.0 * sfreq) / 120.0))

    filtecg = filter_data(
        ecg,
        sfreq,
        l_freq,
        h_freq,
        None,
        filter_length,
        0.5,
        0.5,
        phase="zero-double",
        fir_window="hann",
        fir_design="firwin2",
    )

    ecg_abs = np.abs(filtecg)
    init = int(sfreq)

    n_samples_start = int(sfreq * tstart)
    ecg_abs = ecg_abs[n_samples_start:]

    n_points = len(ecg_abs)

    maxpt = np.empty(3)
    maxpt[0] = np.max(ecg_abs[:init])
    maxpt[1] = np.max(ecg_abs[init : init * 2])
    maxpt[2] = np.max(ecg_abs[init * 2 : init * 3])

    init_max = np.mean(maxpt)

    if thresh_value == "auto":
        thresh_runs = np.arange(0.3, 1.1, 0.05)
    elif isinstance(thresh_value, str):
        raise ValueError('threshold value must be "auto" or a float')
    else:
        thresh_runs = [thresh_value]

    # Try a few thresholds (or just one)
    clean_events = list()
    for thresh_value in thresh_runs:
        thresh1 = init_max * thresh_value
        numcross = list()
        time = list()
        rms = list()
        ii = 0
        while ii < (n_points - win_size):
            window = ecg_abs[ii : ii + win_size]
            if window[0] > thresh1:
                max_time = np.argmax(window)
                time.append(ii + max_time)
                nx = np.sum(
                    np.diff(((window > thresh1).astype(np.int64) == 1).astype(int))
                )
                numcross.append(nx)
                rms.append(np.sqrt(sum_squared(window) / window.size))
                ii += win_size
            else:
                ii += 1

        if len(rms) == 0:
            rms.append(0.0)
            time.append(0.0)
        time = np.array(time)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_thresh = rms_mean + (rms_std * levels)
        b = np.where(rms < rms_thresh)[0]
        a = np.array(numcross)[b]
        ce = time[b[a < n_thresh]]

        ce += n_samples_start
        if ce.size > 0:  # We actually found an event
            clean_events.append(ce)

    if clean_events:
        # pick the best threshold; first get effective heart rates
        rates = np.array(
            [60.0 * len(cev) / (len(ecg) / float(sfreq)) for cev in clean_events]
        )

        # now find heart rates that seem reasonable (infant through adult
        # athlete)
        idx = np.where(np.logical_and(rates <= 160.0, rates >= 40.0))[0]
        if idx.size > 0:
            ideal_rate = np.median(rates[idx])  # get close to the median
        else:
            ideal_rate = 80.0  # get close to a reasonable default

        idx = np.argmin(np.abs(rates - ideal_rate))
        clean_events = clean_events[idx]
    else:
        clean_events = np.array([])

    return clean_events


@verbose
def find_ecg_events(
    raw,
    event_id=999,
    ch_name=None,
    tstart=0.0,
    l_freq=5,
    h_freq=35,
    qrs_threshold="auto",
    filter_length="10s",
    return_ecg=False,
    reject_by_annotation=True,
    verbose=None,
):
    """Find ECG events by localizing the R wave peaks.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    %(event_id_ecg)s
    %(ch_name_ecg)s
    %(tstart_ecg)s
    %(l_freq_ecg_filter)s
    qrs_threshold : float | str
        Between 0 and 1. qrs detection threshold. Can also be "auto" to
        automatically choose the threshold that generates a reasonable
        number of heartbeats (40-160 beats / min).
    %(filter_length_ecg)s
    return_ecg : bool
        Return the ECG data. This is especially useful if no ECG channel
        is present in the input data, so one will be synthesized (only works if MEG
        channels are present in the data). Defaults to ``False``.
    %(reject_by_annotation_all)s

        .. versionadded:: 0.18
    %(verbose)s

    Returns
    -------
    ecg_events : array
        The events corresponding to the peaks of the R waves.
    ch_ecg : int | None
        Index of channel used.
    average_pulse : float
        The estimated average pulse. If no ECG events could be found, this will
        be zero.
    ecg : array | None
        The ECG data of the synthesized ECG channel, if any. This will only
        be returned if ``return_ecg=True`` was passed.

    See Also
    --------
    create_ecg_epochs
    compute_proj_ecg
    """
    skip_by_annotation = ("edge", "bad") if reject_by_annotation else ()
    del reject_by_annotation
    idx_ecg = _get_ecg_channel_index(ch_name, raw)
    if idx_ecg is not None:
        logger.info(f"Using channel {raw.ch_names[idx_ecg]} to identify heart beats.")
        ecg = raw.get_data(picks=idx_ecg)
    else:
        ecg, _ = _make_ecg(raw, start=None, stop=None)
    assert ecg.ndim == 2 and ecg.shape[0] == 1
    ecg = ecg[0]
    # Deal with filtering the same way we do in raw, i.e. filter each good
    # segment
    onsets, ends = _annotations_starts_stops(
        raw, skip_by_annotation, "reject_by_annotation", invert=True
    )
    ecgs = list()
    max_idx = (ends - onsets).argmax()
    for si, (start, stop) in enumerate(zip(onsets, ends)):
        # Only output filter params once (for info level), and only warn
        # once about the length criterion (longest segment is too short)
        use_verbose = verbose if si == max_idx else "error"
        ecgs.append(
            filter_data(
                ecg[start:stop],
                raw.info["sfreq"],
                l_freq,
                h_freq,
                [0],
                filter_length,
                0.5,
                0.5,
                1,
                "fir",
                None,
                copy=False,
                phase="zero-double",
                fir_window="hann",
                fir_design="firwin2",
                verbose=use_verbose,
            )
        )
    ecg = np.concatenate(ecgs)

    # detecting QRS and generating events. Since not user-controlled, don't
    # output filter params here (hardcode verbose=False)
    ecg_events = qrs_detector(
        raw.info["sfreq"],
        ecg,
        tstart=tstart,
        thresh_value=qrs_threshold,
        l_freq=None,
        h_freq=None,
        verbose=False,
    )

    # map ECG events back to original times
    remap = np.empty(len(ecg), int)
    offset = 0
    for start, stop in zip(onsets, ends):
        this_len = stop - start
        assert this_len >= 0
        remap[offset : offset + this_len] = np.arange(start, stop)
        offset += this_len
    assert offset == len(ecg)

    if ecg_events.size > 0:
        ecg_events = remap[ecg_events]
    else:
        ecg_events = np.array([])

    n_events = len(ecg_events)
    duration_sec = len(ecg) / raw.info["sfreq"] - tstart
    duration_min = duration_sec / 60.0
    average_pulse = n_events / duration_min
    logger.info(
        f"Number of ECG events detected : {n_events} "
        f"(average pulse {average_pulse} / min.)"
    )

    ecg_events = np.array(
        [
            ecg_events + raw.first_samp,
            np.zeros(n_events, int),
            event_id * np.ones(n_events, int),
        ]
    ).T

    out = (ecg_events, idx_ecg, average_pulse)
    ecg = ecg[np.newaxis]  # backward compat output 2D
    if return_ecg:
        out += (ecg,)
    return out


def _get_ecg_channel_index(ch_name, inst):
    """Get ECG channel index, if no channel found returns None."""
    if ch_name is None:
        ecg_idx = pick_types(
            inst.info,
            meg=False,
            eeg=False,
            stim=False,
            eog=False,
            ecg=True,
            emg=False,
            ref_meg=False,
            exclude="bads",
        )
    else:
        if ch_name not in inst.ch_names:
            raise ValueError(f"{ch_name} not in channel list ({inst.ch_names})")
        ecg_idx = pick_channels(inst.ch_names, include=[ch_name])

    if len(ecg_idx) == 0:
        return None

    if len(ecg_idx) > 1:
        warn(
            f"More than one ECG channel found. Using only {inst.ch_names[ecg_idx[0]]}."
        )

    return ecg_idx[0]


@verbose
def create_ecg_epochs(
    raw,
    ch_name=None,
    event_id=999,
    picks=None,
    tmin=-0.5,
    tmax=0.5,
    l_freq=8,
    h_freq=16,
    reject=None,
    flat=None,
    baseline=None,
    preload=True,
    keep_ecg=False,
    reject_by_annotation=True,
    decim=1,
    verbose=None,
):
    """Conveniently generate epochs around ECG artifact events.

    %(create_ecg_epochs)s

    .. note:: Filtering is only applied to the ECG channel while finding
                events. The resulting ``ecg_epochs`` will have no filtering
                applied (i.e., have the same filter properties as the input
                ``raw`` instance).

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    %(ch_name_ecg)s
    %(event_id_ecg)s
    %(picks_all)s
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    %(l_freq_ecg_filter)s
    %(reject_epochs)s
    %(flat)s
    %(baseline_epochs)s
    preload : bool
        Preload epochs or not (default True). Must be True if
        keep_ecg is True.
    keep_ecg : bool
        When ECG is synthetically created (after picking), should it be added
        to the epochs? Must be False when synthetic channel is not used.
        Defaults to False.
    %(reject_by_annotation_epochs)s

        .. versionadded:: 0.14.0
    %(decim)s

        .. versionadded:: 0.21.0
    %(verbose)s

    Returns
    -------
    ecg_epochs : instance of Epochs
        Data epoched around ECG R wave peaks.

    See Also
    --------
    find_ecg_events
    compute_proj_ecg

    Notes
    -----
    If you already have a list of R-peak times, or want to compute R-peaks
    outside MNE-Python using a different algorithm, the recommended approach is
    to call the :class:`~mne.Epochs` constructor directly, with your R-peaks
    formatted as an :term:`events` array (here we also demonstrate the relevant
    default values)::

        mne.Epochs(raw, r_peak_events_array, tmin=-0.5, tmax=0.5,
                   baseline=None, preload=True, proj=False)  # doctest: +SKIP
    """
    has_ecg = "ecg" in raw or ch_name is not None
    if keep_ecg and (has_ecg or not preload):
        raise ValueError(
            "keep_ecg can be True only if the ECG channel is "
            "created synthetically and preload=True."
        )

    events, _, _, ecg = find_ecg_events(
        raw,
        ch_name=ch_name,
        event_id=event_id,
        l_freq=l_freq,
        h_freq=h_freq,
        return_ecg=True,
        reject_by_annotation=reject_by_annotation,
    )

    picks = _picks_to_idx(raw.info, picks, "all", exclude=())

    # create epochs around ECG events and baseline (important)
    ecg_epochs = Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=False,
        flat=flat,
        picks=picks,
        reject=reject,
        baseline=baseline,
        reject_by_annotation=reject_by_annotation,
        preload=preload,
        decim=decim,
    )

    if keep_ecg:
        # We know we have created a synthetic channel and epochs are preloaded
        ecg_raw = RawArray(
            ecg,
            create_info(
                ch_names=["ECG-SYN"], sfreq=raw.info["sfreq"], ch_types=["ecg"]
            ),
            first_samp=raw.first_samp,
        )
        with ecg_raw.info._unlock():
            ignore = ["ch_names", "chs", "nchan", "bads"]
            for k, v in raw.info.items():
                if k not in ignore:
                    ecg_raw.info[k] = v
        syn_epochs = Epochs(
            ecg_raw,
            events=ecg_epochs.events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            proj=False,
            picks=[0],
            baseline=baseline,
            decim=decim,
            preload=True,
        )
        ecg_epochs = ecg_epochs.add_channels([syn_epochs])

    return ecg_epochs


@verbose
def _make_ecg(inst, start, stop, reject_by_annotation=False, verbose=None):
    """Create ECG signal from cross channel average."""
    if not any(c in inst for c in ["mag", "grad"]):
        raise ValueError(
            "Generating an artificial ECG channel can only be done for MEG data."
        )
    for ch in ["mag", "grad"]:
        if ch in inst:
            break
    logger.info(
        "Reconstructing ECG signal from {}".format(
            {"mag": "Magnetometers", "grad": "Gradiometers"}[ch]
        )
    )
    picks = pick_types(inst.info, meg=ch, eeg=False, ref_meg=False)

    # Handle start/stop
    msg = (
        "integer arguments for the start and stop parameters are "
        "not supported for Epochs and Evoked objects. Please "
        "consider using float arguments specifying start and stop "
        "time in seconds."
    )
    begin_param_name = "tmin"
    if isinstance(start, int_like):
        if isinstance(inst, BaseRaw):
            # Raw has start param, can just use int
            begin_param_name = "start"
        else:
            raise ValueError(msg)

    end_param_name = "tmax"
    if isinstance(start, int_like):
        if isinstance(inst, BaseRaw):
            # Raw has stop param, can just use int
            end_param_name = "stop"
        else:
            raise ValueError(msg)

    kwargs = {begin_param_name: start, end_param_name: stop}

    if isinstance(inst, BaseRaw):
        reject_by_annotation = "omit" if reject_by_annotation else None
        ecg, times = inst.get_data(
            picks,
            return_times=True,
            **kwargs,
            reject_by_annotation=reject_by_annotation,
        )
    elif isinstance(inst, BaseEpochs):
        ecg = np.hstack(inst.copy().get_data(picks, **kwargs))
        times = inst.times
    elif isinstance(inst, Evoked):
        ecg = inst.get_data(picks, **kwargs)
        times = inst.times
    return ecg.mean(0, keepdims=True), times
