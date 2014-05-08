from ..externals.six import string_types
import numpy as np

from .. import pick_types, pick_channels
from ..utils import logger, verbose, sum_squared
from ..filter import band_pass_filter


def qrs_detector(sfreq, ecg, thresh_value=0.6, levels=2.5, n_thresh=3,
                 l_freq=5, h_freq=35, tstart=0, filter_length='10s'):
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
    tstart : float
        Start detection after tstart seconds.
    filter_length : str | int | None
        Number of taps to use for filtering.

    Returns
    -------
    events : array
        Indices of ECG peaks
    """
    win_size = int(round((60.0 * sfreq) / 120.0))

    filtecg = band_pass_filter(ecg, sfreq, l_freq, h_freq,
                               filter_length=filter_length)

    ecg_abs = np.abs(filtecg)
    init = int(sfreq)

    n_samples_start = int(sfreq * tstart)
    ecg_abs = ecg_abs[n_samples_start:]

    n_points = len(ecg_abs)

    maxpt = np.empty(3)
    maxpt[0] = np.max(ecg_abs[:init])
    maxpt[1] = np.max(ecg_abs[init:init * 2])
    maxpt[2] = np.max(ecg_abs[init * 2:init * 3])

    init_max = np.mean(maxpt)

    if thresh_value == 'auto':
        thresh_runs = np.arange(0.3, 1.1, 0.05)
    elif isinstance(thresh_value, string_types):
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
            window = ecg_abs[ii:ii + win_size]
            if window[0] > thresh1:
                max_time = np.argmax(window)
                time.append(ii + max_time)
                nx = np.sum(np.diff(((window > thresh1).astype(np.int)
                                     == 1).astype(int)))
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
        clean_events.append(ce)

    # pick the best threshold; first get effective heart rates
    rates = np.array([60. * len(ce) / (len(ecg) / float(sfreq))
                      for ce in clean_events])

    # now find heart rates that seem reasonable (infant thru adult athlete)
    idx = np.where(np.logical_and(rates <= 160., rates >= 40.))[0]
    if len(idx) > 0:
        ideal_rate = np.median(rates[idx])  # get close to the median
    else:
        ideal_rate = 80.  # get close to a reasonable default
    idx = np.argmin(np.abs(rates - ideal_rate))
    clean_events = clean_events[idx]
    return clean_events


@verbose
def find_ecg_events(raw, event_id=999, ch_name=None, tstart=0.0,
                    l_freq=5, h_freq=35, qrs_threshold='auto',
                    filter_length='10s', verbose=None):
    """Find ECG peaks

    Parameters
    ----------
    raw : instance of Raw
        The raw data
    event_id : int
        The index to assign to found events
    ch_name : str
        The name of the channel to use for ECG peak detection.
        The argument is mandatory if the dataset contains no ECG
        channels.
    tstart : float
        Start detection after tstart seconds. Useful when beginning
        of run is noisy.
    l_freq : float
        Low pass frequency.
    h_freq : float
        High pass frequency.
    qrs_threshold : float | str
        Between 0 and 1. qrs detection threshold. Can also be "auto" to
        automatically choose the threshold that generates a reasonable
        number of heartbeats (40-160 beats / min).
    filter_length : str | int | None
        Number of taps to use for filtering.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    ecg_events : array
        Events.
    ch_ECG : string
        Name of channel used.
    average_pulse : float
        Estimated average pulse.
    """
    info = raw.info

    # Geting ECG Channel
    if ch_name is None:
        ch_ECG = pick_types(info, meg=False, eeg=False, stim=False,
                            eog=False, ecg=True, emg=False, ref_meg=False,
                            exclude='bads')
    else:
        ch_ECG = pick_channels(raw.ch_names, include=[ch_name])
        if len(ch_ECG) == 0:
            raise ValueError('%s not in channel list (%s)' %
                             (ch_name, raw.ch_names))

    if len(ch_ECG) == 0 and ch_name is None:
        raise Exception('No ECG channel found. Please specify ch_name '
                        'parameter e.g. MEG 1531')

    assert len(ch_ECG) == 1

    logger.info('Using channel %s to identify heart beats'
                % raw.ch_names[ch_ECG[0]])

    ecg, times = raw[ch_ECG, :]

    # detecting QRS and generating event file
    ecg_events = qrs_detector(info['sfreq'], ecg.ravel(), tstart=tstart,
                              thresh_value=qrs_threshold, l_freq=l_freq,
                              h_freq=h_freq, filter_length=filter_length)

    n_events = len(ecg_events)
    average_pulse = n_events * 60.0 / (times[-1] - times[0])
    logger.info("Number of ECG events detected : %d (average pulse %d / "
                "min.)" % (n_events, average_pulse))

    ecg_events = np.c_[ecg_events + raw.first_samp, np.zeros(n_events),
                       event_id * np.ones(n_events)]
    return ecg_events, ch_ECG, average_pulse
