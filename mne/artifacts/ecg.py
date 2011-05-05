import numpy as np

from .. import fiff
from ..filter import band_pass_filter


def qrs_detector(sfreq, ecg, thresh_value=0.6, levels=2.5, n_thresh=3):
    """Detect QRS component in ECG channels.

    QRS is the main wave on the heart beat.

    Parameters
    ----------
    sfreq : float
        Sampling rate
    ecg : array
        ECG signal
    thresh_value: float
        qrs detection threshold
    levels: float
        number of std from mean to include for detection
    n_thresh: int
        max number of crossings

    Returns
    -------
    events : array
        Indices of ECG peaks
    """
    win_size = round((60.0 * sfreq) / 120.0)

    filtecg = band_pass_filter(ecg, sfreq, 5, 35)
    n_points = len(filtecg)

    absecg = np.abs(filtecg)
    init = int(sfreq)

    maxpt = np.empty(3)
    maxpt[0] = np.max(absecg[:init])
    maxpt[1] = np.max(absecg[init:init * 2])
    maxpt[2] = np.max(absecg[init * 2:init * 3])

    init_max = np.mean(maxpt)

    thresh1 = init_max * thresh_value

    numcross = []
    time = []
    rms = []
    i = 0
    while i < (n_points - win_size):
        window = absecg[i:i + win_size]
        if window[0] > thresh1:
            maxTime = np.argmax(window)
            time.append(i + maxTime)
            numcross.append(np.sum(np.diff(window > thresh1) == 1))
            rms.append(np.sqrt(np.mean(window ** 2)))
            i += win_size
        else:
            i += 1

    time = np.array(time)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    rms_thresh = rms_mean + (rms_std * levels)
    b = np.where(rms < rms_thresh)[0]
    a = np.array(numcross)[b]
    clean_events = time[b[a < n_thresh]]

    return clean_events


def find_ecg_events(raw, event_id=999):
    """Find ECG peaks

    Parameters
    ----------
    raw : instance of Raw
        The raw data
    event_id : int
        The index to assign to found events

    Returns
    -------
    ecg_events : array
        Events
    """
    info = raw.info

    # Geting ECG Channel
    ch_ECG = fiff.pick_types(info, meg=False, eeg=False, stim=False,
                                 eog=False, ecg=True, emg=False)

    if len(ch_ECG) == 0:
        # closest to the heart normally, In future we can search for it.
        ch_ECG = fiff.pick_channels(raw.ch_names, include='MEG 1531')
        print 'Using channel index %d to identify heart beats' % ch_ECG
    else:
        print 'ECG channel index for this subject is: %s' % ch_ECG

    assert len(ch_ECG) == 1
    ecg, times = raw[ch_ECG, :]

    # detecting QRS and generating event file
    ecg_events = qrs_detector(info['sfreq'], ecg.ravel())
    n_events = len(ecg_events)
    average_pulse = 60.0 * (times[-1] - times[0]) / n_events
    print ("Number of ECG events detected : %d (average pulse %d / min.)"
                                           % (n_events, average_pulse))

    ecg_events = np.c_[ecg_events + raw.first_samp, np.zeros(n_events),
                       event_id * np.ones(n_events)]
    return ecg_events
