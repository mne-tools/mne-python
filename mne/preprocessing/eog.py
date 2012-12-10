import numpy as np

import logging
logger = logging.getLogger('mne')

from .peak_finder import peak_finder
from .. import fiff, verbose
from ..filter import band_pass_filter


@verbose
def find_eog_events(raw, event_id=998, l_freq=1, h_freq=10, verbose=None):
    """Locate EOG artifacts

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    event_id : int
        The index to assign to found events.
    low_pass : float
        Low pass frequency.
    high_pass : float
        High pass frequency.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    eog_events : array
        Events.
    """
    info = raw.info

    # Geting EOG Channel
    ch_EOG = fiff.pick_types(info, meg=False, eeg=False, stim=False,
                             eog=True, ecg=False, emg=False)

    if len(ch_EOG) == 0:
        logger.info('No EOG channels found')
        logger.info('Trying with EEG 061 and EEG 062')
        ch_EOG = fiff.pick_channels(raw.ch_names,
                                        include=['EEG 061', 'EEG 062'])
        if len(ch_EOG) != 2:
            raise ValueError('EEG 61 or EEG 62 channel not found !!')

    logger.info('EOG channel index for this subject is: %s' % ch_EOG)

    eog, _ = raw[ch_EOG, :]

    eog_events = _find_eog_events(eog, event_id=event_id, l_freq=l_freq,
                                  h_freq=h_freq,
                                  sampling_rate=raw.info['sfreq'],
                                  first_samp=raw.first_samp)

    return eog_events


def _find_eog_events(eog, event_id, l_freq, h_freq, sampling_rate, first_samp):
    """Helper function"""

    logger.info('Filtering the data to remove DC offset to help '
                'distinguish blinks from saccades')

    # filtering to remove dc offset so that we know which is blink and saccades
    filteog = np.array([band_pass_filter(x, sampling_rate, 2, 45)
                        for x in eog])
    temp = np.sqrt(np.sum(filteog ** 2, axis=1))

    indexmax = np.argmax(temp)

    # easier to detect peaks with filtering.
    filteog = band_pass_filter(eog[indexmax], sampling_rate, l_freq, h_freq)

    # detecting eog blinks and generating event file

    logger.info('Now detecting blinks and generating corresponding events')

    temp = filteog - np.mean(filteog)
    if np.abs(np.max(temp)) > np.abs(np.min(temp)):
        eog_events, _ = peak_finder(filteog, extrema=1)
    else:
        eog_events, _ = peak_finder(filteog, extrema=-1)

    n_events = len(eog_events)
    logger.info("Number of EOG events detected : %d" % n_events)
    eog_events = np.c_[eog_events + first_samp, np.zeros(n_events),
                       event_id * np.ones(n_events)]

    return eog_events
