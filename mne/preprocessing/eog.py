import numpy as np

import logging
logger = logging.getLogger('mne')

from .peak_finder import peak_finder
from .. import fiff, verbose
from ..filter import band_pass_filter


@verbose
def find_eog_events(raw, event_id=998, l_freq=1, h_freq=10,
                    filter_length='10s', ch_name=None, tstart=0,
                    verbose=None):
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
    filter_length : str | int | None
        Number of taps to use for filtering.
    ch_name: str | None
        If not None, use specified channel(s) for EOG
    tstart : float
        Start detection after tstart seconds.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    eog_events : array
        Events.
    """

    info = raw.info

    # Getting EOG Channel
    if ch_name is None:
        ch_eog = fiff.pick_types(info, meg=False, eeg=False, stim=False,
                                 eog=True, ecg=False, emg=False,
                                 exclude='bads')
        if len(ch_eog) == 0:
            logger.info('No EOG channels found')
            logger.info('Trying with EEG 061 and EEG 062')
            ch_eog = fiff.pick_channels(raw.ch_names,
                                    include=['EEG 061', 'EEG 062'])
            if len(ch_eog) != 2:
                raise ValueError('EEG 61 or EEG 62 channel not found !!')

    else:

        # Check if multiple EOG Channels
        if ',' in ch_name:
            ch_name = ch_name.split(',')
        else:
            ch_name = [ch_name]

        ch_eog = fiff.pick_channels(raw.ch_names, include=ch_name)

        if len(ch_eog) == 0:
            raise ValueError('%s not in channel list' % ch_name)
        else:
            logger.info('Using channel %s as EOG channel%s' % (
                   " and ".join(ch_name), '' if len(ch_eog) < 2 else 's'))

    logger.info('EOG channel index for this subject is: %s' % ch_eog)

    eog, _ = raw[ch_eog, :]

    eog_events = _find_eog_events(eog, event_id=event_id, l_freq=l_freq,
                                  h_freq=h_freq,
                                  sampling_rate=raw.info['sfreq'],
                                  first_samp=raw.first_samp,
                                  filter_length=filter_length,
                                  tstart=tstart)

    return eog_events


def _find_eog_events(eog, event_id, l_freq, h_freq, sampling_rate, first_samp,
                     filter_length='10s', tstart=0.):
    """Helper function"""

    logger.info('Filtering the data to remove DC offset to help '
                'distinguish blinks from saccades')

    # filtering to remove dc offset so that we know which is blink and saccades
    filteog = np.array([band_pass_filter(x, sampling_rate, 2, 45,
                                         filter_length=filter_length)
                        for x in eog])
    temp = np.sqrt(np.sum(filteog ** 2, axis=1))

    indexmax = np.argmax(temp)

    # easier to detect peaks with filtering.
    filteog = band_pass_filter(eog[indexmax], sampling_rate, l_freq, h_freq,
                               filter_length=filter_length)

    # detecting eog blinks and generating event file

    logger.info('Now detecting blinks and generating corresponding events')

    temp = filteog - np.mean(filteog)
    n_samples_start = int(sampling_rate * tstart)
    if np.abs(np.max(temp)) > np.abs(np.min(temp)):
        eog_events, _ = peak_finder(filteog[n_samples_start:], extrema=1)
    else:
        eog_events, _ = peak_finder(filteog[n_samples_start:], extrema=-1)

    eog_events += n_samples_start
    n_events = len(eog_events)
    logger.info("Number of EOG events detected : %d" % n_events)
    eog_events = np.c_[eog_events + first_samp, np.zeros(n_events),
                       event_id * np.ones(n_events)]

    return eog_events