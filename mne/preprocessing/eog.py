# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from .peak_finder import peak_finder
from .. import pick_types, pick_channels
from ..utils import logger, verbose, _pl
from ..filter import filter_data
from ..epochs import Epochs
from ..externals.six import string_types


@verbose
def find_eog_events(raw, event_id=998, l_freq=1, h_freq=10,
                    filter_length='10s', ch_name=None, tstart=0,
                    verbose=None):
    """Locate EOG artifacts.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    event_id : int
        The index to assign to found events.
    l_freq : float
        Low cut-off frequency in Hz.
    h_freq : float
        High cut-off frequency in Hz.
    filter_length : str | int | None
        Number of taps to use for filtering.
    ch_name: str | None
        If not None, use specified channel(s) for EOG
    tstart : float
        Start detection after tstart seconds.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    eog_events : array
        Events.
    """
    # Getting EOG Channel
    eog_inds = _get_eog_channel_index(ch_name, raw)
    logger.info('EOG channel index for this subject is: %s' % eog_inds)

    eog, _ = raw[eog_inds, :]

    eog_events = _find_eog_events(eog, event_id=event_id, l_freq=l_freq,
                                  h_freq=h_freq,
                                  sampling_rate=raw.info['sfreq'],
                                  first_samp=raw.first_samp,
                                  filter_length=filter_length,
                                  tstart=tstart)

    return eog_events


def _find_eog_events(eog, event_id, l_freq, h_freq, sampling_rate, first_samp,
                     filter_length='10s', tstart=0.):
    """Helper function."""
    logger.info('Filtering the data to remove DC offset to help '
                'distinguish blinks from saccades')

    # filtering to remove dc offset so that we know which is blink and saccades
    fmax = np.minimum(45, sampling_rate / 2.0 - 0.75)  # protect Nyquist
    filteog = np.array([filter_data(
        x, sampling_rate, 2, fmax, None, filter_length, 0.5, 0.5,
        phase='zero-double', fir_window='hann') for x in eog])
    temp = np.sqrt(np.sum(filteog ** 2, axis=1))

    indexmax = np.argmax(temp)

    # easier to detect peaks with filtering.
    filteog = filter_data(
        eog[indexmax], sampling_rate, l_freq, h_freq, None,
        filter_length, 0.5, 0.5, phase='zero-double', fir_window='hann')

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
    eog_events = np.array([eog_events + first_samp,
                           np.zeros(n_events, int),
                           event_id * np.ones(n_events, int)]).T

    return eog_events


def _get_eog_channel_index(ch_name, inst):
    """Get EOG channel index."""
    if isinstance(ch_name, string_types):
        # Check if multiple EOG Channels
        if ',' in ch_name:
            ch_name = ch_name.split(',')
        else:
            ch_name = [ch_name]

        eog_inds = pick_channels(inst.ch_names, include=ch_name)

        if len(eog_inds) == 0:
            raise ValueError('%s not in channel list' % ch_name)
        else:
            logger.info('Using channel %s as EOG channel%s' % (
                        " and ".join(ch_name), _pl(eog_inds)))
    elif ch_name is None:

        eog_inds = pick_types(inst.info, meg=False, eeg=False, stim=False,
                              eog=True, ecg=False, emg=False, ref_meg=False,
                              exclude='bads')

        if len(eog_inds) == 0:
            logger.info('No EOG channels found')
            logger.info('Trying with EEG 061 and EEG 062')
            eog_inds = pick_channels(inst.ch_names,
                                     include=['EEG 061', 'EEG 062'])
            if len(eog_inds) != 2:
                raise RuntimeError('EEG 61 or EEG 62 channel not found !!')

    else:
        raise ValueError('Could not find EOG channel.')
    return eog_inds


@verbose
def create_eog_epochs(raw, ch_name=None, event_id=998, picks=None,
                      tmin=-0.5, tmax=0.5, l_freq=1, h_freq=10,
                      reject=None, flat=None, baseline=None,
                      preload=True, verbose=None):
    """Conveniently generate epochs around EOG artifact events.

    Parameters
    ----------
    raw : instance of Raw
        The raw data
    ch_name : str
        The name of the channel to use for EOG peak detection.
        The argument is mandatory if the dataset contains no EOG channels.
    event_id : int
        The index to assign to found events
    picks : array-like of int | None (default)
        Indices of channels to include (if None, all channels
        are used).
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    l_freq : float
        Low pass frequency.
    h_freq : float
        High pass frequency.
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
                          )

    flat : dict | None
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
    baseline : tuple or list of length 2, or None
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used. If None, no correction is applied.
    preload : bool
        Preload epochs or not.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    eog_epochs : instance of Epochs
        Data epoched around EOG events.
    """
    events = find_eog_events(raw, ch_name=ch_name, event_id=event_id,
                             l_freq=l_freq, h_freq=h_freq)

    # create epochs around EOG events
    eog_epochs = Epochs(raw, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax, proj=False, reject=reject,
                        flat=flat, picks=picks, baseline=baseline,
                        preload=preload)
    return eog_epochs
