import numpy as np

from .peak_finder import peak_finder
from .. import fiff
from ..filter import band_pass_filter


def find_eog_events(raw, event_id=998):
    """Locate EOG artifacts

    Parameters
    ----------
    raw : instance of Raw
        The raw data
    event_id : int
        The index to assign to found events

    Returns
    -------
    eog_events : array
        Events
    """
    info = raw.info

    # Geting EOG Channel
    ch_EOG = fiff.pick_types(info, meg=False, eeg=False, stim=False,
                                                eog=True, ecg=False, emg=False)

    if len(ch_EOG) == 0:
        print 'No EOG channels found'
        print 'Trying with EEG 061 and EEG 062'
        ch_EOG = fiff.pick_channels(raw.ch_names,
                                        include=['EEG 061', 'EEG 062'])
        if len(ch_EOG) != 2:
            raise ValueError('EEG 61 or EEG 62 channel not found !!')

    print 'EOG channel index for this subject is: %s' % ch_EOG

    sampRate = info['sfreq']

    eog, _ = raw[ch_EOG, :]

    print ('Filtering the data to remove DC offset to help distinguish '
           'blinks from saccades')

    # filtering to remove dc offset so that we know which is blink and saccades
    filteog = np.array([band_pass_filter(x, sampRate, 2, 45) for x in eog])
    temp = np.sqrt(np.sum(filteog ** 2, axis=1))

    indexmax = np.argmax(temp)

    # easy to detect peaks with this filtering.
    filteog = band_pass_filter(eog[indexmax], sampRate, 1, 10)

    # detecting eog blinks and generating event file

    print 'Now detecting blinks and generating corresponding event file'

    temp = filteog - np.mean(filteog)
    if np.abs(np.max(temp)) > np.abs(np.min(temp)):
        eog_events, _ = peak_finder(filteog, extrema=1)
    else:
        eog_events, _ = peak_finder(filteog, extrema=-1)

    print 'Saving event file'
    n_events = len(eog_events)
    print "Number of EOG events detected : %d" % n_events
    eog_events = np.c_[eog_events + raw.first_samp, np.zeros(n_events),
                       event_id * np.ones(n_events)]

    return eog_events
