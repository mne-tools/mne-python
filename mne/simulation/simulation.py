# author: ngayraud.
#
# Created on Tue Feb 20 11:14:54 2018.

import numpy as np

from .source import simulate_sparse_stc
from ..forward import apply_forward_raw
from ..utils import warn, logger
from ..io import RawArray
from ..io.constants import FIFF

from .waveforms import get_waveform
from .noise import generate_noise_data


class Simulation(dict):
    """Simulation of meg/eeg data.

    Parameters
    ----------
    fwd : Forward
        a forward solution containing an instance of Info and src
    n_dipoles : int
        Number of dipoles to simulate.
    labels : None | list of Labels
        The labels. The default is None, otherwise its size must be n_dipoles.
    location : str
        The label location to choose a dipole from. Can be ``random`` (default)
        or ``center`` to use :func:`mne.Label.center_of_mass`. Note that for
        ``center`` mode the label values are used as weights.
    subject : string | None
        The subject the label is defined for.
        Only used with location=``center``.
    subjects_dir : str, or None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using the
        environment variable SUBJECTS_DIR. Only used with location=``center``.
    waveform: list of callables/str of length n_dipoles | str | callable
        To simulate a waveform (activity) on each dipole. If it is a string or
        a callable, the same activity will be generated over all dipoles
    window_times : array | list | str
        time window(s) to generate activity. If list, its size should be
        len(waveform). If str, should be ``all`` (default)

    Notes
    -----
    Some notes.
    """

    def __init__(self, fwd, n_dipoles=2, labels=None, location='random',
                 subject=None, subjects_dir=None, waveform='sin',
                 window_times='all'):
        self.fwd = fwd  # TODO: check fwd
        labels, n_dipoles = self._get_sources(labels, n_dipoles)
        self.update(n_dipoles=n_dipoles, labels=labels, subject=subject,
                    subjects_dir=subjects_dir, location=location,
                    info=self.fwd['info'])
        self['info']['projs'] = []
        self['info']['bads'] = []
        self['info']['sfreq'] = None
        self.waveforms = self._get_waveform(waveform)
        self.window_times = self._get_window_times(window_times)

    def _get_sources(self, labels, n_dipoles):
        """Get labels and number of dipoles. Can be upgraded to do more.

        Return a list of labels or None and the number of dipoles.
        """
        if labels is None:
            return labels, n_dipoles

        n_labels = min(n_dipoles, len(labels))
        if n_dipoles != len(labels):
            warn('The number of labels is different from the number of '
                 'dipoles. %s dipole(s) will be generated.'
                 % n_labels)
        labels = labels[:n_labels]
        return labels, n_labels

    def _get_waveform(self, waveform):
        """Check the waveform given as imput wrt the number of dipoles.

        Return a list of callables.
        """
        if isinstance(waveform, str):
            return [get_waveform(waveform)]

        elif isinstance(waveform, list):

            if len(waveform) > self['n_dipoles']:
                warn('The number of waveforms is greater from the number of '
                     'dipoles. %s waveform(s) will be generated.'
                     % self['n_dipoles'])
                waveform = waveform[:self['n_dipoles']]

            elif len(waveform) < self['n_dipoles']:
                raise ValueError('Found fewer waveforms than dipoles.')
            return [get_waveform(f) for f in waveform]

        else:
            raise TypeError('Unrecognised type. Accepted inputs: str, list, '
                            'callable, or list containing any of the above.')

    def _check_window_time(self, w_t):
        """Check if window time has the correct value and frequency."""
        if isinstance(w_t, np.ndarray):
            freq = np.floor(1. / (w_t[-1] - w_t[-2]))
            _check_frequency(self['info'], freq, 'The frequency of the '
                             'time windows is not the same')
        elif w_t is 'all':
            pass
        else:
            raise TypeError('Unrecognised type. Accepted inputs: array, '
                            '\'all\'')
        return w_t

    def _get_window_times(self, window_times):
        """Get a list of window_times."""
        if isinstance(window_times, list):

            if len(window_times) > len(self.waveforms):
                n_func = len(self.waveforms)
                warn('The number of window times is greater than the number '
                     'of waveforms. %s waveform(s) will be generated.'
                     % n_func)
                window_times = window_times[:n_func]

            elif len(window_times) < len(self.waveforms):
                pad = len(self.waveforms) - len(window_times)
                warn('The number of window times is smaller than the number '
                     'of waveforms. Assuming that the last ones are \'all\'')
                window_times = window_times + ['all'] * pad
        else:
            window_times = [window_times]

        return [self._check_window_time(w_t) for w_t in window_times]


def _check_frequency(info, freq, error_message):
    """Compare two frequency values and assert they are the same."""
    if info['sfreq'] is not None:
        if info['sfreq'] != freq:
            raise ValueError(error_message)
    else:
        info['sfreq'] = freq
    return True


def _correct_window_times(w_t, e_t, times):
    """Check if window time has the correct length."""
    if (isinstance(w_t, str) and w_t == 'all') or e_t is None:
        return times
    else:
        if len(w_t) > len(times):
            warn('Window is too large, will be cut to match the '
                 'length of parameter \'times\'')
        return w_t[:len(times)]


def _iterate_simulation_sources(sim, events, times):
    """Iterate over all stimulation waveforms."""
    if len(sim.waveforms) == 1:
        yield (sim['n_dipoles'], sim['labels'],
               _correct_window_times(sim.window_times[0], events[0], times),
               events[0], sim.waveforms[0])
    else:
        dipoles = 1
        for index, waveform in enumerate(sim.waveforms):
            n_wt = min(index, len(sim.window_times) - 1)
            n_ev = min(index, len(events) - 1)
            labels = None
            if sim['labels'] is not None:
                labels = [sim['labels'][index]]
            yield (dipoles, labels,
                   _correct_window_times(sim.window_times[n_wt], events[n_ev],
                                         times), events[n_ev], waveform)


def _check_event(event, times):
    """Check if event array has the correct shape/length."""
    if isinstance(event, np.ndarray) and event.shape[1] == 3:
        if np.max(event) > len(times) - 1:
            warn('The indices in the event array is not the same as '
                 'the time points in the simulations.')
            event[np.where(event > len(times) - 1), 0] = len(times) - 1
        return np.array(event)
    elif event is not None:
        warn('Urecognized type. Will generated signal without events.')
    return None


def get_events(sim, times, events):
    """Get a list of events.

    Checks if the input events correspond to the simulation times.

    Parameters
    ----------
    sim : instance of Simulation
        Initialized Simulation object with parameters
    times : array
        Time array
    events : array,  | list of arrays | None
        events corresponding to some stimulation. If array, its size should be
        shape=(len(times), 3). If list, its size should be len(n_dipoles). If
        None, defaults to no event (default)

    Returns
    -------
    events : list | None
        a list of events of type array, shape=(n_events, 3)
    """
    if isinstance(events, list):
        n_waveforms = len(sim.waveforms)
        if len(events) > n_waveforms:
            warn('The number of event arrays is greater than the number '
                 'of waveforms. %s event arrays(s) will be generated.'
                 % n_waveforms)
            events = events[:n_waveforms]
        elif len(events) < n_waveforms:
            pad = len(sim.waveforms) - len(events)
            warn('The number of event arrays is smaller than the number '
                 'of waveforms. Assuming that the last ones are None')
            events = events + [None] * pad
    else:
        events = [events]

    return [_check_event(event, times) for event in events]


def simulate_raw_signal(sim, times, cov=None, events=None, random_state=None,
                        verbose=None):
    """Simulate a raw signal.

    Parameters
    ----------
    sim : instance of Simulation
        Initialized Simulation object with parameters
    times : array
        Time array
    cov : Covariance | string | dict | None
        Covariance of the noise
    events : array, shape = (n_events, 3) | list of arrays | None
        events corresponding to some stimulation.
        If list, its size should be len(n_dipoles)
        If None, defaults to no event (default)
    random_state : None | int | np.random.RandomState
        To specify the random generator state.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : instance of RawArray
        The simulated raw file.
    """
    if len(times) <= 2:  # to ensure event encoding works
        raise ValueError('stc must have at least three time points')

    info = sim['info'].copy()
    freq = np.floor(1. / (times[-1] - times[-2]))
    _check_frequency(info, freq, 'The frequency of the time windows is not '
                     'the same as the experiment time. ')
    info['projs'] = []
    info['lowpass'] = None

    # TODO: generate data for blinks and other physiological noise

    raw_data = np.zeros((len(info['ch_names']), len(times)))

    events = get_events(sim, times, events)

    logger.info('Simulating signal from %s sources' % sim['n_dipoles'])

    for dipoles, labels, window_time, event, data_fun in \
            _iterate_simulation_sources(sim, events, times):

        source_data = simulate_sparse_stc(sim.fwd['src'], dipoles,
                                          window_time, data_fun, labels,
                                          None, sim['location'],
                                          sim['subject'], sim['subjects_dir'])

        propagation = _get_propagation(event, times, window_time)
        source_data.data = np.dot(source_data.data, propagation)
        raw_data += apply_forward_raw(sim.fwd, source_data, info,
                                      verbose=verbose).get_data()

    # Noise
    if cov is not None:
        raw_data += generate_noise_data(info, cov, len(times), random_state)[0]

    # Add an empty stimulation channel
    raw_data = np.vstack((raw_data, np.zeros((1, len(times)))))
    stim_chan = dict(ch_name='STI 014', coil_type=FIFF.FIFFV_COIL_NONE,
                     kind=FIFF.FIFFV_STIM_CH, logno=len(info["chs"]) + 1,
                     scanno=len(info["chs"]) + 1, cal=1., range=1.,
                     loc=np.full(12, np.nan), unit=FIFF.FIFF_UNIT_NONE,
                     unit_mul=0., coord_frame=FIFF.FIFFV_COORD_UNKNOWN)
    info['chs'].append(stim_chan)
    info._update_redundant()

    # Create RawArray object with all data
    raw = RawArray(raw_data, info, first_samp=times[0], verbose=verbose)

    # Update the stimulation channel with stimulations
    stimulations = [event for event in events if event is not None]
    if len(stimulations) != 0:
        stimulations = np.unique(np.vstack(stimulations), axis=0)
        # Add events onto a stimulation channel
        raw.add_events(stimulations, stim_channel='STI 014')

    logger.info('Done')
    return raw


def _get_propagation(event, times, window_time):
    """Return the matrix that propagates the waveforms."""
    propagation = 1.0

    if event is not None:

        # generate stimulation timeline
        stimulation_timeline = np.zeros(len(times))
        stimulation_indices = np.array(event[:, 0], dtype=int)
        stimulation_timeline[stimulation_indices] = event[:, 2]

        from scipy.linalg import toeplitz
        # Create toeplitz array. Equivalent to convoluting the signal with the
        # stimulation timeline
        index = stimulation_timeline != 0
        trig = np.zeros((len(times)))
        trig[index] = 1
        propagation = toeplitz(trig[0:len(window_time)], trig)

    return propagation
