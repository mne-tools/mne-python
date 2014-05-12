
"""Tools for creating MNE objects from numpy arrays"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ...constants import FIFF
from ..meas_info import Info
from ..base import _BaseRaw
from ...epochs import Epochs
from ...utils import verbose, logger, _check_type_picks
from ...externals.six import string_types


_kind_dict = dict(
    eeg=(FIFF.FIFFV_EEG_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    mag=(FIFF.FIFFV_MEG_CH, FIFF.FIFFV_COIL_VV_MAG_T3, FIFF.FIFF_UNIT_T),
    grad=(FIFF.FIFFV_MEG_CH, FIFF.FIFFV_COIL_VV_PLANAR_T1, FIFF.FIFF_UNIT_T_M),
    misc=(FIFF.FIFFV_MISC_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_NONE),
    stim=(FIFF.FIFFV_STIM_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    eog=(FIFF.FIFFV_EOG_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    ecg=(FIFF.FIFFV_ECG_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
)


def create_info(ch_names, sfreq, ch_types=None):
    """Create a basic Info instance suitable for use with create_raw

    Parameters
    ----------
    ch_names : list of str
        Channel names.
    sfreq : float
        Sample rate of the data.
    ch_types : list of str
        Channel types. If None, data are assumed to be misc.
        Currently supported fields are "mag", "grad", "eeg", and "misc".

    Notes
    -----
    The info dictionary will be sparsely populated to enable functionality
    within the rest of the package. Advanced functionality such as source
    localization can only be obtained through substantial, proper
    modifications of the info structure (not recommended).
    """
    if not isinstance(ch_names, (list, tuple)):
        raise TypeError('ch_names must be a list or tuple')
    sfreq = float(sfreq)
    if sfreq <= 0:
        raise ValueError('sfreq must be positive')
    nchan = len(ch_names)
    if ch_types is None:
        ch_types = [_kind_dict['misc']] * nchan
    if len(ch_types) != nchan:
        raise ValueError('ch_types and ch_names must be the same length')
    info = Info()
    info['meas_date'] = [0, 0]
    info['sfreq'] = sfreq
    for key in ['bads', 'projs', 'comps']:
        info[key] = list()
    for key in ['meas_id', 'file_id', 'highpass', 'lowpass', 'acq_pars',
                'acq_stim', 'filename', 'dig']:
        info[key] = None
    info['ch_names'] = ch_names
    info['nchan'] = nchan
    info['chs'] = list()
    loc = np.concatenate((np.zeros(3), np.eye(3).ravel())).astype(np.float32)
    for ci, (name, kind) in enumerate(zip(ch_names, ch_types)):
        if not isinstance(name, string_types):
            raise TypeError('each entry in ch_names must be a string')
        if not isinstance(kind, string_types):
            raise TypeError('each entry in ch_names must be a string')
        if kind not in _kind_dict:
            raise KeyError('kind must be one of %s, not %s'
                           % (list(_kind_dict.keys()), kind))
        kind = _kind_dict[kind]
        chan_info = dict(loc=loc, eeg_loc=None, unit_mul=0, range=1., cal=1.,
                         coil_trans=None, kind=kind[0], coil_type=kind[1],
                         unit=kind[2], coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                         ch_name=name, scanno=ci + 1, logno=ci + 1)
        info['chs'].append(chan_info)
    info['dev_head_t'] = None
    info['dev_ctf_t'] = None
    info['ctf_head_t'] = None
    return info


class RawArray(_BaseRaw):
    """Raw object from numpy array

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
    """
    @verbose
    def __init__(self, data, info, verbose=None):
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        data = np.asanyarray(data, dtype=dtype)
        if data.ndim != 2:
            raise ValueError('data must be a 2D array')
        logger.info('Creating RawArray with %s data, n_channels=%s, n_times=%s'
                    % (dtype.__name__, data.shape[0], data.shape[1]))
        if len(data) != len(info['ch_names']):
            raise ValueError('len(data) does not match len(info["ch_names"])')
        assert len(info['ch_names']) == info['nchan']
        cals = np.zeros(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']

        self.verbose = verbose
        self.cals = cals
        self.rawdir = None
        self.proj = None
        self.comp = None
        self._filenames = list()
        self._preloaded = True
        self.info = info
        self._data = data
        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        self._times = np.arange(self.first_samp,
                                self.last_samp + 1) / info['sfreq']
        self._projectors = list()
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                    self.first_samp, self.last_samp,
                    float(self.first_samp) / info['sfreq'],
                    float(self.last_samp) / info['sfreq']))
        logger.info('Ready.')


class EpochsArray(Epochs):
    """Epochs object from numpy array

    Parameters
    ----------
    data : array, shape (n_epochs, n_channels, n_times)
        The channels' time series for each epoch.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
    events : array, of shape [n_events, 3]
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be marked as 'IGNORED' in the drop log.
    tmin : float
        Start time before event.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to acces associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.
    """

    @verbose
    def __init__(self, data, info, events, event_id=None, tmin=0,
                 verbose=None):
        self.info = info
        self._data = data
        if event_id is None:  # convert to int to make typing-checks happy
            event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))
        self.event_id = event_id
        self.events = events

        for key, val in self.event_id.items():
            if val not in events[:, 2]:
                msg = ('No matching events found for %s '
                       '(event id %i)' % (key, val))
                raise ValueError(msg)

        self.proj = None
        self.baseline = None
        self.preload = True
        self.reject = None
        self.decim = 0
        self.raw = None
        self.drop_log = [[] for _ in range(len(events))]
        self._bad_dropped = True

        self.selection = np.arange(len(events))
        self.picks = None
        self.times = (np.arange(data.shape[-1], dtype=np.float) /
                                info['sfreq'] + tmin)
        self.tmin = self.times[0]
        self.tmax = self.times[-1]
        self.verbose = verbose
        self.name = 'Unknown'
        self._projector = None
