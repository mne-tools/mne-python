# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
from scipy import linalg
import copy as cp

from ..io.proj import Projection
from ..io.base import _BaseRaw
from ..epochs import _BaseEpochs
from .. import Covariance, EvokedArray, compute_raw_data_covariance
from ..io.pick import pick_types, pick_info
from .ica import _get_fast_dot
from ..utils import logger

def _least_square_evoked(data, events, event_id, tmin, tmax, sfreq, decim):
    """Least square estimation of evoked response from data
    return evoked data and toeplitz matrices.
    """
    nmin = int(tmin * sfreq / decim)
    nmax = int(tmax * sfreq / decim)
    data = data[:, ::decim]

    window = nmax - nmin
    ne, ns = data.shape
    to = {}

    for eid in event_id:
        # select events by type
        ix_ev = events[:, -1] == event_id[eid]

        # build toeplitz matrix
        trig = np.zeros((ns, 1))
        ix_trig = (events[ix_ev, 0] / decim) + nmin
        trig[ix_trig] = 1
        toep = linalg.toeplitz(trig[0:window], trig)
        to[eid] = toep

    # Concatenate toeplitz
    to_tot = np.concatenate(to.values())

    # least square estimation
    evo = np.dot(np.dot(linalg.pinv(np.dot(to_tot, to_tot.T)), to_tot), data.T)

    # parse evoked response
    evoked_data = {}
    for i, eid in enumerate(event_id):
        evoked_data[eid] = np.array(evo[(i * window):(i + 1) * window, :]).T

    return evoked_data, to


def least_square_evoked(raw, events, event_id, tmin=0.0, tmax=1.0, decim=1,
                        picks=None, return_toeplitz=False):
    """ Least square estimation of evoked response from a raw instance.

    Parameters
    ----------
    raw : Raw object
        An instance of Raw.
    events : array, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be ignored.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to acces associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    picks : array-like of int | None (default)
        Indices of channels to include (if None, all channels are used).
    decim : int
        Factor by which to downsample the data from the raw file upon import.
        Warning: This simply selects every nth sample, data is not filtered
        here. If data is not properly filtered, aliasing artifacts may occur.
    return_toeplitz : bool
        if true, compute the toeplitz matrix

    Returns
    -------
    evokeds : array of evoked instance
        An array of evoked instance for each event type in event_id.
    """
    if not isinstance(raw, _BaseRaw):
        raise ValueError('The raw must be an instance of `mne.io.Raw`')

    if event_id is None:  # convert to int to make typing-checks happy
        event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))
    elif isinstance(event_id, int):
        event_id = dict(str(event_id), int(event_id))
    elif isinstance(event_id, list):
        event_id = dict((str(e), int(e)) for e in event_id)
    elif not isinstance(event_id, dict):
        raise ValueError('event_id must be None, int, list of int, or dict')

    for key, val in event_id.items():
        if val not in events[:, 2]:
            raise ValueError('No matching events found for %s '
                             '(event id %i)' % (key, val))
    if picks is None:
        picks = pick_types(raw.info, meg=True, eeg=True)

    evs = events.copy()
    evs[:, 0] -= raw.first_samp
    evo, to = _least_square_evoked(raw._data[picks], evs, event_id,
                                   tmin=tmin, tmax=tmax,
                                   sfreq=raw.info['sfreq'], decim=decim)
    evokeds = []
    info = pick_info(raw.info, picks)
    info['sfreq'] /= decim

    for name, data in evo.items():
        n_events = len(events[events[:, 2] == event_id[name]])
        evoked = EvokedArray(data, info, tmin=tmin,
                             comment=name, nave=n_events)
        evokeds.append(evoked)

    if return_toeplitz:
        return evokeds, to

    return evokeds

class Xdawn():
    """
    Parameters
    ----------
    n_components : int, default 2
        The number of components to decompose M/EEG signals.
    reg : float, str, None
        if not None, allow regularization for covariance estimation
        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('lws') or
        Oracle Approximating Shrinkage ('oas').

    Attributes
    ----------
    filters_ : dict of ndarray
        If fit, the Xdawn components used to decompose the data for each event
        type, else empty.
    patterns_ : dict of ndarray
        If fit, the Xdawn patterns used to restore M/EEG signals for each event
        type, else empty.
    evokeds_ : dict of evoked instance
        If fit, the evoked response for each event type
    """
    def __init__(self,n_components=2, reg=None):
        self.n_components = n_components
        self.reg = reg
        self.filters_ = {}
        self.patterns_ = {}
        self.evokeds_ = {}
        self.projs_ = {}

    def fit(self, raw, events, event_id, tmin=0.0, tmax=1.0, decim=1,
            picks=None):
        """
        Fit xdawn
        """
        if picks is None:
            picks = pick_types(raw.info, meg=True, eeg=True)

        self.picks = picks
        evokeds, to = least_square_evoked(raw=raw, events=events,
                                          event_id=event_id, tmin=tmin,
                                          tmax=tmax, decim=decim,
                                          picks=self.picks,
                                          return_toeplitz=True)
        self.ch_names = [raw.ch_names[i] for i in self.picks]
        self.exclude = range(self.n_components, len(self.picks))
        self.event_id = event_id
        # Compute noise cov
        # FIXME : Use mne method
        noise_cov = np.cov(raw._data[self.picks, ::decim])

        for i, eid in enumerate(event_id):
            self.evokeds_[eid] = evokeds[i]

            if self.reg is None:
                evoked_cov = np.cov(np.dot(evokeds[i].data, to[eid]))
            else:
                # FIXME : use mne reg
                raise NotImplementedError('Regularization not implemented')

            evals, evecs = linalg.eigh(evoked_cov, noise_cov)
            evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
            evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)

            self.filters_[eid] = evecs
            self.patterns_[eid] = linalg.inv(evecs.T)

            # Create and add projectors
            ch_names = evokeds[i].ch_names
            projs = []
            for j in range(self.n_components):
                proj_data = dict(col_names=ch_names, row_names=None,
                                 data=evecs[:, j].T, nrow=1,
                                 ncol=len(ch_names))
                projs.append(Projection(active=True, data=proj_data,
                             desc="%s Xdawn #%d" % (eid, j)))

            # self.evokeds_[eid].add_proj(projs)
            self.projs_[eid] = projs

        return self

    def apply(self, inst, event_id=None, include=None, exclude=None):
        """Remove selected components from the signal.

        Given the unmixing matrix, transform data,
        zero out components, and inverse transform the data.
        This procedure will reconstruct M/EEG signals from which
        the dynamics described by the excluded components is subtracted.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            The data to be processed.
        include : array_like of int.
            The indices refering to columns in the ummixing matrix. The
            components to be kept.
        exclude : array_like of int.
            The indices refering to columns in the ummixing matrix. The
            components to be zeroed out.
        event_id : dict or None
            The kind of event to appy. if none, an array of inst will be return
            one for each type of event xdawn has been fitted.
        copy : bool
            Whether to return a copy or whether to apply the solution in place.
            Defaults to False.
        """
        if event_id is None:
            event_id = self.event_id

        if isinstance(inst, _BaseRaw):
            out = self._apply_raw(raw=inst, include=include,
                                  exclude=exclude, event_id=event_id)
        elif isinstance(inst, _BaseEpochs):
            out = self._apply_epochs(epochs=inst, include=include,
                                     exclude=exclude, event_id=event_id)
        elif isinstance(inst, Evoked):
            out = self._apply_evoked(evoked=inst, include=include,
                                     exclude=exclude, event_id=event_id)
        else:
            raise ValueError('Data input must be of Raw, Epochs or Evoked '
                             'type')
        return out

    def _apply_raw(self, raw, include, exclude, event_id):
        """Aux method"""
        if not raw.preload:
            raise ValueError('Raw data must be preloaded to apply Xdawn')

        if exclude is None:
            exclude = list(set(self.exclude))
        else:
            exclude = list(set(self.exclude + exclude))

        picks = pick_types(raw.info, meg=False, include=self.ch_names,
                           exclude='bads')
        raws = {}
        for eid in event_id:
            data = raw[picks, :][0]

            data = self._pick_sources(data, include, exclude, eid)

            raw_r = raw.copy()

            raw_r[picks, :] = data
            raws[eid] = raw_r
        return raws

    def _apply_epochs(self, epochs, include, exclude, event_id):

        if not epochs.preload:
            raise ValueError('Epochs must be preloaded to apply ICA')

        picks = pick_types(epochs.info, meg=False, ref_meg=False,
                           include=self.ch_names,
                           exclude='bads')

        # special case where epochs come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Epochs don\'t match fitted data: %i channels '
                               'fitted but %i channels supplied. \nPlease '
                               'provide Epochs compatible with '
                               'ica.ch_names' % (len(self.ch_names),
                                                 len(picks)))

        epochs_dict = {}
        data = np.hstack(epochs.get_data()[:, picks])

        for eid in event_id:

            data_r = self._pick_sources(data, include, exclude, eid)

            epochs_r = epochs.copy()
            epochs_r._data[:, picks] = np.array(np.split(data_r,
                                                len(epochs.events), 1))
            epochs_r.preload = True
            epochs_dict[eid] = epochs_r

        return epochs_dict

    def _apply_evoked(self, evoked, include, exclude, event_id):

        picks = pick_types(evoked.info, meg=False, ref_meg=False,
                           include=self.ch_names,
                           exclude='bads')

        # special case where evoked come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Evoked does not match fitted data: %i channels'
                               ' fitted but %i channels supplied. \nPlease '
                               'provide an Evoked object that\'s compatible '
                               'with ica.ch_names' % (len(self.ch_names),
                                                      len(picks)))

        data = evoked.data[picks]
        evokeds = {}

        for eid in event_id:

            data_r = self._pick_sources(data, include, exclude, eid)
            evo = evoked.copy()

            # restore evoked
            evo.data[picks] = data_r
            evokeds[eid] = evo

        return evokeds

    def _pick_sources(self, data, include, exclude, eid):
        """Aux function"""
        fast_dot = _get_fast_dot()
        if exclude is None:
            exclude = self.exclude
        else:
            exclude = list(set(self.exclude + list(exclude)))


        n_components = self.n_components
        logger.info('Transforming to Xdawn space (%i components)' %
                    n_components)


        # Apply unmixing
        sources = fast_dot(self.filters_[eid].T, data)

        if include not in (None, []):
            mask = np.ones(len(sources), dtype=np.bool)
            mask[np.unique(include)] = False
            sources[mask] = 0.
            logger.info('Zeroing out %i Xdawn components' % mask.sum())
        elif exclude not in (None, []):
            exclude_ = np.unique(exclude)
            sources[exclude_] = 0.
            logger.info('Zeroing out %i Xdawn components' % len(exclude_))
        logger.info('Inverse transforming to sensor space')
        data = fast_dot(self.patterns_[eid], sources)


        return data
