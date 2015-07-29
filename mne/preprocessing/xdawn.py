"""Xdawn implementation."""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import copy as cp
import numpy as np
from scipy import linalg

from ..io.base import _BaseRaw
from ..epochs import _BaseEpochs
from .. import Covariance, EvokedArray, Evoked, EpochsArray
from ..io.pick import pick_types
from .ica import _get_fast_dot
from ..utils import logger
from ..decoding.mixin import TransformerMixin
from ..cov import _regularized_covariance
from ..channels.channels import ContainsMixin


def _least_square_evoked(data, events, event_id, tmin, tmax, sfreq):
    """Least square estimation of evoked response from data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        The data to estimates evoked
    events : ndarray, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be ignored.
    event_id : dict
        The id of the events to consider
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    sfreq : float
        Sampling frequency.

    Returns
    -------
    evokeds_data : dict of ndarray
        A dict of evoked data for each event type in event_id.
    toeplitz : dict of ndarray
        A dict of toeplitz matrix for each event type in event_id.
    """
    nmin = int(tmin * sfreq)
    nmax = int(tmax * sfreq)

    window = nmax - nmin
    n_samples = data.shape[1]
    toeplitz_mat = dict()
    full_toep = list()
    for eid in event_id:
        # select events by type
        ix_ev = events[:, -1] == event_id[eid]

        # build toeplitz matrix
        trig = np.zeros((n_samples, 1))
        ix_trig = (events[ix_ev, 0]) + nmin
        trig[ix_trig] = 1
        toep_mat = linalg.toeplitz(trig[0:window], trig)
        toeplitz_mat[eid] = toep_mat
        full_toep.append(toep_mat)

    # Concatenate toeplitz
    full_toep = np.concatenate(full_toep)

    # least square estimation
    predictor = np.dot(linalg.pinv(np.dot(full_toep, full_toep.T)), full_toep)
    all_evokeds = np.dot(predictor, data.T)
    all_evokeds = np.vsplit(all_evokeds, len(event_id))

    # parse evoked response
    evoked_data = dict()
    for idx, eid in enumerate(event_id):
        evoked_data[eid] = all_evokeds[idx].T

    return evoked_data, toeplitz_mat


def _check_overlapp(epochs):
    """check if events are overlapped."""
    isi = np.diff(epochs.events[:, 0])
    window = int((epochs.tmax - epochs.tmin) * epochs.info['sfreq'])
    # Events are overlapped if the minimal inter-stimulus interval is smaller
    # than the time window.
    return isi.min() < window


def _construct_signal_from_epochs(epochs):
    """Reconstruct pseudo continuous signal from epochs."""
    start = (np.min(epochs.events[:, 0])
             + int(epochs.tmin * epochs.info['sfreq']))
    stop = (np.max(epochs.events[:, 0])
            + int(epochs.tmax * epochs.info['sfreq']) + 1)

    n_samples = stop - start
    epochs_data = epochs.get_data()
    n_epochs, n_channels, n_times = epochs_data.shape
    events_pos = epochs.events[:, 0] - epochs.events[0, 0]

    data = np.zeros((n_channels, n_samples))
    for idx in range(n_epochs):
        onset = events_pos[idx]
        offset = onset + n_times
        data[:, onset:offset] = epochs_data[idx]

    return data


def least_square_evoked(epochs, return_toeplitz=False):
    """Least square estimation of evoked response from a Epochs instance.

    Parameters
    ----------
    epochs : Epochs instance
        An instance of Epochs.
    return_toeplitz : bool (default False)
        If true, compute the toeplitz matrix.

    Returns
    -------
    evokeds : dict of evoked instance
        An dict of evoked instance for each event type in epochs.event_id.
    toeplitz : dict of ndarray
        If return_toeplitz is true, return the toeplitz matrix for each event
        type in epochs.event_id.
    """
    if not isinstance(epochs, _BaseEpochs):
        raise ValueError('epochs must be an instance of `mne.Epochs`')

    events = epochs.events.copy()
    events[:, 0] -= events[0, 0] + int(epochs.tmin * epochs.info['sfreq'])
    data = _construct_signal_from_epochs(epochs)
    evoked_data, toeplitz = _least_square_evoked(data, events, epochs.event_id,
                                                 tmin=epochs.tmin,
                                                 tmax=epochs.tmax,
                                                 sfreq=epochs.info['sfreq'])
    evokeds = dict()
    info = cp.deepcopy(epochs.info)
    for name, data in evoked_data.items():
        n_events = len(events[events[:, 2] == epochs.event_id[name]])
        evoked = EvokedArray(data, info, tmin=epochs.tmin,
                             comment=name, nave=n_events)
        evokeds[name] = evoked

    if return_toeplitz:
        return evokeds, toeplitz

    return evokeds


class Xdawn(TransformerMixin, ContainsMixin):

    """Implementation of the Xdawn Algorithm.

    Xdawn is a spatial filtering method designed to improve the signal
    to signal + noise ratio (SSNR) of the ERP responses. Xdawn was originaly
    designed for P300 evoked potential by enhancing the target response with
    respect to the non-target response. This implementation is a generalization
    to any type of ERP.

    Parameters
    ----------
    n_components : int (default 2)
        The number of components to decompose M/EEG signals.
    signal_cov : None | Covariance | ndarray, shape (n_channels, n_channels)
        (default None). The signal covariance used for whitening of the data.
        if None, the covariance is estimated from the epochs signal.
    correct_overlap : 'auto' or bool (default 'auto')
        Apply correction for overlaped ERP for the estimation of evokeds
        responses. if 'auto', the overlapp correction is chosen in function
        of the events in epochs.events.
    reg : float | str | None (default None)
        if not None, allow regularization for covariance estimation
        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
        or Oracle Approximating Shrinkage ('oas').

    Attributes
    ----------
    filters_ : dict of ndarray
        If fit, the Xdawn components used to decompose the data for each event
        type, else empty.
    patterns_ : dict of ndarray
        If fit, the Xdawn patterns used to restore M/EEG signals for each event
        type, else empty.
    evokeds_ : dict of evoked instance
        If fit, the evoked response for each event type.

    Notes
    -----
    .. versionadded:: 0.10

    See Also
    --------
    ICA
    CSP

    References
    ----------
    [1] Rivet, B., Souloumiac, A., Attina, V., & Gibert, G. (2009). xDAWN
    algorithm to enhance evoked potentials: application to brain-computer
    interface. Biomedical Engineering, IEEE Transactions on, 56(8), 2035-2043.

    [2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., & Mattout, J. (2011,
    August). Theoretical analysis of xDAWN algorithm: application to an
    efficient sensor selection in a P300 BCI. In Signal Processing Conference,
    2011 19th European (pp. 1382-1386). IEEE.
    """

    def __init__(self, n_components=2, signal_cov=None, correct_overlap='auto',
                 reg=None):
        """init xdawn."""
        self.n_components = n_components
        self.signal_cov = signal_cov
        if reg == 'lws':
            raise DeprecationWarning('`lws` has been deprecated for the `reg`'
                                     ' argument. It will be removed in 0.11.'
                                     ' Use `ledoit_wolf` instead.')
            reg = 'ledoit_wolf'
        self.reg = reg
        self.filters_ = dict()
        self.patterns_ = dict()
        self.evokeds_ = dict()

        if correct_overlap not in ['auto', True, False]:
            raise ValueError('correct_overlap must be a bool or "auto"')
        self.correct_overlap = correct_overlap

    def fit(self, epochs, y=None):
        """Fit Xdawn from epochs.

        Parameters
        ----------
        epochs : Epochs object
            An instance of Epoch on which Xdawn filters will be trained.
        y : ndarray | None (default None)
            Not used, here for compatibility with decoding API.

        Returns
        -------
        self : Xdawn instance
            The Xdawn instance.
        """
        if self.correct_overlap == 'auto':
            self.correct_overlap = _check_overlapp(epochs)

        # Extract signal covariance
        if self.signal_cov is None:
            if self.correct_overlap:
                sig_data = _construct_signal_from_epochs(epochs)
            else:
                sig_data = np.hstack(epochs.get_data())
            self.signal_cov_ = _regularized_covariance(sig_data, self.reg)
        elif isinstance(self.signal_cov, Covariance):
            self.signal_cov_ = self.signal_cov.data
        elif isinstance(self.signal_cov, np.ndarray):
            self.signal_cov_ = self.signal_cov
        else:
            raise ValueError('signal_cov must be None, a covariance instance '
                             'or a ndarray')

        # estimates evoked covariance
        self.evokeds_cov_ = dict()
        if self.correct_overlap:
            if epochs.baseline is not None:
                raise ValueError('Baseline correction must be None if overlap '
                                 'correction activated')
            evokeds, toeplitz = least_square_evoked(epochs,
                                                    return_toeplitz=True)
        else:
            evokeds = dict()
            toeplitz = dict()
            for eid in epochs.event_id:
                evokeds[eid] = epochs[eid].average()
                toeplitz[eid] = 1.0
        self.evokeds_ = evokeds

        for eid in epochs.event_id:
            data = np.dot(evokeds[eid].data, toeplitz[eid])
            self.evokeds_cov_[eid] = _regularized_covariance(data, self.reg)

        # estimates spatial filters
        for eid in epochs.event_id:

            if self.signal_cov_.shape != self.evokeds_cov_[eid].shape:
                raise ValueError('Size of signal cov must be the same as the'
                                 ' number of channels in epochs')

            evals, evecs = linalg.eigh(self.evokeds_cov_[eid],
                                       self.signal_cov_)
            evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
            evecs /= np.sqrt(np.sum(evecs ** 2, axis=0))

            self.filters_[eid] = evecs
            self.patterns_[eid] = linalg.inv(evecs.T)

        # store some values
        self.ch_names = epochs.ch_names
        self.exclude = list(range(self.n_components, len(self.ch_names)))
        self.event_id = epochs.event_id
        return self

    def transform(self, epochs):
        """Apply Xdawn dim reduction.

        Parameters
        ----------
        epochs : Epochs | ndarray, shape (n_epochs, n_channels, n_times)
            Data on which Xdawn filters will be applied.

        Returns
        -------
        X : ndarray, shape (n_epochs, n_components * event_types, n_times)
            Spatially filtered signals.
        """
        if isinstance(epochs, _BaseEpochs):
            data = epochs.get_data()
        elif isinstance(epochs, np.ndarray):
            data = epochs
        else:
            raise ValueError('Data input must be of Epoch '
                             'type or numpy array')

        # create full matrix of spatial filter
        full_filters = list()
        for filt in self.filters_.values():
            full_filters.append(filt[:, 0:self.n_components])
        full_filters = np.concatenate(full_filters, axis=1)

        # Apply spatial filters
        X = np.dot(full_filters.T, data)
        X = X.transpose((1, 0, 2))
        return X

    def apply(self, inst, event_id=None, include=None, exclude=None):
        """Remove selected components from the signal.

        Given the unmixing matrix, transform data,
        zero out components, and inverse transform the data.
        This procedure will reconstruct M/EEG signals from which
        the dynamics described by the excluded components is subtracted.

        Parameters
        ----------
        inst : instance of Raw | Epochs | Evoked
            The data to be processed.
        event_id : dict | list of str | None (default None)
            The kind of event to apply. if None, a dict of inst will be return
            one for each type of event xdawn has been fitted.
        include : array_like of int | None (default None)
            The indices refering to columns in the ummixing matrix. The
            components to be kept. If None, the first n_components (as defined
            in the Xdawn constructor) will be kept.
        exclude : array_like of int | None (default None)
            The indices refering to columns in the ummixing matrix. The
            components to be zeroed out. If None, all the components except the
            first n_components will be exclude.

        Returns
        -------
        out : dict of instance
            A dict of instance (from the same type as inst input) for each
            event type in event_id.
        """
        if event_id is None:
            event_id = self.event_id

        if isinstance(inst, _BaseRaw):
            out = self._apply_raw(raw=inst, include=include, exclude=exclude,
                                  event_id=event_id)
        elif isinstance(inst, _BaseEpochs):
            out = self._apply_epochs(epochs=inst, include=include,
                                     exclude=exclude, event_id=event_id)
        elif isinstance(inst, Evoked):
            out = self._apply_evoked(evoked=inst, include=include,
                                     exclude=exclude, event_id=event_id)
        else:
            raise ValueError('Data input must be Raw, Epochs or Evoked type')
        return out

    def _apply_raw(self, raw, include, exclude, event_id):
        """Aux method."""
        if not raw.preload:
            raise ValueError('Raw data must be preloaded to apply Xdawn')

        picks = pick_types(raw.info, meg=False, include=self.ch_names,
                           exclude='bads')
        raws = dict()
        for eid in event_id:
            data = raw[picks, :][0]

            data = self._pick_sources(data, include, exclude, eid)

            raw_r = raw.copy()

            raw_r[picks, :] = data
            raws[eid] = raw_r
        return raws

    def _apply_epochs(self, epochs, include, exclude, event_id):
        """Aux method."""
        if not epochs.preload:
            raise ValueError('Epochs must be preloaded to apply Xdawn')

        picks = pick_types(epochs.info, meg=False, ref_meg=False,
                           include=self.ch_names, exclude='bads')

        # special case where epochs come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Epochs don\'t match fitted data: %i channels '
                               'fitted but %i channels supplied. \nPlease '
                               'provide Epochs compatible with '
                               'xdawn.ch_names' % (len(self.ch_names),
                                                   len(picks)))

        epochs_dict = dict()
        data = np.hstack(epochs.get_data()[:, picks])

        for eid in event_id:

            data_r = self._pick_sources(data, include, exclude, eid)
            data_r = np.array(np.split(data_r, len(epochs.events), 1))
            info_r = cp.deepcopy(epochs.info)
            epochs_r = EpochsArray(data=data_r, info=info_r,
                                   events=epochs.events, tmin=epochs.tmin,
                                   event_id=epochs.event_id, verbose=False)
            epochs_r.preload = True
            epochs_dict[eid] = epochs_r

        return epochs_dict

    def _apply_evoked(self, evoked, include, exclude, event_id):
        """Aux method."""
        picks = pick_types(evoked.info, meg=False, ref_meg=False,
                           include=self.ch_names,
                           exclude='bads')

        # special case where evoked come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Evoked does not match fitted data: %i channels'
                               ' fitted but %i channels supplied. \nPlease '
                               'provide an Evoked object that\'s compatible '
                               'with xdawn.ch_names' % (len(self.ch_names),
                                                        len(picks)))

        data = evoked.data[picks]
        evokeds = dict()

        for eid in event_id:

            data_r = self._pick_sources(data, include, exclude, eid)
            evokeds[eid] = evoked.copy()

            # restore evoked
            evokeds[eid].data[picks] = data_r

        return evokeds

    def _pick_sources(self, data, include, exclude, eid):
        """Aux method."""
        fast_dot = _get_fast_dot()
        if exclude is None:
            exclude = self.exclude
        else:
            exclude = list(set(list(self.exclude) + list(exclude)))

        logger.info('Transforming to Xdawn space')

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
