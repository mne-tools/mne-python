# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#          Asish Panda <asishrocks95@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg
from ..cov import Covariance, _regularized_covariance
from ..decoding import TransformerMixin, BaseEstimator


def _construct_signal_from_epochs(epochs, events, sfreq, tmin):
    """Reconstruct pseudo continuous signal from epochs."""
    n_epochs, n_channels, n_times = epochs.shape
    tmax = tmin + (n_times) / float(sfreq)
    start = (np.min(events[:, 0]) + int(tmin * sfreq))
    stop = (np.max(events[:, 0]) + int(tmax * sfreq) + 1)

    n_samples = stop - start
    n_epochs, n_channels, n_times = epochs.shape
    events_pos = events[:, 0] - events[0, 0]

    raw = np.zeros((n_channels, n_samples))
    for idx in range(n_epochs):
        onset = events_pos[idx]
        offset = onset + n_times
        raw[:, onset:offset] = epochs[idx]

    return raw


def _least_square_evoked(epochs, events, tmin, tmax, sfreq):
    """Least square estimation of evoked response from epochs.

    Parameters
    ----------
    epochs : array, shape (n_channels, n_times)
        The epochs data to estimate evoked.
    events : array, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be ignored.
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    sfreq : float
        Sampling frequency.

    Returns
    -------
    evokeds_data : dict of array
        A dict of evoked data for each event type in event_id.
    toeplitz : dict of array
        A dict of toeplitz matrix for each event type in event_id.
    """

    n_epochs, n_channels, n_times = epochs.shape

    # Deal with shuffled epochs
    events = events.copy()
    events[:, 0] -= events[0, 0] + int(tmin * sfreq)

    # Contruct raw signal
    raw = _construct_signal_from_epochs(epochs, events, sfreq, tmin, tmax)

    # Compute average evoked
    n_min, n_max = int(tmin * sfreq), int(tmax * sfreq)
    window = n_max - n_min
    n_samples = raw.shape[1]
    toeplitz_mat = np.zeros((len(window), n_times))
    full_toep = list()
    classes = np.unique(events[:, 2])
    for ii, this_class in enumerate(classes):
        # select events by type
        sel = events[:, 2] == ii

        # build toeplitz matrix
        trig = np.zeros((n_samples, 1))
        ix_trig = (events[sel, 0]) + n_min
        trig[ix_trig] = 1
        toep_mat = linalg.toeplitz(trig[0:window], trig)
        toeplitz_mat[ii] = toep_mat
        full_toep.append(toep_mat)

    # Concatenate toeplitz
    full_toep = np.concatenate(full_toep)

    # least square estimation
    predictor = np.dot(linalg.pinv(np.dot(full_toep, full_toep.T)), full_toep)
    all_evokeds = np.dot(predictor, raw.T)
    all_evokeds = np.vsplit(all_evokeds, len(classes)).transpose(0, 2, 1)

    return all_evokeds, toeplitz_mat


def _fit_xdawn(epochs, y, n_components, signal_cov, reg,
               tmin=0, sfreq=1., events=None):
    n_epochs, n_channels, n_times = epochs.shape

    classes = np.unique(y)

    # Retrieve or compute whitening covariance
    if signal_cov is None:
        signal_cov = _regularized_covariance(np.hstack(epochs), reg)
    elif isinstance(signal_cov, Covariance):
        signal_cov = signal_cov.data
    if not isinstance(signal_cov, np.ndarray) or (
            not np.array_equal(signal_cov.shape, np.tile(epochs.shape[1], 2))):
        raise ValueError('signal_cov must be None, a covariance instance '
                         'or a array of shape (n_chans, n_chans)')

    # Get prototype events
    if events is not None:
        evokeds, toeplitz = _least_square_evoked(epochs, events, tmin, sfreq)
    else:
        evokeds, toeplitzs = list(), list()
        for c in classes:
            # Prototyped responce for each class
            evokeds.append(np.mean(epochs[y == c, :, :], axis=0))
            toeplitzs.append(1.)

    filters = []
    patterns = []
    for evo, toeplitz in zip(evokeds, toeplitzs):
        # Covariance matrix of the prototyper response & signal
        evo = np.dot(evo, toeplitz)
        evo_cov = np.matrix(_regularized_covariance(evo, reg))

        # Spatial filters
        evals, evecs = linalg.eigh(evo_cov, signal_cov)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)
        _patterns = np.linalg.pinv(evecs.T)
        filters.append(evecs[:, :n_components].T)
        patterns.append(_patterns[:, :n_components].T)

    filters = np.concatenate(filters, axis=0)
    patterns = np.concatenate(patterns, axis=0)
    evokeds = np.array(evokeds)
    return filters, patterns, evokeds


class XdawnTransformer(BaseEstimator, TransformerMixin):

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
    reg : float | str | None (default None)
        If not None, allow regularization for covariance estimation
        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
        or Oracle Approximating Shrinkage ('oas').
    signal_cov : None | Covariance | array, shape (n_channels, n_channels)
        The signal covariance used for whitening of the data.
        if None, the covariance is estimated from the epochs signal.

    Attributes
    ----------
    filters_ : array, shape (n_channels, n_channels)
        The Xdawn components used to decompose the data for each event type.
    patterns_ : array, shape (n_channels, n_channels)
        The Xdawn patterns used to restore M/EEG signals for each event type.

    References
    ----------
    [1] Rivet, B., Souloumiac, A., Attina, V., & Gibert, G. (2009). xDAWN
    algorithm to enhance evoked potentials: application to brain-computer
    interface. Biomedical Engineering, IEEE Transactions on, 56(8), 2035-2043.
    [2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., & Mattout, J. (2011,
    August). Theoretical analysis of xDAWN algorithm: application to an
    efficient sensor selection in a P300 BCI. In Signal Processing Conference,
    2011 19th European (pp. 1382-1386). IEEE.


    See Also
    --------
    CSP
    """

    def __init__(self, n_components=2, signal_cov=None, reg=None,
                 tmin=0, sfreq=1.):
        """Init."""
        self.n_components = n_components
        self.signal_cov = signal_cov
        self.reg = reg
        self.tmin = tmin
        self.sfreq = sfreq

    def fit(self, X, y, events=None):
        """Train xdawn spatial filters.

        Parameters
        ----------
        X : array, shape (n_trials, n_channels, n_samples)
            array of trials.
        y : array shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : Xdawn instance
            The Xdawn instance.
        """
        self.filters_, self.patterns_, _ = _fit_xdawn(
            X, y,  n_components=self.n_components, signal_cov=self.signal_cov,
            reg=self.reg, tmin=self.tmin, sfreq=self.sfreq, events=events)
        return self

    def transform(self, X):
        """Apply spatial filters.

        Parameters
        ----------
        X : array, shape (n_trials, n_channels, n_samples)
            array of trials.

        Returns
        -------
        Xf : array, shape (n_trials, n_components * n_classes, n_samples)
            array of spatialy filtered trials.
        """
        X = np.dot(self.filters_, X)
        X = X.transpose((1, 0, 2))
        return X


# to be deprecated
import copy as cp
from ..epochs import _BaseEpochs, EpochsArray
from ..io import _BaseRaw
from ..io.pick import _pick_data_channels
from .. import EvokedArray, Evoked
from ..utils import logger, deprecated
from .ica import _get_fast_dot


@deprecated("Xdawn will be removed in mne 0.14; use XdawnTransformer instead.")
class Xdawn(XdawnTransformer):
    def __init__(self, n_components=2, signal_cov=None, correct_overlap='auto',
                 reg=None):

        if correct_overlap not in ['auto', True, False]:
            raise ValueError('correct_overlap must be a bool or "auto"')
        self.correct_overlap = correct_overlap
        super(Xdawn, self).__init__(n_components, signal_cov, reg)

    def _get_data(self, inst, picks=None):
        if isinstance(inst, np.ndarray):
            X = inst
            picks = range(inst.shape[1]) if picks is None else picks
        else:
            picks = _pick_data_channels(inst.info) if picks is None else picks
            if hasattr(inst, 'get_data'):
                X = inst.get_data()
            elif hasattr(inst, '_data'):
                X = inst._data
            elif hasattr(inst, 'data'):
                X = inst.data
            else:
                raise ValueError('inst must be array, Epochs, Raw, or Evoked')
        return X[:, picks], picks

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
        # Check whether epochs overlap in time
        correct_overlap = False
        if self.correct_overlap == 'auto':
            isi = np.diff(epochs.events[:, 0])
            window = int((epochs.tmax - epochs.tmin) * epochs.info['sfreq'])
            # Events are overlapped if the minimal inter-stimulus interval is
            #  smaller than the time window.
            correct_overlap = isi.min() < window

        # XXX FIXME shouldn't test on self.correct_overlap but direclty co
        if (epochs.baseline is not None) and self.correct_overlap:
            raise ValueError('Baseline correction must be None if overlap '
                             'correction activated')

        X, self._picks = self._get_data(epochs)
        y = epochs.events[:, 2]

        events, tmin, sfreq = None, None, None
        if correct_overlap:
            events = epochs.events
            tmin = epochs.tmin
            sfreq = epochs.info['sfreq']

        # XXX In this old version of Xdawn we keep all components
        n_components = len(self._picks)
        filters, patterns, evokeds = _fit_xdawn(
            X, y,  n_components=n_components, signal_cov=self.signal_cov,
            reg=self.reg, tmin=tmin, sfreq=sfreq, events=events)

        filters = filters.reshape(-1, n_components, filters.shape[-1])
        patterns = patterns.reshape(-1, n_components, patterns.shape[-1])
        self.filters_, self.patterns_, self.evokeds_ = dict(), dict(), dict()

        # sort event_id to be in order
        idx = np.argsort([value for _, value in epochs.event_id.iteritems()])
        for eid, this_filter, this_pattern, this_evo in zip(
                epochs.event_id, filters[idx], patterns[idx], evokeds[idx]):
            self.filters_[eid] = this_filter.T
            self.patterns_[eid] = this_pattern.T
            n_events = len(epochs[eid])
            evoked = EvokedArray(this_evo, epochs.info, tmin=epochs.tmin,
                                 comment=eid, nave=n_events)
            self.evokeds_[eid] = evoked

        # Store some values
        self.ch_names = epochs.ch_names
        self.exclude = list(range(self.n_components, len(self.ch_names)))
        self.event_id = epochs.event_id

        # update overlap XXX FIXME params shouldn't be changed!
        self.correct_overlap = correct_overlap
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
        X, self._picks = self._get_data(epochs, self._picks)
        filters = [filt[:self.n_components]
                   for _, filt in self.filters_.iteritems()]
        filters = np.concatenate(filters, axis=0)
        X = np.dot(filters, X)
        return X.transpose((1, 0, 2))

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
            The indices referring to columns in the ummixing matrix. The
            components to be kept. If None, the first n_components (as defined
            in the Xdawn constructor) will be kept.
        exclude : array_like of int | None (default None)
            The indices referring to columns in the ummixing matrix. The
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

        raws = dict()
        for eid in event_id:
            data = raw[self._picks, :][0]

            data = self._pick_sources(data, include, exclude, eid)

            raw_r = raw.copy()

            raw_r[self._picks, :] = data
            raws[eid] = raw_r
        return raws

    def _apply_epochs(self, epochs, include, exclude, event_id):
        """Aux method."""
        if not epochs.preload:
            raise ValueError('Epochs must be preloaded to apply Xdawn')

        # special case where epochs come picked but fit was 'unpicked'.
        epochs_dict = dict()
        data = np.hstack(epochs.get_data()[:, self._picks])

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
        data = evoked.data[self._picks]
        evokeds = dict()

        for eid in event_id:

            data_r = self._pick_sources(data, include, exclude, eid)
            evokeds[eid] = evoked.copy()

            # restore evoked
            evokeds[eid].data[self._picks] = data_r

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
