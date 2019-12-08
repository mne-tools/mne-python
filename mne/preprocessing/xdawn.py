# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#          Asish Panda <asishrocks95@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from .. import EvokedArray, Evoked
from ..cov import Covariance, _regularized_covariance
from ..decoding import TransformerMixin, BaseEstimator
from ..epochs import BaseEpochs
from ..io import BaseRaw
from ..io.pick import _pick_data_channels, pick_info
from ..utils import logger, _check_option


def _construct_signal_from_epochs(epochs, events, sfreq, tmin):
    """Reconstruct pseudo continuous signal from epochs."""
    n_epochs, n_channels, n_times = epochs.shape
    tmax = tmin + n_times / float(sfreq)
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


def _least_square_evoked(epochs_data, events, tmin, sfreq):
    """Least square estimation of evoked response from epochs data.

    Parameters
    ----------
    epochs_data : array, shape (n_channels, n_times)
        The epochs data to estimate evoked.
    events : array, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be ignored.
    tmin : float
        Start time before event.
    sfreq : float
        Sampling frequency.

    Returns
    -------
    evokeds : array, shape (n_class, n_components, n_times)
        An concatenated array of evoked data for each event type.
    toeplitz : array, shape (n_class * n_components, n_channels)
        An concatenated array of toeplitz matrix for each event type.
    """
    n_epochs, n_channels, n_times = epochs_data.shape
    tmax = tmin + n_times / float(sfreq)

    # Deal with shuffled epochs
    events = events.copy()
    events[:, 0] -= events[0, 0] + int(tmin * sfreq)

    # Construct raw signal
    raw = _construct_signal_from_epochs(epochs_data, events, sfreq, tmin)

    # Compute the independent evoked responses per condition, while correcting
    # for event overlaps.
    n_min, n_max = int(tmin * sfreq), int(tmax * sfreq)
    window = n_max - n_min
    n_samples = raw.shape[1]
    toeplitz = list()
    classes = np.unique(events[:, 2])
    for ii, this_class in enumerate(classes):
        # select events by type
        sel = events[:, 2] == this_class

        # build toeplitz matrix
        trig = np.zeros((n_samples, 1))
        ix_trig = (events[sel, 0]) + n_min
        trig[ix_trig] = 1
        toeplitz.append(linalg.toeplitz(trig[0:window], trig))

    # Concatenate toeplitz
    toeplitz = np.array(toeplitz)
    X = np.concatenate(toeplitz)

    # least square estimation
    predictor = np.dot(linalg.pinv(np.dot(X, X.T)), X)
    evokeds = np.dot(predictor, raw.T)
    evokeds = np.transpose(np.vsplit(evokeds, len(classes)), (0, 2, 1))
    return evokeds, toeplitz


def _fit_xdawn(epochs_data, y, n_components, reg=None, signal_cov=None,
               events=None, tmin=0., sfreq=1., method_params=None, info=None):
    """Fit filters and coefs using Xdawn Algorithm.

    Xdawn is a spatial filtering method designed to improve the signal
    to signal + noise ratio (SSNR) of the event related responses. Xdawn was
    originally designed for P300 evoked potential by enhancing the target
    response with respect to the non-target response. This implementation is a
    generalization to any type of event related response.

    Parameters
    ----------
    epochs_data : array, shape (n_epochs, n_channels, n_times)
        The epochs data.
    y : array, shape (n_epochs)
        The epochs class.
    n_components : int (default 2)
        The number of components to decompose the signals signals.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow
        regularization for covariance estimation.
        If float, shrinkage is used (0 <= shrinkage <= 1).
        For str options, ``reg`` will be passed as ``method`` to
        :func:`mne.compute_covariance`.
    signal_cov : None | Covariance | array, shape (n_channels, n_channels)
        The signal covariance used for whitening of the data.
        if None, the covariance is estimated from the epochs signal.
    events : array, shape (n_epochs, 3)
        The epochs events, used to correct for epochs overlap.
    tmin : float
        Epochs starting time. Only used if events is passed to correct for
        epochs overlap.
    sfreq : float
        Sampling frequency.  Only used if events is passed to correct for
        epochs overlap.

    Returns
    -------
    filters : array, shape (n_channels, n_channels)
        The Xdawn components used to decompose the data for each event type.
        Each row corresponds to one component.
    patterns : array, shape (n_channels, n_channels)
        The Xdawn patterns used to restore the signals for each event type.
    evokeds : array, shape (n_class, n_components, n_times)
        The independent evoked responses per condition.
    """
    if not isinstance(epochs_data, np.ndarray) or epochs_data.ndim != 3:
        raise ValueError('epochs_data must be 3D ndarray')

    classes = np.unique(y)

    # XXX Eventually this could be made to deal with rank deficiency properly
    # by exposing this "rank" parameter, but this will require refactoring
    # the linalg.eigh call to operate in the lower-dimension
    # subspace, then project back out.

    # Retrieve or compute whitening covariance
    if signal_cov is None:
        signal_cov = _regularized_covariance(
            np.hstack(epochs_data), reg, method_params, info, rank='full')
    elif isinstance(signal_cov, Covariance):
        signal_cov = signal_cov.data
    if not isinstance(signal_cov, np.ndarray) or (
            not np.array_equal(signal_cov.shape,
                               np.tile(epochs_data.shape[1], 2))):
        raise ValueError('signal_cov must be None, a covariance instance, '
                         'or an array of shape (n_chans, n_chans)')

    # Get prototype events
    if events is not None:
        evokeds, toeplitzs = _least_square_evoked(
            epochs_data, events, tmin, sfreq)
    else:
        evokeds, toeplitzs = list(), list()
        for c in classes:
            # Prototyped response for each class
            evokeds.append(np.mean(epochs_data[y == c, :, :], axis=0))
            toeplitzs.append(1.)

    filters = list()
    patterns = list()
    for evo, toeplitz in zip(evokeds, toeplitzs):
        # Estimate covariance matrix of the prototype response
        evo = np.dot(evo, toeplitz)
        evo_cov = _regularized_covariance(evo, reg, method_params, info,
                                          rank='full')

        # Fit spatial filters
        try:
            evals, evecs = linalg.eigh(evo_cov, signal_cov)
        except np.linalg.LinAlgError as exp:
            raise ValueError('Could not compute eigenvalues, ensure '
                             'proper regularization (%s)' % (exp,))
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)
        _patterns = np.linalg.pinv(evecs.T)
        filters.append(evecs[:, :n_components].T)
        patterns.append(_patterns[:, :n_components].T)

    filters = np.concatenate(filters, axis=0)
    patterns = np.concatenate(patterns, axis=0)
    evokeds = np.array(evokeds)
    return filters, patterns, evokeds


class _XdawnTransformer(BaseEstimator, TransformerMixin):
    """Implementation of the Xdawn Algorithm compatible with scikit-learn.

    Xdawn is a spatial filtering method designed to improve the signal
    to signal + noise ratio (SSNR) of the event related responses. Xdawn was
    originally designed for P300 evoked potential by enhancing the target
    response with respect to the non-target response. This implementation is a
    generalization to any type of event related response.

    .. note:: _XdawnTransformer does not correct for epochs overlap. To correct
              overlaps see ``Xdawn``.

    Parameters
    ----------
    n_components : int (default 2)
        The number of components to decompose the signals.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow
        regularization for covariance estimation.
        If float, shrinkage is used (0 <= shrinkage <= 1).
        For str options, ``reg`` will be passed to ``method`` to
        :func:`mne.compute_covariance`.
    signal_cov : None | Covariance | array, shape (n_channels, n_channels)
        The signal covariance used for whitening of the data.
        if None, the covariance is estimated from the epochs signal.
    method_params : dict | None
        Parameters to pass to :func:`mne.compute_covariance`.

        .. versionadded:: 0.16

    Attributes
    ----------
    classes_ : array, shape (n_classes)
        The event indices of the classes.
    filters_ : array, shape (n_channels, n_channels)
        The Xdawn components used to decompose the data for each event type.
    patterns_ : array, shape (n_channels, n_channels)
        The Xdawn patterns used to restore the signals for each event type.
    """

    def __init__(self, n_components=2, reg=None, signal_cov=None,
                 method_params=None):
        """Init."""
        self.n_components = n_components
        self.signal_cov = signal_cov
        self.reg = reg
        self.method_params = method_params

    def fit(self, X, y=None):
        """Fit Xdawn spatial filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_samples)
            The target data.
        y : array, shape (n_epochs,) | None
            The target labels. If None, Xdawn fit on the average evoked.

        Returns
        -------
        self : Xdawn instance
            The Xdawn instance.
        """
        X, y = self._check_Xy(X, y)

        # Main function
        self.classes_ = np.unique(y)
        self.filters_, self.patterns_, _ = _fit_xdawn(
            X, y, n_components=self.n_components, reg=self.reg,
            signal_cov=self.signal_cov, method_params=self.method_params)
        return self

    def transform(self, X):
        """Transform data with spatial filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_samples)
            The target data.

        Returns
        -------
        X : array, shape (n_epochs, n_components * n_classes, n_samples)
            The transformed data.
        """
        X, _ = self._check_Xy(X)

        # Check size
        if self.filters_.shape[1] != X.shape[1]:
            raise ValueError('X must have %i channels, got %i instead.' % (
                self.filters_.shape[1], X.shape[1]))

        # Transform
        X = np.dot(self.filters_, X)
        X = X.transpose((1, 0, 2))
        return X

    def inverse_transform(self, X):
        """Remove selected components from the signal.

        Given the unmixing matrix, transform data, zero out components,
        and inverse transform the data. This procedure will reconstruct
        the signals from which the dynamics described by the excluded
        components is subtracted.

        Parameters
        ----------
        X : array, shape (n_epochs, n_components * n_classes, n_times)
            The transformed data.

        Returns
        -------
        X : array, shape (n_epochs, n_channels * n_classes, n_times)
            The inverse transform data.
        """
        # Check size
        X, _ = self._check_Xy(X)
        n_epochs, n_comp, n_times = X.shape
        if n_comp != (self.n_components * len(self.classes_)):
            raise ValueError('X must have %i components, got %i instead' % (
                self.n_components * len(self.classes_), n_comp))

        # Transform
        return np.dot(self.patterns_.T, X).transpose(1, 0, 2)

    def _check_Xy(self, X, y=None):
        """Check X and y types and dimensions."""
        # Check data
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError('X must be an array of shape (n_epochs, '
                             'n_channels, n_samples).')
        if y is None:
            y = np.ones(len(X))
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError('X and y must have the same length')
        return X, y


class Xdawn(_XdawnTransformer):
    """Implementation of the Xdawn Algorithm.

    Xdawn [1]_ [2]_ is a spatial filtering method designed to improve the
    signal to signal + noise ratio (SSNR) of the ERP responses. Xdawn was
    originally designed for P300 evoked potential by enhancing the target
    response with respect to the non-target response. This implementation
    is a generalization to any type of ERP.

    Parameters
    ----------
    n_components : int, (default 2)
        The number of components to decompose the signals.
    signal_cov : None | Covariance | ndarray, shape (n_channels, n_channels)
        (default None). The signal covariance used for whitening of the data.
        if None, the covariance is estimated from the epochs signal.
    correct_overlap : 'auto' or bool (default 'auto')
        Compute the independent evoked responses per condition, while
        correcting for event overlaps if any. If 'auto', then
        overlapp_correction = True if the events do overlap.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow
        regularization for covariance estimation.
        If float, shrinkage is used (0 <= shrinkage <= 1).
        For str options, ``reg`` will be passed as ``method`` to
        :func:`mne.compute_covariance`.

    Attributes
    ----------
    filters_ : dict of ndarray
        If fit, the Xdawn components used to decompose the data for each event
        type, else empty. For each event type, the filters are in the rows of
        the corresponding array.
    patterns_ : dict of ndarray
        If fit, the Xdawn patterns used to restore the signals for each event
        type, else empty.
    evokeds_ : dict of Evoked
        If fit, the evoked response for each event type.
    event_id_ : dict
        The event id.
    correct_overlap_ : bool
        Whether overlap correction was applied.

    See Also
    --------
    mne.decoding.CSP, mne.decoding.SPoC

    Notes
    -----
    .. versionadded:: 0.10

    References
    ----------
    .. [1] Rivet, B., Souloumiac, A., Attina, V., & Gibert, G. (2009). xDAWN
           algorithm to enhance evoked potentials: application to
           brain-computer interface. Biomedical Engineering, IEEE Transactions
           on, 56(8), 2035-2043.

    .. [2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., & Mattout, J.
           (2011, August). Theoretical analysis of xDAWN algorithm:
           application to an efficient sensor selection in a P300 BCI. In
           Signal Processing Conference, 2011 19th European (pp. 1382-1386).
           IEEE.
    """

    def __init__(self, n_components=2, signal_cov=None, correct_overlap='auto',
                 reg=None):
        """Init."""
        super(Xdawn, self).__init__(n_components=n_components,
                                    signal_cov=signal_cov, reg=reg)
        _check_option('correct_overlap', correct_overlap,
                      ['auto', True, False])
        self.correct_overlap = correct_overlap

    def fit(self, epochs, y=None):
        """Fit Xdawn from epochs.

        Parameters
        ----------
        epochs : instance of Epochs
            An instance of Epoch on which Xdawn filters will be fitted.
        y : ndarray | None (default None)
            If None, used epochs.events[:, 2].

        Returns
        -------
        self : instance of Xdawn
            The Xdawn instance.
        """
        # Check data
        if not isinstance(epochs, BaseEpochs):
            raise ValueError('epochs must be an Epochs object.')
        picks = _pick_data_channels(epochs.info)
        use_info = pick_info(epochs.info, picks)
        X = epochs.get_data()[:, picks, :]
        y = epochs.events[:, 2] if y is None else y
        self.event_id_ = epochs.event_id

        # Check that no baseline was applied with correct overlap
        correct_overlap = self.correct_overlap
        if correct_overlap == 'auto':
            # Events are overlapped if the minimal inter-stimulus
            # interval is smaller than the time window.
            isi = np.diff(np.sort(epochs.events[:, 0]))
            window = int((epochs.tmax - epochs.tmin) * epochs.info['sfreq'])
            correct_overlap = isi.min() < window

        if epochs.baseline and correct_overlap:
            raise ValueError('Cannot apply correct_overlap if epochs'
                             ' were baselined.')

        events, tmin, sfreq = None, 0., 1.
        if correct_overlap:
            events = epochs.events
            tmin = epochs.tmin
            sfreq = epochs.info['sfreq']
        self.correct_overlap_ = correct_overlap

        # Note: In this original version of Xdawn we compute and keep all
        # components. The selection comes at transform().
        n_components = X.shape[1]

        # Main fitting function
        filters, patterns, evokeds = _fit_xdawn(
            X, y, n_components=n_components, reg=self.reg,
            signal_cov=self.signal_cov, events=events, tmin=tmin, sfreq=sfreq,
            method_params=self.method_params, info=use_info)

        # Re-order filters and patterns according to event_id
        filters = filters.reshape(-1, n_components, filters.shape[-1])
        patterns = patterns.reshape(-1, n_components, patterns.shape[-1])
        self.filters_, self.patterns_, self.evokeds_ = dict(), dict(), dict()
        idx = np.argsort([value for _, value in epochs.event_id.items()])
        for eid, this_filter, this_pattern, this_evo in zip(
                epochs.event_id, filters[idx], patterns[idx], evokeds[idx]):
            self.filters_[eid] = this_filter
            self.patterns_[eid] = this_pattern
            n_events = len(epochs[eid])
            evoked = EvokedArray(this_evo, use_info, tmin=epochs.tmin,
                                 comment=eid, nave=n_events)
            self.evokeds_[eid] = evoked
        return self

    def transform(self, inst):
        """Apply Xdawn dim reduction.

        Parameters
        ----------
        inst : Epochs | Evoked | ndarray, shape ([n_epochs, ]n_channels, n_times)
            Data on which Xdawn filters will be applied.

        Returns
        -------
        X : ndarray, shape ([n_epochs, ]n_components * n_event_types, n_times)
            Spatially filtered signals.
        """  # noqa: E501
        if isinstance(inst, BaseEpochs):
            X = inst.get_data()
        elif isinstance(inst, Evoked):
            X = inst.data
        elif isinstance(inst, np.ndarray):
            X = inst
            if X.ndim not in (2, 3):
                raise ValueError('X must be 2D or 3D, got %s' % (X.ndim,))
        else:
            raise ValueError('Data input must be of Epoch type or numpy array')

        filters = [filt[:self.n_components]
                   for filt in self.filters_.values()]
        filters = np.concatenate(filters, axis=0)
        X = np.dot(filters, X)
        if X.ndim == 3:
            X = X.transpose((1, 0, 2))
        return X

    def apply(self, inst, event_id=None, include=None, exclude=None):
        """Remove selected components from the signal.

        Given the unmixing matrix, transform data,
        zero out components, and inverse transform the data.
        This procedure will reconstruct the signals from which
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
        out : dict
            A dict of instance (from the same type as inst input) for each
            event type in event_id.
        """
        if event_id is None:
            event_id = self.event_id_

        if not isinstance(inst, (BaseRaw, BaseEpochs, Evoked)):
            raise ValueError('Data input must be Raw, Epochs or Evoked type')
        picks = _pick_data_channels(inst.info)

        # Define the components to keep
        default_exclude = list(range(self.n_components, len(inst.ch_names)))
        if exclude is None:
            exclude = default_exclude
        else:
            exclude = list(set(list(default_exclude) + list(exclude)))

        if isinstance(inst, BaseRaw):
            out = self._apply_raw(raw=inst, include=include, exclude=exclude,
                                  event_id=event_id, picks=picks)
        elif isinstance(inst, BaseEpochs):
            out = self._apply_epochs(epochs=inst, include=include, picks=picks,
                                     exclude=exclude, event_id=event_id)
        elif isinstance(inst, Evoked):
            out = self._apply_evoked(evoked=inst, include=include, picks=picks,
                                     exclude=exclude, event_id=event_id)
        return out

    def _apply_raw(self, raw, include, exclude, event_id, picks):
        """Aux method."""
        if not raw.preload:
            raise ValueError('Raw data must be preloaded to apply Xdawn')

        raws = dict()
        for eid in event_id:
            data = raw[picks, :][0]

            data = self._pick_sources(data, include, exclude, eid)

            raw_r = raw.copy()

            raw_r[picks, :] = data
            raws[eid] = raw_r
        return raws

    def _apply_epochs(self, epochs, include, exclude, event_id, picks):
        """Aux method."""
        if not epochs.preload:
            raise ValueError('Epochs must be preloaded to apply Xdawn')

        # special case where epochs come picked but fit was 'unpicked'.
        epochs_dict = dict()
        data = np.hstack(epochs.get_data()[:, picks])

        for eid in event_id:

            data_r = self._pick_sources(data, include, exclude, eid)
            data_r = np.array(np.split(data_r, len(epochs.events), 1))
            epochs_r = epochs.copy().load_data()
            epochs_r._data[:, picks, :] = data_r
            epochs_dict[eid] = epochs_r

        return epochs_dict

    def _apply_evoked(self, evoked, include, exclude, event_id, picks):
        """Aux method."""
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
        logger.info('Transforming to Xdawn space')

        # Apply unmixing
        sources = np.dot(self.filters_[eid], data)

        if include not in (None, list()):
            mask = np.ones(len(sources), dtype=np.bool)
            mask[np.unique(include)] = False
            sources[mask] = 0.
            logger.info('Zeroing out %i Xdawn components' % mask.sum())
        elif exclude not in (None, list()):
            exclude_ = np.unique(exclude)
            sources[exclude_] = 0.
            logger.info('Zeroing out %i Xdawn components' % len(exclude_))
        logger.info('Inverse transforming to sensor space')
        data = np.dot(self.patterns_[eid].T, sources)

        return data

    def inverse_transform(self):
        """Not implemented, see Xdawn.apply() instead."""
        # Exists because of _XdawnTransformer
        raise NotImplementedError('See Xdawn.apply()')
