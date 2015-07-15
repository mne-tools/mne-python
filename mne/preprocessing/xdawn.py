"""Xdawn implementation."""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
from scipy import linalg
import copy as cp

from ..io.proj import Projection
from ..io.base import _BaseRaw
from ..epochs import _BaseEpochs
from .. import Covariance, EvokedArray, Evoked
from ..io.pick import pick_types
from .ica import _get_fast_dot
from ..utils import logger
from ..decoding.mixin import TransformerMixin
from ..channels.channels import ContainsMixin


def _least_square_evoked(data, events, event_id, tmin, tmax, sfreq, decim):
    """Least square estimation of evoked response from data.

    Parameters
    ----------
    data : array, shape=(n_channels, n_times)
        The data to estimates evoked
    events : array, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be ignored.
    event_id : dict
        The id of the event to consider
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    sfreq : float
        Sampling frequency.
    decim : int
        The decimation factor.

    Returns
    -------
    evokeds : dict of evoked instance
        A dict of evoked instance for each event type in event_id.
    toeplitz : dict of array
        A dict of toeplitz matrix for each event type in event_id.
    """
    nmin = int(tmin * sfreq / decim)
    nmax = int(tmax * sfreq / decim)
    data = data[:, ::decim]

    window = nmax - nmin
    ne, ns = data.shape
    to = dict()

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
    evoked_data = dict()
    for i, eid in enumerate(event_id):
        evoked_data[eid] = np.array(evo[(i * window):(i + 1) * window, :]).T

    return evoked_data, to


def _check_overlapp(epochs):
    """check if events are overlapped."""
    isi = np.diff(epochs.events[:, 0])
    window = int((epochs.tmax - epochs.tmin) * epochs.info['sfreq'])
    return isi.min() < window


def _construct_signal_from_epochs(epochs):
    """Reconstruct pseudo continuous signal from epochs."""
    start_ix = (np.min(epochs.events[:, 0])
                + int(epochs.tmin * epochs.info['sfreq']))
    end_ix = (np.max(epochs.events[:, 0])
              + int(epochs.tmax * epochs.info['sfreq']) + 1)

    ns = end_ix - start_ix
    ns_epochs = epochs._data.shape[2]
    ne = epochs._data.shape[1]
    ix_events = epochs.events[:, 0] - epochs.events[0, 0]

    data = np.zeros((ne, ns))
    for i, ix in enumerate(ix_events):
        data[:, ix:(ix + ns_epochs)] = epochs._data[i]

    return data


def least_square_evoked(epochs, return_toeplitz=False):
    """Least square estimation of evoked response from a epoch instance.

    Parameters
    ----------
    epochs : Epoch object
        An instance of Epoch.
    return_toeplitz : bool
        If true, compute the toeplitz matrix.

    Returns
    -------
    evokeds : dict of evoked instance
        An dict of evoked instance for each event type in epochs.event_id.
    toeplitz : dict of array
        If return_toeplitz is true, return the toeplitz matrix for each event
        type in epochs.event_id.
    """
    if not isinstance(epochs, _BaseEpochs):
        raise ValueError('epochs must be an instance of `mne.Epochs`')

    evs = epochs.events.copy()
    evs[:, 0] -= evs[0, 0] + int(epochs.tmin * epochs.info['sfreq'])
    data = _construct_signal_from_epochs(epochs)
    evo, to = _least_square_evoked(data, evs, epochs.event_id,
                                   tmin=epochs.tmin, tmax=epochs.tmax,
                                   sfreq=epochs.info['sfreq'], decim=1)
    evokeds = dict()
    info = cp.deepcopy(epochs.info)
    for name, data in evo.items():
        n_events = len(evs[evs[:, 2] == epochs.event_id[name]])
        evoked = EvokedArray(data, info, tmin=epochs.tmin,
                             comment=name, nave=n_events)
        evokeds[name] = evoked

    if return_toeplitz:
        return evokeds, to

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
    n_components : int, default 2
        The number of components to decompose M/EEG signals.
    signal_cov : None, Covariance instance or ndarray
        The signal covariance used for whitening of the data. if None, the
        covariance is estimated from the epochs signal.
    correct_overlap : 'auto' or bool
        Apply correction for overlaped ERP for the estimation of evokeds
        responses. if 'auto', the overlapp correction is chosen in function
        of the events in epochs.events.
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
        If fit, the evoked response for each event type.

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

    def __init__(self, n_components=2, correct_overlap='auto', signal_cov=None,
                 reg=None):
        """init xdawn."""
        self.n_components = n_components
        self.signal_cov = signal_cov
        self.reg = reg
        self.filters_ = dict()
        self.patterns_ = dict()
        self.evokeds_ = dict()
        self.projs_ = dict()

        if correct_overlap not in ['auto', True, False]:
            raise ValueError('correct_overlap must be a bool or "auto"')
        self.correct_overlap = correct_overlap

        # Extract signal covariance
        if signal_cov is None:
            self.signal_cov_ = None
        elif isinstance(signal_cov, Covariance):
            self.signal_cov_ = signal_cov.data
        elif isinstance(signal_cov, np.ndarray):
            self.signal_cov_ = signal_cov
        else:
            raise ValueError('signal_cov must be None, a covariance instance '
                             'or a ndarray')

    def fit(self, epochs, y=None):
        """Fit Xdawn from epochs.

        Parameters
        ----------
        epochs : Epoch object
            An instance of Epoch on which Xdawn filters will be trained.
        y : None
            Compatibility with decoding.
        Returns
        -------
        self : Xdawn instance
            The Xdawn instance.
        """
        if self.correct_overlap == 'auto':
            self.correct_overlap = _check_overlapp(epochs)

        # Extract signal covariance
        if self.signal_cov_ is None:
            if self.correct_overlap:
                sig_data = _construct_signal_from_epochs(epochs)
            else:
                sig_data = np.hstack(epochs.get_data())
            # FIXME use MNE cov estimator
            self.signal_cov_ = np.cov(sig_data)

        # estimates evoked covariance
        self.evokeds_cov_ = {}
        if self.correct_overlap:
            if epochs.baseline is not None:
                raise ValueError('Baseline correction must be None if overlap '
                                 'correction activated')
            evo, to = least_square_evoked(epochs, return_toeplitz=True)
        else:
            evo = dict()
            to = dict()
            for eid in epochs.event_id:
                evo[eid] = epochs[eid].average()
                to[eid] = 1.0 / len(epochs[eid])
        self.evokeds_ = evo

        for eid in epochs.event_id:
            # FIXME use mne covariance estimator
            self.evokeds_cov_[eid] = np.cov(np.dot(evo[eid].data, to[eid]))

        # estimates spatial filters
        for eid in epochs.event_id:

            if self.signal_cov_.shape != self.evokeds_cov_[eid].shape:
                raise ValueError('Size of signal cov must be the same as the'
                                 ' number of channels in epochs')

            evals, evecs = linalg.eigh(self.evokeds_cov_[eid],
                                       self.signal_cov_)
            evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
            evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)

            self.filters_[eid] = evecs
            self.patterns_[eid] = linalg.inv(evecs.T)

            # Create and add projectors
            ch_names = self.evokeds_[eid].ch_names
            projs = []
            for j in range(self.n_components):
                proj_data = dict(col_names=ch_names, row_names=None,
                                 data=evecs[:, j].T, nrow=1,
                                 ncol=len(ch_names))
                projs.append(Projection(active=True, data=proj_data,
                             desc="%s Xdawn #%d" % (eid, j)))

            self.projs_[eid] = projs

        # store some values
        self.ch_names = epochs.ch_names
        self.exclude = range(self.n_components, len(self.ch_names))
        self.event_id = epochs.event_id
        return self

    def transform(self, epochs):
        """Apply Xdawn dim reduction.

        Parameters
        ----------
        epochs : Epoch or array, shape=(n_trial x n_channels x n_times)
            Data on which Xdawn filters will be applied.

        Returns
        -------
        X : array, shape(n_trials x n_components * event_types x n_times)
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
        V = []
        for f in self.filters_.values():
            V.append(f[:, 0:self.n_components])
        V = np.concatenate(V, axis=1)

        # Apply spatial filters
        X = np.dot(V.T, data)
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
        inst : instance of Raw, Epochs or Evoked
            The data to be processed.
        event_id : dict or None
            The kind of event to apply. if none, a dict of inst will be return
            one for each type of event xdawn has been fitted.
        include : array_like of int.
            The indices refering to columns in the ummixing matrix. The
            components to be kept.
        exclude : array_like of int.
            The indices refering to columns in the ummixing matrix. The
            components to be zeroed out.
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
        """Aux method."""
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
        """Aux method."""
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
        """Aux method."""
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
        """Aux method."""
        fast_dot = _get_fast_dot()
        if exclude is None:
            exclude = self.exclude
        else:
            exclude = list(set(self.exclude + list(exclude)))

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
