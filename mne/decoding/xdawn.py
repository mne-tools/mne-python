"""Contains refactored Xdawn class."""

# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#          Asish Panda <asishrocks95@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..preprocessing.xdawn import _Xdawn
from . import TransformerMixin


class XdawnTransformer(TransformerMixin, _Xdawn):

    """Implementation of the Xdawn Algorithm.

    Xdawn is a spatial filtering method designed to improve the signal
    to signal + noise ratio (SSNR) of the ERP responses. Xdawn was originally
    designed for P300 evoked potential by enhancing the target response with
    respect to the non-target response. This implementation is a light version
    of the original version and follows scikit-learn API strictly.

    .. note:: This does not handle overlapping events. Use original
              preprocessing.xdawn instead.

    Parameters
    ----------
    n_components : int (default 2)
        The number of components to decompose M/EEG signals.
    signal_cov : None | Covariance | ndarray, shape (n_channels, n_channels)
        (default None). The signal covariance used for whitening of the data.
        if None, the covariance is estimated from the epochs signal.
    reg : float | str | None (default None)
        if not None, allow regularization for covariance estimation
        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
        or Oracle Approximating Shrinkage ('oas').

    Attributes
    ----------
    filters_ : dict of ndarray
        The Xdawn components used to decompose the data for each event type.
    patterns_ : dict of ndarray
        The Xdawn patterns used to restore M/EEG signals for each event type.


    See Also
    --------
    mne.preprocessing.Xdawn
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

    def __init__(self, n_components=2, signal_cov=None, reg=None):
        """init xdawn."""
        # XXX correct_overlap is not yet supported
        # This attribute is introduced to make it compatible with parent
        # class
        self.correct_overlap = False
        super(XdawnTransformer, self).__init__(n_components, signal_cov, reg)

    def fit(self, X, y):
        """Fit Xdawn from epochs.

        Parameters
        ----------
        X : ndarray, shape(n_channels, n_times, n_freqs)
            Data of epochs.
        y : ndarray shape(n_samples,)
            Target values.

        Returns
        -------
        self : XdawnTransformer instance
            The XdawnTransformer instance.
        """

        if X.ndim != 3 or not isinstance(X, np.ndarray):
            raise ValueError("X should be 3 dimensional ndarray")
        if not isinstance(y, np.ndarray):
            raise ValueError("Labels must be numpy array")

        self._n_chan = len(X[1])
        event_id = dict()
        for idx in y:
            event_id['cond' + str(idx)] = idx

        # Extract signal covariance
        self._get_signal_cov(X)

        # estimates evoked covariance
        self._fit_xdawn(X, y, event_id)
        self.event_id = event_id
        self.exclude = list(range(self.n_components, self._n_chan))
        return self

    def transform(self, X):
        """Apply Xdawn dimemsionality reduction.

        Parameters
        ----------
        X : ndarray, shape(n_channels, n_times, n_freqs)
            data of epochs.

        Returns
        -------
        X : ndarray, shape (n_epochs, n_components * event_types, n_times)
            Spatially filtered signals.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('Data input must be of type numpy array')

        # create full matrix of spatial filter
        return self._transform_xdawn(X)

    def fit_transform(self, X, y):
        """First fit the data, then transform it.

        Parameters
        ----------
        X : ndarray, shape(n_channels, n_times, n_freqs)
            data of epochs.
        y : ndarray, shape(n_samples,)
            labels of data.

        Returns
        -------
        X : ndarray, shape(n_epochs, n_components * event_types, n_times)
            spatially filtered signals.
        """
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        """Remove selected components from the signal.

        Given the unmixing matrix, transform data, zero out components,
        and inverse transform the data. This procedure will reconstruct
        M/EEG signals from which the dynamics described by the excluded
        components is subtracted.

        Parameters
        ----------
        X : np.ndarray, shape(n_epochs, n_components * event_types, n_times)
            The signal data to undergo inverse transform.

        Returns
        -------
        out : np.ndarray, shape(n_channels, n_times, n_freqs)
            Array for each event type in event_id concatenated sequentially.
        """

        if not isinstance(X, np.ndarray):
            raise ValueError("Given data should be numpy array, got "
                             "%s instead" % type(X))

        data = np.hstack(X)
        result = []
        event_id = self.event_id

        for eid in event_id:

            data_r = self._pick_sources(data, None, None, eid)
            data_r = np.array(np.split(data_r, self._n_chan, 1))
            result.append(data_r)

        return np.concatenate(result)
