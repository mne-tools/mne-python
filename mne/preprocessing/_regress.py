# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..epochs import BaseEpochs
from ..io.pick import _picks_to_idx
from ..io.base import BaseRaw
from ..utils import _check_preload, _validate_type, verbose


@verbose
def regress(inst, picks=None, picks_ref='eog', betas=None, copy=True,
            verbose=None):
    """Regress artifacts using reference channels.

    Parameters
    ----------
    inst : instance of Epochs | Raw
        The instance to process.
    %(picks_good_data)s
    picks_ref : array-like
        Picks to use as the reference channels.
    betas : ndarray, shape (n_picks, n_picks_ref) | None
        The regression coeffients to use. If None, they will be estimated
        from the data.
    copy : bool
        If True (default), copy the instance before modifying it.
    %(verbose)s

    Returns
    -------
    inst : instance of Epochs
        The processed data.
    betas : ndarray, shape (n_picks, n_picks_ref)
        The betas used during regression.

    Notes
    -----
    To implement the method outlined in :footcite:`GrattonEtAl1983`,
    remove the evoked response from epochs before estimating the
    regression coefficients, then apply those regression coefficients to the
    original data in two calls like (here for a single-condition ``epochs``
    only):

        >>> epochs_no_ave = epochs.copy().subtract_evoked()
        >>> _, betas = mne.preprocessing.regress(epochs_no_ave)
        >>> epochs_clean, _ = mne.preprocessing.regress(epochs, betas=betas)

    References
    ----------
    .. footbibliography::
    """
    _check_preload(inst, 'regress')
    _validate_type(inst, (BaseEpochs, BaseRaw), 'inst', 'Epochs or Raw')
    picks = _picks_to_idx(inst.info, picks, none='data')
    picks_ref = _picks_to_idx(inst.info, picks_ref, allow_empty=False)
    if np.in1d(picks_ref, picks).any():
        raise ValueError('ref_picks cannot be contained in picks')
    inst = inst.copy() if copy else inst
    ref_data = inst._data[..., picks_ref, :]
    ref_data = ref_data - np.mean(ref_data, -1, keepdims=True)
    if ref_data.ndim == 3:
        ref_data = ref_data.transpose(1, 0, 2).reshape(len(picks_ref), -1)
    cov = np.dot(ref_data, ref_data.T)
    # process each one separately to reduce memory load
    betas_shape = (len(picks), len(picks_ref))
    if betas is None:
        betas = np.empty(betas_shape)
        estimate = True
    else:
        estimate = False
        assert betas.shape == betas_shape
    for pi, pick in enumerate(picks):
        this_data = inst._data[..., pick, :]  # view
        orig_shape = this_data.shape
        if estimate:
            # subtract mean over time from every trial/channel
            cov_data = this_data - np.mean(this_data, -1, keepdims=True)
            cov_data = cov_data.reshape(1, -1)
            betas[pi] = np.linalg.solve(cov, np.dot(ref_data, cov_data.T)).T[0]
        # subtract weighted (demeaned) eye channels from channel
        this_data -= (betas[pi] @ ref_data).reshape(orig_shape)
    return inst, betas
