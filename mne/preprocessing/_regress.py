# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import numpy as np

from ..epochs import BaseEpochs
from ..io.pick import _picks_to_idx, pick_info
from ..io.base import BaseRaw
from ..utils import _check_preload, _validate_type, _check_option, verbose
from .eog import EOGRegression


@verbose
def regress_artifact(inst, picks=None, picks_artifact='eog', betas=None,
                     copy=True, verbose=None):
    """Remove artifacts using regression based on reference channels.

    Parameters
    ----------
    inst : instance of Epochs | Raw
        The instance to process.
    %(picks_good_data)s
    picks_artifact : array-like | str
        Channel picks to use as predictor/explanatory variables capturing
        the artifact of interest (default is "eog").
    betas : ndarray, shape (n_picks, n_picks_ref) | None
        The regression coefficients to use. If None (default), they will be
        estimated from the data.
    copy : bool
        If True (default), copy the instance before modifying it.
    %(verbose)s

    Returns
    -------
    inst : instance of Epochs | Raw
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

        >>> epochs_no_ave = epochs.copy().subtract_evoked()  # doctest:+SKIP
        >>> _, betas = mne.preprocessing.regress(epochs_no_ave)  # doctest:+SKIP
        >>> epochs_clean, _ = mne.preprocessing.regress(epochs, betas=betas)  # doctest:+SKIP

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    if betas is None:
        betas = EOGRegression(picks, picks_artifact).fit(inst)
    else:
        # Create an EOGRegression object and load the given betas into it.
        picks = _picks_to_idx(inst.info, picks, none='data')
        picks_artifact = _picks_to_idx(inst.info, picks_artifact)
        betas_shape = (len(picks), len(picks_artifact))
        _check_option('betas.shape', betas.shape, (betas_shape,))
        r = EOGRegression(picks, picks_artifact)
        r.info = pick_info(inst.info, picks)
        r.coef_ = betas
        betas = r
    return betas.apply(inst, copy=copy), betas.coef_
