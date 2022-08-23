# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import numpy as np

from ..defaults import (_INTERPOLATION_DEFAULT, _EXTRAPOLATE_DEFAULT,
                        _BORDER_DEFAULT)
from ..epochs import BaseEpochs
from ..io.pick import _picks_to_idx, pick_info
from ..io.base import BaseRaw
from ..utils import (_check_preload, _validate_type, _check_option, verbose,
                     fill_doc)
from ..minimum_norm.inverse import _needs_eeg_average_ref_proj
from .. import Evoked, EvokedArray


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


@fill_doc
class EOGRegression():
    """Remove EOG artifact signals from other channels by regression.

    Employs linear regression to remove signals captured by some channels,
    typically EOG, but it also works with ECG from other channels, as described
    in [1]_. You can also chose to fit the regression coefficients on evoked
    blink/saccade data and then apply them to continuous data, as described in
    [2]_.

    Parameters
    ----------
    %(picks_good_data)s
    picks_artifact : array-like | str
        Channel picks to use as predictor/explanatory variables capturing
        the artifact of interest (default is "eog").

    References
    ----------
    .. [1] Gratton, G. and Coles, M. G. H. and Donchin, E. (1983). A new method
           for off-line removal of ocular artifact. Electroencephalography and
           Clinical Neurophysiology, 468-484.
           https://doi.org/10.1016/0013-4694(83)90135-9
    .. [2] Croft, R. J. and Barry, R. J. (1998). EOG correction: a new
           aligned-artifact average solution. Clinical Neurophysiology, 107(6),
           395-401. http://doi.org/10.1016/s0013-4694(98)00087-x
    """

    def __init__(self, picks=None, picks_artifact='eog'):
        self._picks = picks
        self._picks_artifact = picks_artifact

    def fit(self, inst):
        """Fit EOG regression coefficients.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked
            The data on which the EOG regression weights should be fitted.

        Returns
        -------
        self : EOGRegression
            The fitted ``EOGRegression`` object. The regression coefficients
            are available as the ``.coef_`` and ``.intercep_`` attributes.

        Notes
        -----
        If your data contains EEG channels, make sure to apply the desired
        reference (see :func:`set_eeg_reference`) before performing EOG
        regression.
        """
        self._check_inst(inst)
        picks = _picks_to_idx(inst.info, self._picks, none='data')
        picks_artifact = _picks_to_idx(inst.info, self._picks_artifact)

        # Calculate regression coefficients. Add a row of ones to also fit the
        # intercept.
        _check_preload(inst, 'artifact regression')
        artifact_data = inst._data[..., picks_artifact, :]
        ref_data = artifact_data - np.mean(artifact_data, -1, keepdims=True)
        if ref_data.ndim == 3:
            ref_data = ref_data.transpose(1, 0, 2)
            ref_data = ref_data.reshape(len(picks_artifact), -1)
        cov_ref = ref_data @ ref_data.T

        # Process each channel separately to reduce memory load
        coef = np.zeros((len(picks), len(picks_artifact)))
        for pi, pick in enumerate(picks):
            this_data = inst._data[..., pick, :]  # view
            # Subtract mean over time from every trial/channel
            cov_data = this_data - np.mean(this_data, -1, keepdims=True)
            cov_data = cov_data.reshape(1, -1)
            # Perform the linear regression
            coef[pi] = np.linalg.solve(cov_ref, ref_data @ cov_data.T).T[0]

        # Store relevant parameters in the object.
        self.info = pick_info(inst.info, picks)
        self.coef_ = coef
        return self

    @fill_doc
    def apply(self, inst, copy=True):
        """Apply the regression coefficients to some data.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked
            The data on which to apply the regression.
        %(copy_df)s

        Returns
        -------
        inst : Raw | Epochs | Evoked
            A version of the data with the artifact channels regressed out.

        Notes
        -----
        Only works after ``.fit()`` has been used.
        """
        self._check_inst(inst)
        # The channels indices may not exactly match those of the object used
        # during .fit(). We align then using channel names.
        picks = [inst.ch_names.index(ch) for ch in self.info['ch_names']]
        picks_artifact = _picks_to_idx(inst.info, self._picks_artifact)

        if copy:
            inst = inst.copy()
        artifact_data = inst._data[..., picks_artifact, :]
        ref_data = artifact_data - np.mean(artifact_data, -1, keepdims=True)

        # Prepare the data matrix for regression
        _check_preload(inst, 'artifact regression')
        for pi, pick in enumerate(picks):
            this_data = inst._data[..., pick, :]  # view
            this_data -= (self.coef_[pi] @ ref_data).reshape(this_data.shape)

        return inst

    @fill_doc
    def plot(self, ch_type=None, vmin=None, vmax=None, cmap=None, sensors=True,
             colorbar=True, res=64, size=1, cbar_fmt='%3.1f', show=True,
             show_names=False, title='Regression coefficients', mask=None,
             mask_params=None, outlines='head', contours=6,
             image_interp=_INTERPOLATION_DEFAULT, axes=None,
             extrapolate=_EXTRAPOLATE_DEFAULT, sphere=None,
             border=_BORDER_DEFAULT):
        """Plot the regression weights.

        Parameters
        ----------
        %(ch_type_evoked_topomap)s
        %(vmin_vmax_topomap)s
        %(cmap_topomap)s
        %(sensors_topomap)s
        %(colorbar_topomap)s
        %(res_topomap)s
        %(size_topomap)s
        %(cbar_fmt_topomap)s
        %(show)s
        %(show_names_topomap)s
        %(title_none)s
        %(mask_evoked_topomap)s
        %(mask_params_topomap)s
        %(outlines_topomap)s
        %(contours_topomap)s
        %(image_interp_topomap)s
        %(axes_evoked_plot_topomap)s
        %(extrapolate_topomap)s
        %(sphere_topomap_auto)s
        %(border_topomap)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Figure with a topomap subplot for each channel type.

        Notes
        -----
        Only works after ``.fit()`` has been used.
        """
        ev = EvokedArray(self.coef_, self.info, comment='Regression coefs')
        return ev.plot_topomap(times=0, scalings=1, units='weight',
                               ch_type=ch_type, vmin=vmin, vmax=vmax,
                               cmap=cmap, sensors=sensors, colorbar=colorbar,
                               res=res, size=size, cbar_fmt=cbar_fmt,
                               show=show, show_names=show_names, title=title,
                               mask=mask, mask_params=mask_params,
                               outlines=outlines, contours=contours,
                               image_interp=image_interp, axes=axes,
                               extrapolate=extrapolate, sphere=sphere,
                               border=border, time_format='')

    def _check_inst(self, inst):
        """Perform some sanity checks on the input."""
        _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), 'inst',
                       'Raw, Epochs, Evoked')
        if _needs_eeg_average_ref_proj(inst.info):
            raise RuntimeError('No reference for the EEG channels has been '
                               'set. Use inst.set_eeg_reference to do so.')
        if not inst.proj:
            inst.apply_proj()
