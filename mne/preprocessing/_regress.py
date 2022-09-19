# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import numpy as np

from ..defaults import _BORDER_DEFAULT
from ..epochs import BaseEpochs
from ..io.pick import _picks_to_idx, pick_info
from ..io.base import BaseRaw
from ..utils import (_check_preload, _validate_type, _check_option, verbose,
                     fill_doc, copy_function_doc_to_method_doc, _check_fname,
                     _import_h5io_funcs)
from ..minimum_norm.inverse import _needs_eeg_average_ref_proj
from ..viz import plot_regression_weights
from .. import Evoked


@verbose
def regress_artifact(inst, picks=None, *, exclude='bads', picks_artifact='eog',
                     betas=None, taa=False, proj=True, copy=True,
                     verbose=None):
    """Remove artifacts using regression based on reference channels.

    Parameters
    ----------
    inst : instance of Epochs | Raw
        The instance to process.
    %(picks_good_data)s
    exclude : list | 'bads'
        List of channels to exclude from the regression, only used when picking
        based on types (e.g., exclude="bads" when picks="meg").
        Specify ``'bads'`` (the default) to exclude all channels marked as bad.

        .. versionadded:: 1.2
    picks_artifact : array-like | str
        Channel picks to use as predictor/explanatory variables capturing
        the artifact of interest (default is "eog").
    betas : ndarray, shape (n_picks, n_picks_ref) | None
        The regression coefficients to use. If None (default), they will be
        estimated from the data.
    taa : bool
        Whether to apply TAA correction as described in equation 9 of
        :footcite:`CroftBarry2000`. Defaults to ``False``.

        .. versionadded:: 1.2
    proj : bool
        Whether to automatically apply SSP projection vectors before performing
        the regression. Default is ``True``.
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
        model = EOGRegression(picks=picks, exclude=exclude,
                              picks_artifact=picks_artifact, taa=taa,
                              proj=proj)
        model.fit(inst)
    else:
        # Create an EOGRegression object and load the given betas into it.
        picks = _picks_to_idx(inst.info, picks, exclude=exclude, none='data')
        picks_artifact = _picks_to_idx(inst.info, picks_artifact)
        want_betas_shape = (len(picks), len(picks_artifact))
        _check_option('betas.shape', betas.shape, (want_betas_shape,))
        model = EOGRegression(picks, picks_artifact, taa=taa, proj=proj)
        model.info = pick_info(inst.info, picks)
        model.coef_ = betas
        model._picks = model.info['ch_names']
        model._exclude = exclude
        model._picks = model.info['ch_names']
        model._picks_artifact = [inst.ch_names[ch] for ch in picks_artifact]
    return model.apply(inst, copy=copy), model.coef_


@fill_doc
class EOGRegression():
    """Remove EOG artifact signals from other channels by regression.

    Employs linear regression to remove signals captured by some channels,
    typically EOG, but it also works with ECG from other channels, as described
    in :footcite:`GrattonEtAl1983`. You can also chose to fit the regression
    coefficients on evoked blink/saccade data and then apply them to continuous
    data, as described in :footcite:`CroftBarry2000`.

    Parameters
    ----------
    %(picks_good_data)s
    exclude : list | 'bads'
        List of channels to exclude from the regression, only used when picking
        based on types (e.g., exclude="bads" when picks="meg").
        Specify ``'bads'`` (the default) to exclude all channels marked as bad.
    picks_artifact : array-like | str
        Channel picks to use as predictor/explanatory variables capturing
        the artifact of interest (default is "eog").
    taa : bool
        Whether to apply TAA correction as described in equation 9 of
        :footcite:`CroftBarry2000`. Defaults to ``False``.
    proj : bool
        Whether to automatically apply SSP projection vectors before fitting
        and applying the regression. Default is ``True``.

    Notes
    -----
    .. versionadded:: 1.2

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, picks=None, exclude='bads', picks_artifact='eog',
                 taa=False, proj=True):
        self._picks = picks
        self._exclude = exclude
        self._picks_artifact = picks_artifact
        self.taa = taa
        self.proj = proj

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
            are available as the ``.coef_`` and ``.intercept_`` attributes.

        Notes
        -----
        If your data contains EEG channels, make sure to apply the desired
        reference (see :func:`mne.set_eeg_reference`) before performing EOG
        regression.
        """
        self._check_inst(inst)
        picks = _picks_to_idx(inst.info, self._picks, none='data',
                              exclude=self._exclude)
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
        self._picks = [inst.ch_names[ch] for ch in picks]
        self._picks_artifact = [inst.ch_names[ch] for ch in picks_artifact]
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

        References
        ----------
        .. footbibliography::
        """
        if copy:
            inst = inst.copy()
        self._check_inst(inst)
        # The channels indices may not exactly match those of the object used
        # during .fit(). We align then using channel names.
        picks = [inst.ch_names.index(ch) for ch in self._picks]
        picks_artifact = [inst.ch_names.index(ch)
                          for ch in self._picks_artifact]
        artifact_data = inst._data[..., picks_artifact, :]
        ref_data = artifact_data - np.mean(artifact_data, -1, keepdims=True)

        # Prepare the data matrix
        _check_preload(inst, 'artifact regression')
        for pi, pick in enumerate(picks):
            this_data = inst._data[..., pick, :]  # view
            this_data -= (self.coef_[pi] @ ref_data).reshape(this_data.shape)
            if self.taa:
                # TAA correction (Croft & Barry 2000, eqn. 9)
                this_data /= 1 - np.sum(self.coef_[pi] ** 2)

        return inst

    @copy_function_doc_to_method_doc(plot_regression_weights)
    def plot(self, ch_type=None, vmin=None, vmax=None, cmap=None, sensors=True,
             colorbar=True, res=64, size=1, cbar_fmt='%.2g', show=True,
             show_names=False, title=None, mask=None, mask_params=None,
             outlines='head', axes=None, sphere=None, border=_BORDER_DEFAULT):
        return plot_regression_weights(self, ch_type=ch_type, vmin=vmin,
                                       vmax=vmax, cmap=cmap, sensors=sensors,
                                       res=res, show=show,
                                       show_names=show_names, mask=mask,
                                       mask_params=mask_params,
                                       outlines=outlines, axes=axes,
                                       sphere=sphere, border=border)

    def _check_inst(self, inst):
        """Perform some sanity checks on the input."""
        _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), 'inst',
                       'Raw, Epochs, Evoked')
        if _needs_eeg_average_ref_proj(inst.info):
            raise RuntimeError('No reference for the EEG channels has been '
                               'set. Use inst.set_eeg_reference to do so.')
        if self.proj and not inst.proj:
            inst.apply_proj()
        if not inst.proj and len(inst.info.get('projs', [])) > 0:
            raise RuntimeError('Projections need to be applied before '
                               'regression can be performed. Use the '
                               '.apply_proj() method to do so.')

    def __repr__(self):
        """Produce a string representation of this object."""
        s = '<EOGRegression | '
        if hasattr(self, 'coef_'):
            n_art = self.coef_.shape[1]
            plural = 's' if n_art > 1 else ''
            s += f'fitted to {n_art} artifact channel{plural}, '
        else:
            s += 'not fitted, '
        s += f'TAA={self.taa}>'
        return s

    @fill_doc
    def save(self, fname, overwrite=False):
        """Save the regression model to an HDF5 file.

        Parameters
        ----------
        fname : str
            The file to write the regression weights to. Should end in ``.h5``.
        %(overwrite)s
        """
        _, write_hdf5 = _import_h5io_funcs()
        _validate_type(fname, 'path-like', 'fname')
        fname = _check_fname(fname, overwrite=overwrite, name='fname')
        write_hdf5(fname, self.__dict__, overwrite=overwrite)


def read_eog_regression(fname):
    """Read an EOG regression model from an HDF5 file.

    Parameters
    ----------
    fname : str
        The file to read the regression model from. Should end in ``.h5``.

    Returns
    -------
    model : EOGRegression
        The regression model read from the file.

    Notes
    -----
    .. versionadded:: 1.2
    """
    read_hdf5, _ = _import_h5io_funcs()
    _validate_type(fname, 'path-like', 'fname')
    fname = _check_fname(fname, overwrite='read', must_exist=True,
                         name='fname')
    model = EOGRegression()
    model.__dict__.update(read_hdf5(fname))
    return model
