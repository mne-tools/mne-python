# -*- coding: utf-8 -*-
#
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Juergen Dammers <j.dammers@fz-juelich.de>
#
# License: BSD-3-Clause

from inspect import isfunction
from collections import namedtuple
from copy import deepcopy
from numbers import Integral
from time import time
from dataclasses import dataclass
from typing import Optional, List

import math
import os
import json

import numpy as np

from .ecg import (qrs_detector, _get_ecg_channel_index, _make_ecg,
                  create_ecg_epochs)
from .eog import _find_eog_events, _get_eog_channel_index
from .infomax_ import infomax

from ..cov import compute_whitener
from .. import Covariance, Evoked
from ..io.pick import (pick_types, pick_channels, pick_info,
                       _picks_to_idx, _get_channel_types, _DATA_CH_TYPES_SPLIT)
from ..io.proj import make_projector
from ..io.write import (write_double_matrix, write_string,
                        write_name_list, write_int, start_block,
                        end_block)
from ..io.tree import dir_tree_find
from ..io.open import fiff_open
from ..io.tag import read_tag
from ..io.meas_info import write_meas_info, read_meas_info
from ..io.constants import FIFF
from ..io.base import BaseRaw
from ..io.eeglab.eeglab import _get_info, _check_load_mat

from ..epochs import BaseEpochs
from ..viz import (plot_ica_components, plot_ica_scores,
                   plot_ica_sources, plot_ica_overlay)
from ..viz.ica import plot_ica_properties
from ..viz.topomap import _plot_corrmap

from ..channels.channels import _contains_ch_type, ContainsMixin
from ..io.write import start_file, end_file, write_id
from ..utils import (check_version, logger, check_fname, _check_fname, verbose,
                     _reject_data_segments, check_random_state, _validate_type,
                     compute_corr, _get_inst_data, _ensure_int,
                     copy_function_doc_to_method_doc, _pl, warn, Bunch,
                     _check_preload, _check_compensation_grade, fill_doc,
                     _check_option, _PCA, int_like,
                     _check_all_same_channel_names, deprecated)

from ..fixes import _get_args, _safe_svd
from ..filter import filter_data
from .bads import _find_outliers
from .ctps_ import ctps
from ..io.pick import pick_channels_regexp, _picks_by_type
from ..data.html_templates import ica_template

__all__ = ('ICA', 'ica_find_ecg_events', 'ica_find_eog_events',
           'get_score_funcs', 'read_ica', 'read_ica_eeglab')


def _make_xy_sfunc(func, ndim_output=False):
    """Aux function."""
    if ndim_output:
        def sfunc(x, y):
            return np.array([func(a, y.ravel()) for a in x])[:, 0]
    else:
        def sfunc(x, y):
            return np.array([func(a, y.ravel()) for a in x])
    sfunc.__name__ = '.'.join(['score_func', func.__module__, func.__name__])
    sfunc.__doc__ = func.__doc__
    return sfunc


# Violate our assumption that the output is 1D so can't be used.
# Could eventually be added but probably not worth the effort unless someone
# requests it.
_BLOCKLIST = {'somersd'}


# makes score funcs attr accessible for users
def get_score_funcs():
    """Get the score functions.

    Returns
    -------
    score_funcs : dict
        The score functions.
    """
    from scipy import stats
    from scipy.spatial import distance
    score_funcs = Bunch()
    xy_arg_dist_funcs = [(n, f) for n, f in vars(distance).items()
                         if isfunction(f) and not n.startswith('_') and
                         n not in _BLOCKLIST]
    xy_arg_stats_funcs = [(n, f) for n, f in vars(stats).items()
                          if isfunction(f) and not n.startswith('_') and
                          n not in _BLOCKLIST]
    score_funcs.update({n: _make_xy_sfunc(f)
                        for n, f in xy_arg_dist_funcs
                        if _get_args(f) == ['u', 'v']})
    score_funcs.update({n: _make_xy_sfunc(f, ndim_output=True)
                        for n, f in xy_arg_stats_funcs
                        if _get_args(f) == ['x', 'y']})
    return score_funcs


def _check_for_unsupported_ica_channels(picks, info, allow_ref_meg=False):
    """Check for channels in picks that are not considered valid channels.

    Accepted channels are the data channels
    ('seeg', 'dbs', 'ecog', 'eeg', 'hbo', 'hbr', 'mag', and 'grad'), 'eog'
    and 'ref_meg'.
    This prevents the program from crashing without
    feedback when a bad channel is provided to ICA whitening.
    """
    types = _DATA_CH_TYPES_SPLIT + ('eog',)
    types += ('ref_meg',) if allow_ref_meg else ()
    chs = _get_channel_types(info, picks, unique=True, only_data_chs=False)
    check = all([ch in types for ch in chs])
    if not check:
        raise ValueError('Invalid channel type%s passed for ICA: %s.'
                         'Only the following types are supported: %s'
                         % (_pl(chs), chs, types))


_KNOWN_ICA_METHODS = ('fastica', 'infomax', 'picard')


@fill_doc
class ICA(ContainsMixin):
    u"""Data decomposition using Independent Component Analysis (ICA).

    This object estimates independent components from :class:`mne.io.Raw`,
    :class:`mne.Epochs`, or :class:`mne.Evoked` objects. Components can
    optionally be removed (for artifact repair) prior to signal reconstruction.

    .. warning:: ICA is sensitive to low-frequency drifts and therefore
                 requires the data to be high-pass filtered prior to fitting.
                 Typically, a cutoff frequency of 1 Hz is recommended.

    Parameters
    ----------
    n_components : int | float | None
        Number of principal components (from the pre-whitening PCA step) that
        are passed to the ICA algorithm during fitting:

        - :class:`int`
            Must be greater than 1 and less than or equal to the number of
            channels.
        - :class:`float` between 0 and 1 (exclusive)
            Will select the smallest number of components required to explain
            the cumulative variance of the data greater than ``n_components``.
            Consider this hypothetical example: we have 3 components, the first
            explaining 70%%, the second 20%%, and the third the remaining 10%%
            of the variance. Passing 0.8 here (corresponding to 80%% of
            explained variance) would yield the first two components,
            explaining 90%% of the variance: only by using both components the
            requested threshold of 80%% explained variance can be exceeded. The
            third component, on the other hand, would be excluded.
        - ``None``
            ``0.999999`` will be used. This is done to avoid numerical
            stability problems when whitening, particularly when working with
            rank-deficient data.

        Defaults to ``None``. The actual number used when executing the
        :meth:`ICA.fit` method will be stored in the attribute
        ``n_components_`` (note the trailing underscore).

        .. versionchanged:: 0.22
           For a :class:`python:float`, the number of components will account
           for *greater than* the given variance level instead of *less than or
           equal to* it. The default (None) will also take into account the
           rank deficiency of the data.
    noise_cov : None | instance of Covariance
        Noise covariance used for pre-whitening. If None (default), channels
        are scaled to unit variance ("z-standardized") as a group by channel
        type prior to the whitening by PCA.
    %(random_state)s
    method : 'fastica' | 'infomax' | 'picard'
        The ICA method to use in the fit method. Use the ``fit_params`` argument
        to set additional parameters. Specifically, if you want Extended
        Infomax, set ``method='infomax'`` and ``fit_params=dict(extended=True)``
        (this also works for ``method='picard'``). Defaults to ``'fastica'``.
        For reference, see :footcite:`Hyvarinen1999,BellSejnowski1995,LeeEtAl1999,AblinEtAl2018`.
    fit_params : dict | None
        Additional parameters passed to the ICA estimator as specified by
        ``method``. Allowed entries are determined by the various algorithm
        implementations: see :class:`~sklearn.decomposition.FastICA`,
        :func:`~picard.picard`, :func:`~mne.preprocessing.infomax`.
    max_iter : int | 'auto'
        Maximum number of iterations during fit. If ``'auto'``, it
        will set maximum iterations to ``1000`` for ``'fastica'``
        and to ``500`` for ``'infomax'`` or ``'picard'``. The actual number of
        iterations it took :meth:`ICA.fit` to complete will be stored in the
        ``n_iter_`` attribute.
    allow_ref_meg : bool
        Allow ICA on MEG reference channels. Defaults to False.

        .. versionadded:: 0.18
    %(verbose)s

    Attributes
    ----------
    current_fit : str
        Flag informing about which data type (raw or epochs) was used for the
        fit.
    ch_names : list-like
        Channel names resulting from initial picking.
    n_components_ : int
        If fit, the actual number of PCA components used for ICA decomposition.
    pre_whitener_ : ndarray, shape (n_channels, 1) or (n_channels, n_channels)
        If fit, array used to pre-whiten the data prior to PCA.
    pca_components_ : ndarray, shape ``(n_channels, n_channels)``
        If fit, the PCA components.
    pca_mean_ : ndarray, shape (n_channels,)
        If fit, the mean vector used to center the data before doing the PCA.
    pca_explained_variance_ : ndarray, shape ``(n_channels,)``
        If fit, the variance explained by each PCA component.
    mixing_matrix_ : ndarray, shape ``(n_components_, n_components_)``
        If fit, the whitened mixing matrix to go back from ICA space to PCA
        space.
        It is, in combination with the ``pca_components_``, used by
        :meth:`ICA.apply` and :meth:`ICA.get_components` to re-mix/project
        a subset of the ICA components into the observed channel space.
        The former method also removes the pre-whitening (z-scaling) and the
        de-meaning.
    unmixing_matrix_ : ndarray, shape ``(n_components_, n_components_)``
        If fit, the whitened matrix to go from PCA space to ICA space.
        Used, in combination with the ``pca_components_``, by the methods
        :meth:`ICA.get_sources` and :meth:`ICA.apply` to unmix the observed
        data.
    exclude : array-like of int
        List or np.array of sources indices to exclude when re-mixing the data
        in the :meth:`ICA.apply` method, i.e. artifactual ICA components.
        The components identified manually and by the various automatic
        artifact detection methods should be (manually) appended
        (e.g. ``ica.exclude.extend(eog_inds)``).
        (There is also an ``exclude`` parameter in the :meth:`ICA.apply`
        method.) To scrap all marked components, set this attribute to an empty
        list.
    %(info)s
    n_samples_ : int
        The number of samples used on fit.
    labels_ : dict
        A dictionary of independent component indices, grouped by types of
        independent components. This attribute is set by some of the artifact
        detection functions.
    n_iter_ : int
        If fit, the number of iterations required to complete ICA.

    Notes
    -----
    .. versionchanged:: 0.23
        Version 0.23 introduced the ``max_iter='auto'`` settings for maximum
        iterations. With version 0.24 ``'auto'`` will be the new
        default, replacing the current ``max_iter=200``.

    .. versionchanged:: 0.23
        Warn if `~mne.Epochs` were baseline-corrected.

    .. note:: If you intend to fit ICA on `~mne.Epochs`, it is  recommended to
              high-pass filter, but **not** baseline correct the data for good
              ICA performance. A warning will be emitted otherwise.

    A trailing ``_`` in an attribute name signifies that the attribute was
    added to the object during fitting, consistent with standard scikit-learn
    practice.

    ICA :meth:`fit` in MNE proceeds in two steps:

    1. :term:`Whitening <whitening>` the data by means of a pre-whitening step
       (using ``noise_cov`` if provided, or the standard deviation of each
       channel type) and then principal component analysis (PCA).
    2. Passing the ``n_components`` largest-variance components to the ICA
       algorithm to obtain the unmixing matrix (and by pseudoinversion, the
       mixing matrix).

    ICA :meth:`apply` then:

    1. Unmixes the data with the ``unmixing_matrix_``.
    2. Includes ICA components based on ``ica.include`` and ``ica.exclude``.
    3. Re-mixes the data with ``mixing_matrix_``.
    4. Restores any data not passed to the ICA algorithm, i.e., the PCA
       components between ``n_components`` and ``n_pca_components``.

    ``n_pca_components`` determines how many PCA components will be kept when
    reconstructing the data when calling :meth:`apply`. This parameter can be
    used for dimensionality reduction of the data, or dealing with low-rank
    data (such as those with projections, or MEG data processed by SSS). It is
    important to remove any numerically-zero-variance components in the data,
    otherwise numerical instability causes problems when computing the mixing
    matrix. Alternatively, using ``n_components`` as a float will also avoid
    numerical stability problems.

    The ``n_components`` parameter determines how many components out of
    the ``n_channels`` PCA components the ICA algorithm will actually fit.
    This is not typically used for EEG data, but for MEG data, it's common to
    use ``n_components < n_channels``. For example, full-rank
    306-channel MEG data might use ``n_components=40`` to find (and
    later exclude) only large, dominating artifacts in the data, but still
    reconstruct the data using all 306 PCA components. Setting
    ``n_pca_components=40``, on the other hand, would actually reduce the
    rank of the reconstructed data to 40, which is typically undesirable.

    If you are migrating from EEGLAB and intend to reduce dimensionality via
    PCA, similarly to EEGLAB's ``runica(..., 'pca', n)`` functionality,
    pass ``n_components=n`` during initialization and then
    ``n_pca_components=n`` during :meth:`apply`. The resulting reconstructed
    data after :meth:`apply` will have rank ``n``.

    .. note:: Commonly used for reasons of i) computational efficiency and
              ii) additional noise reduction, it is a matter of current debate
              whether pre-ICA dimensionality reduction could decrease the
              reliability and stability of the ICA, at least for EEG data and
              especially during preprocessing :footcite:`ArtoniEtAl2018`.
              (But see also :footcite:`Montoya-MartinezEtAl2017` for a
              possibly confounding effect of the different whitening/sphering
              methods used in this paper (ZCA vs. PCA).)
              On the other hand, for rank-deficient data such as EEG data after
              average reference or interpolation, it is recommended to reduce
              the dimensionality (by 1 for average reference and 1 for each
              interpolated channel) for optimal ICA performance (see the
              `EEGLAB wiki <eeglab_wiki_>`_).

    Caveat! If supplying a noise covariance, keep track of the projections
    available in the cov or in the raw object. For example, if you are
    interested in EOG or ECG artifacts, EOG and ECG projections should be
    temporally removed before fitting ICA, for example::

        >> projs, raw.info['projs'] = raw.info['projs'], []
        >> ica.fit(raw)
        >> raw.info['projs'] = projs

    Methods currently implemented are FastICA (default), Infomax, and Picard.
    Standard Infomax can be quite sensitive to differences in floating point
    arithmetic. Extended Infomax seems to be more stable in this respect,
    enhancing reproducibility and stability of results; use Extended Infomax
    via ``method='infomax', fit_params=dict(extended=True)``. Allowed entries
    in ``fit_params`` are determined by the various algorithm implementations:
    see :class:`~sklearn.decomposition.FastICA`, :func:`~picard.picard`,
    :func:`~mne.preprocessing.infomax`.

    .. note:: Picard can be used to solve the same problems as FastICA,
              Infomax, and extended Infomax, but typically converges faster
              than either of those methods. To make use of Picard's speed while
              still obtaining the same solution as with other algorithms, you
              need to specify ``method='picard'`` and ``fit_params`` as a
              dictionary with the following combination of keys:

              - ``dict(ortho=False, extended=False)`` for Infomax
              - ``dict(ortho=False, extended=True)`` for extended Infomax
              - ``dict(ortho=True, extended=True)`` for FastICA

    Reducing the tolerance (set in ``fit_params``) speeds up estimation at the
    cost of consistency of the obtained results. It is difficult to directly
    compare tolerance levels between Infomax and Picard, but for Picard and
    FastICA a good rule of thumb is ``tol_fastica == tol_picard ** 2``.

    .. _eeglab_wiki: https://eeglab.org/tutorials/06_RejectArtifacts/RunICA.html#how-to-deal-with-corrupted-ica-decompositions

    References
    ----------
    .. footbibliography::
    """  # noqa: E501

    @verbose
    def __init__(self, n_components=None, *, noise_cov=None,
                 random_state=None, method='fastica', fit_params=None,
                 max_iter='auto', allow_ref_meg=False,
                 verbose=None):  # noqa: D102
        _validate_type(method, str, 'method')
        _validate_type(n_components, (float, 'int-like', None))

        if method != 'imported_eeglab':  # internal use only
            _check_option('method', method, _KNOWN_ICA_METHODS)
        if method == 'fastica' and not check_version('sklearn'):
            raise ImportError(
                'The scikit-learn package is required for method="fastica".')
        if method == 'picard' and not check_version('picard'):
            raise ImportError(
                'The python-picard package is required for method="picard".')

        self.noise_cov = noise_cov

        for (kind, val) in [('n_components', n_components)]:
            if isinstance(val, float) and not 0 < val < 1:
                raise ValueError('Selecting ICA components by explained '
                                 'variance needs values between 0.0 and 1.0 '
                                 f'(exclusive), got {kind}={val}')
            if isinstance(val, int_like) and val == 1:
                raise ValueError(
                    f'Selecting one component with {kind}={val} is not '
                    'supported')

        self.current_fit = 'unfitted'
        self.verbose = verbose
        self.n_components = n_components
        # In newer ICAs this should always be None, but keep it for
        # backward compat with older versions of MNE that used it
        self._max_pca_components = None
        self.n_pca_components = None
        self.ch_names = None
        self.random_state = random_state

        if fit_params is None:
            fit_params = {}
        fit_params = deepcopy(fit_params)  # avoid side effects

        if method == 'fastica':
            update = {'algorithm': 'parallel', 'fun': 'logcosh',
                      'fun_args': None}
            fit_params.update({k: v for k, v in update.items() if k
                               not in fit_params})
        elif method == 'infomax':
            # extended=True is default in underlying function, but we want
            # default False here unless user specified True:
            fit_params.setdefault('extended', False)
        _validate_type(max_iter, (str, 'int-like'), 'max_iter')
        if isinstance(max_iter, str):
            _check_option('max_iter', max_iter, ('auto',), 'when str')
            if method == 'fastica':
                max_iter = 1000
            elif method in ['infomax', 'picard']:
                max_iter = 500
        fit_params.setdefault('max_iter', max_iter)
        self.max_iter = max_iter
        self.fit_params = fit_params

        self.exclude = []
        self.info = None
        self.method = method
        self.labels_ = dict()
        self.allow_ref_meg = allow_ref_meg

    def _get_infos_for_repr(self):
        @dataclass
        class _InfosForRepr:
            # XXX replace with Optional[Literal['raw data', 'epochs'] once we
            # drop support for Py 3.7
            fit_on: Optional[str]
            # XXX replace with fit_method: Literal['fastica', 'infomax',
            # 'extended-infomax', 'picard'] once we drop support for Py 3.7
            fit_method: str
            fit_n_iter: Optional[int]
            fit_n_samples: Optional[int]
            fit_n_components: Optional[int]
            fit_n_pca_components: Optional[int]
            fit_explained_variance: Optional[float]
            ch_types: List[str]
            excludes: List[str]

        if self.current_fit == 'unfitted':
            fit_on = None
        elif self.current_fit == 'raw':
            fit_on = 'raw data'
        else:
            fit_on = 'epochs'

        fit_method = self.method
        fit_n_iter = getattr(self, 'n_iter_', None)
        fit_n_samples = getattr(self, 'n_samples_', None)
        fit_n_components = getattr(self, 'n_components_', None)
        fit_n_pca_components = getattr(self, 'pca_components_', None)
        if fit_n_pca_components is not None:
            fit_n_pca_components = len(self.pca_components_)
        fit_explained_variance = getattr(self, 'pca_explained_variance_', None)
        if fit_explained_variance is not None:
            abs_vars = self.pca_explained_variance_
            rel_vars = abs_vars / abs_vars.sum()
            fit_explained_variance = rel_vars[:fit_n_components].sum()

        if self.info is not None:
            ch_types = [c for c in _DATA_CH_TYPES_SPLIT if c in self]
        else:
            ch_types = []

        if self.exclude:
            excludes = [self._ica_names[i] for i in self.exclude]
        else:
            excludes = []

        infos_for_repr = _InfosForRepr(
            fit_on=fit_on,
            fit_method=fit_method,
            fit_n_iter=fit_n_iter,
            fit_n_samples=fit_n_samples,
            fit_n_components=fit_n_components,
            fit_n_pca_components=fit_n_pca_components,
            fit_explained_variance=fit_explained_variance,
            ch_types=ch_types,
            excludes=excludes
        )
        return infos_for_repr

    def __repr__(self):
        """ICA fit information."""
        infos = self._get_infos_for_repr()

        s = (f'{infos.fit_on or "no"} decomposition, '
             f'method: {infos.fit_method}')

        if infos.fit_on is not None:
            s += (
                f' (fit in {infos.fit_n_iter} iterations on '
                f'{infos.fit_n_samples} samples), '
                f'{infos.fit_n_components} ICA components '
                f'explaining {round(infos.fit_explained_variance * 100, 1)} % '
                f'of variance '
                f'({infos.fit_n_pca_components} PCA components available), '
                f'channel types: {", ".join(infos.ch_types)}, '
                f'{len(infos.excludes) or "no"} sources marked for exclusion'
            )

        return f'<ICA | {s}>'

    def _repr_html_(self):
        infos = self._get_infos_for_repr()
        html = ica_template.substitute(
            fit_on=infos.fit_on,
            method=infos.fit_method,
            n_iter=infos.fit_n_iter,
            n_samples=infos.fit_n_samples,
            n_components=infos.fit_n_components,
            n_pca_components=infos.fit_n_pca_components,
            explained_variance=infos.fit_explained_variance,
            ch_types=infos.ch_types,
            excludes=infos.excludes
        )
        return html

    @verbose
    def fit(self, inst, picks=None, start=None, stop=None, decim=None,
            reject=None, flat=None, tstep=2.0, reject_by_annotation=True,
            verbose=None):
        """Run the ICA decomposition on raw data.

        Caveat! If supplying a noise covariance keep track of the projections
        available in the cov, the raw or the epochs object. For example,
        if you are interested in EOG or ECG artifacts, EOG and ECG projections
        should be temporally removed before fitting the ICA.

        Parameters
        ----------
        inst : instance of Raw or Epochs
            The data to be decomposed.
        %(picks_good_data_noref)s
            This selection remains throughout the initialized ICA solution.
        start, stop : int | float | None
            First and last sample to include. If float, data will be
            interpreted as time in seconds. If ``None``, data will be used from
            the first sample and to the last sample, respectively.

            .. note:: These parameters only have an effect if ``inst`` is
                      `~mne.io.Raw` data.
        decim : int | None
            Increment for selecting only each n-th sampling point. If ``None``,
            all samples  between ``start`` and ``stop`` (inclusive) are used.
        reject, flat : dict | None
            Rejection parameters based on peak-to-peak amplitude (PTP)
            in the continuous data. Signal periods exceeding the thresholds
            in ``reject`` or less than the thresholds in ``flat`` will be
            removed before fitting the ICA.

            .. note:: These parameters only have an effect if ``inst`` is
                      `~mne.io.Raw` data. For `~mne.Epochs`, perform PTP
                      rejection via :meth:`~mne.Epochs.drop_bad`.

            Valid keys are all channel types present in the data. Values must
            be integers or floats.

            If ``None``, no PTP-based rejection will be performed. Example::

                reject = dict(
                    grad=4000e-13, # T / m (gradiometers)
                    mag=4e-12, # T (magnetometers)
                    eeg=40e-6, # V (EEG channels)
                    eog=250e-6 # V (EOG channels)
                )
                flat = None  # no rejection based on flatness
        tstep : float
            Length of data chunks for artifact rejection in seconds.

            .. note:: This parameter only has an effect if ``inst`` is
                      `~mne.io.Raw` data.
        %(reject_by_annotation_raw)s

            .. versionadded:: 0.14.0
        %(verbose_meth)s

        Returns
        -------
        self : instance of ICA
            Returns the modified instance.
        """
        _validate_type(inst, (BaseRaw, BaseEpochs), 'inst', 'Raw or Epochs')

        if np.isclose(inst.info['highpass'], 0.):
            warn('The data has not been high-pass filtered. For good ICA '
                 'performance, it should be high-pass filtered (e.g., with a '
                 '1.0 Hz lower bound) before fitting ICA.')

        if isinstance(inst, BaseEpochs) and inst.baseline is not None:
            warn('The epochs you passed to ICA.fit() were baseline-corrected. '
                 'However, we suggest to fit ICA only on data that has been '
                 'high-pass filtered, but NOT baseline-corrected.')

        if not isinstance(inst, BaseRaw):
            ignored_params = [
                param_name for param_name, param_val in zip(
                    ('start', 'stop', 'reject', 'flat'),
                    (start, stop, reject, flat)
                )
                if param_val is not None
            ]
            if ignored_params:
                warn(f'The following parameters passed to ICA.fit() will be '
                     f'ignored, as they only affect raw data (and it appears '
                     f'you passed epochs): {", ".join(ignored_params)}')

        picks = _picks_to_idx(inst.info, picks, allow_empty=False,
                              with_ref_meg=self.allow_ref_meg)
        _check_for_unsupported_ica_channels(
            picks, inst.info, allow_ref_meg=self.allow_ref_meg)

        # Actually start fitting
        t_start = time()
        if self.current_fit != 'unfitted':
            self._reset()

        logger.info('Fitting ICA to data using %i channels '
                    '(please be patient, this may take a while)' % len(picks))

        # n_components could be float 0 < x < 1, but that's okay here
        if self.n_components is not None and self.n_components > len(picks):
            raise ValueError(
                f'ica.n_components ({self.n_components}) cannot '
                f'be greater than len(picks) ({len(picks)})')

        # filter out all the channels the raw wouldn't have initialized
        self.info = pick_info(inst.info, picks)

        if self.info['comps']:
            with self.info._unlock():
                self.info['comps'] = []
        self.ch_names = self.info['ch_names']

        if isinstance(inst, BaseRaw):
            self._fit_raw(inst, picks, start, stop, decim, reject, flat,
                          tstep, reject_by_annotation, verbose)
        else:
            assert isinstance(inst, BaseEpochs)
            self._fit_epochs(inst, picks, decim, verbose)

        # sort ICA components by explained variance
        var = _ica_explained_variance(self, inst)
        var_ord = var.argsort()[::-1]
        _sort_components(self, var_ord, copy=False)
        t_stop = time()
        logger.info("Fitting ICA took {:.1f}s.".format(t_stop - t_start))
        return self

    def _reset(self):
        """Aux method."""
        for key in ('pre_whitener_', 'unmixing_matrix_', 'mixing_matrix_',
                    'n_components_', 'n_samples_', 'pca_components_',
                    'pca_explained_variance_',
                    'pca_mean_', 'n_iter_', 'drop_inds_', 'reject_'):
            if hasattr(self, key):
                delattr(self, key)

    def _fit_raw(self, raw, picks, start, stop, decim, reject, flat, tstep,
                 reject_by_annotation, verbose):
        """Aux method."""
        start, stop = _check_start_stop(raw, start, stop)

        reject_by_annotation = 'omit' if reject_by_annotation else None
        # this will be a copy
        data = raw.get_data(picks, start, stop, reject_by_annotation)

        # this will be a view
        if decim is not None:
            data = data[:, ::decim]

        # this will make a copy
        if (reject is not None) or (flat is not None):
            self.reject_ = reject
            data, self.drop_inds_ = _reject_data_segments(data, reject, flat,
                                                          decim, self.info,
                                                          tstep)

        self.n_samples_ = data.shape[1]
        self._fit(data, 'raw')

        return self

    def _fit_epochs(self, epochs, picks, decim, verbose):
        """Aux method."""
        if epochs.events.size == 0:
            raise RuntimeError('Tried to fit ICA with epochs, but none were '
                               'found: epochs.events is "{}".'
                               .format(epochs.events))

        # this should be a copy (picks a list of int)
        data = epochs.get_data()[:, picks]
        # this will be a view
        if decim is not None:
            data = data[:, :, ::decim]

        self.n_samples_ = data.shape[0] * data.shape[2]

        # This will make at least one copy (one from hstack, maybe one
        # more from _pre_whiten)
        data = np.hstack(data)
        self._fit(data, 'epochs')

        return self

    def _compute_pre_whitener(self, data):
        """Aux function."""
        data = self._do_proj(data, log_suffix='(pre-whitener computation)')

        if self.noise_cov is None:
            # use standardization as whitener
            # Scale (z-score) the data by channel type
            info = self.info
            pre_whitener = np.empty([len(data), 1])
            for _, picks_ in _picks_by_type(info, ref_meg=False, exclude=[]):
                pre_whitener[picks_] = np.std(data[picks_])
            if _contains_ch_type(info, "ref_meg"):
                picks_ = pick_types(info, ref_meg=True, exclude=[])
                pre_whitener[picks_] = np.std(data[picks_])
            if _contains_ch_type(info, "eog"):
                picks_ = pick_types(info, eog=True, exclude=[])
                pre_whitener[picks_] = np.std(data[picks_])
        else:
            pre_whitener, _ = compute_whitener(
                self.noise_cov, self.info, picks=self.info.ch_names)
            assert data.shape[0] == pre_whitener.shape[1]
        self.pre_whitener_ = pre_whitener

    def _do_proj(self, data, log_suffix=''):
        if self.info is not None and self.info['projs']:
            proj, nproj, _ = make_projector(
                [p for p in self.info['projs'] if p['active']],
                self.info['ch_names'], include_active=True)
            if nproj:
                logger.info(
                    f'    Applying projection operator with {nproj} '
                    f'vector{_pl(nproj)}'
                    f'{" " if log_suffix else ""}{log_suffix}')
                if self.noise_cov is None:  # otherwise it's in pre_whitener_
                    data = proj @ data
        return data

    def _pre_whiten(self, data):
        data = self._do_proj(data, log_suffix='(pre-whitener application)')
        if self.noise_cov is None:
            data /= self.pre_whitener_
        else:
            data = self.pre_whitener_ @ data
        return data

    def _fit(self, data, fit_type):
        """Aux function."""
        random_state = check_random_state(self.random_state)
        n_channels, n_samples = data.shape
        self._compute_pre_whitener(data)
        data = self._pre_whiten(data)

        pca = _PCA(n_components=self._max_pca_components, whiten=True)
        data = pca.fit_transform(data.T)
        use_ev = pca.explained_variance_ratio_
        n_pca = self.n_pca_components
        if isinstance(n_pca, float):
            n_pca = int(_exp_var_ncomp(use_ev, n_pca)[0])
        elif n_pca is None:
            n_pca = len(use_ev)
        assert isinstance(n_pca, (int, np.int_))

        # If user passed a float, select the PCA components explaining the
        # given cumulative variance. This information will later be used to
        # only submit the corresponding parts of the data to ICA.
        if self.n_components is None:
            # None case: check if n_pca_components or 0.999999 yields smaller
            msg = 'Selecting by non-zero PCA components'
            self.n_components_ = min(
                n_pca, _exp_var_ncomp(use_ev, 0.999999)[0])
        elif isinstance(self.n_components, float):
            self.n_components_, ev = _exp_var_ncomp(use_ev, self.n_components)
            if self.n_components_ == 1:
                raise RuntimeError(
                    'One PCA component captures most of the '
                    f'explained variance ({100 * ev}%), your threshold '
                    'results in 1 component. You should select '
                    'a higher value.')
            msg = 'Selecting by explained variance'
        else:
            msg = 'Selecting by number'
            self.n_components_ = _ensure_int(self.n_components)
        # check to make sure something okay happened
        if self.n_components_ > n_pca:
            ev = np.cumsum(use_ev)
            ev /= ev[-1]
            evs = 100 * ev[[self.n_components_ - 1, n_pca - 1]]
            raise RuntimeError(
                f'n_components={self.n_components} requires '
                f'{self.n_components_} PCA values (EV={evs[0]:0.1f}%) but '
                f'n_pca_components ({self.n_pca_components}) results in '
                f'only {n_pca} components (EV={evs[1]:0.1f}%)')
        logger.info('%s: %s components' % (msg, self.n_components_))

        # the things to store for PCA
        self.pca_mean_ = pca.mean_
        self.pca_components_ = pca.components_
        self.pca_explained_variance_ = pca.explained_variance_
        del pca
        # update number of components
        self._update_ica_names()
        if self.n_pca_components is not None and \
                self.n_pca_components > len(self.pca_components_):
            raise ValueError(
                f'n_pca_components ({self.n_pca_components}) is greater than '
                f'the number of PCA components ({len(self.pca_components_)})')

        # take care of ICA
        sel = slice(0, self.n_components_)
        if self.method == 'fastica':
            from sklearn.decomposition import FastICA
            ica = FastICA(
                whiten=False, random_state=random_state, **self.fit_params)
            ica.fit(data[:, sel])
            self.unmixing_matrix_ = ica.components_
            self.n_iter_ = ica.n_iter_
        elif self.method in ('infomax', 'extended-infomax'):
            unmixing_matrix, n_iter = infomax(
                data[:, sel], random_state=random_state, return_n_iter=True,
                **self.fit_params)
            self.unmixing_matrix_ = unmixing_matrix
            self.n_iter_ = n_iter
            del unmixing_matrix, n_iter
        elif self.method == 'picard':
            from picard import picard
            _, W, _, n_iter = picard(
                data[:, sel].T, whiten=False, return_n_iter=True,
                random_state=random_state, **self.fit_params)
            self.unmixing_matrix_ = W
            self.n_iter_ = n_iter + 1  # picard() starts counting at 0
            del _, n_iter
        assert self.unmixing_matrix_.shape == (self.n_components_,) * 2
        norms = self.pca_explained_variance_
        stable = norms / norms[0] > 1e-6  # to be stable during pinv
        norms = norms[:self.n_components_]
        if not stable[self.n_components_ - 1]:
            max_int = np.where(stable)[0][-1] + 1
            warn(f'Using n_components={self.n_components} (resulting in '
                 f'n_components_={self.n_components_}) may lead to an '
                 f'unstable mixing matrix estimation because the ratio '
                 f'between the largest ({norms[0]:0.2g}) and smallest '
                 f'({norms[-1]:0.2g}) variances is too large (> 1e6); '
                 f'consider setting n_components=0.999999 or an '
                 f'integer <= {max_int}')
        norms = np.sqrt(norms)
        norms[norms == 0] = 1.
        self.unmixing_matrix_ /= norms  # whitening
        self._update_mixing_matrix()
        self.current_fit = fit_type

    def _update_mixing_matrix(self):
        from scipy import linalg
        self.mixing_matrix_ = linalg.pinv(self.unmixing_matrix_)

    def _update_ica_names(self):
        """Update ICA names when n_components_ is set."""
        self._ica_names = ['ICA%03d' % ii for ii in range(self.n_components_)]

    def _transform(self, data):
        """Compute sources from data (operates inplace)."""
        data = self._pre_whiten(data)
        if self.pca_mean_ is not None:
            data -= self.pca_mean_[:, None]

        # Apply unmixing
        pca_data = np.dot(self.unmixing_matrix_,
                          self.pca_components_[:self.n_components_])
        # Apply PCA
        sources = np.dot(pca_data, data)
        return sources

    def _transform_raw(self, raw, start, stop, reject_by_annotation=False):
        """Transform raw data."""
        if not hasattr(self, 'mixing_matrix_'):
            raise RuntimeError('No fit available. Please fit ICA.')
        start, stop = _check_start_stop(raw, start, stop)
        picks = self._get_picks(raw)
        reject = 'omit' if reject_by_annotation else None
        data = raw.get_data(picks, start, stop, reject)
        return self._transform(data)

    def _transform_epochs(self, epochs, concatenate):
        """Aux method."""
        if not hasattr(self, 'mixing_matrix_'):
            raise RuntimeError('No fit available. Please fit ICA.')
        picks = self._get_picks(epochs)
        data = np.hstack(epochs.get_data()[:, picks])
        sources = self._transform(data)
        if not concatenate:
            # Put the data back in 3D
            sources = np.array(np.split(sources, len(epochs.events), 1))
        return sources

    def _transform_evoked(self, evoked):
        """Aux method."""
        if not hasattr(self, 'mixing_matrix_'):
            raise RuntimeError('No fit available. Please fit ICA.')
        picks = self._get_picks(evoked)
        return self._transform(evoked.data[picks])

    def _get_picks(self, inst):
        """Pick logic for _transform method."""
        picks = _picks_to_idx(
            inst.info, self.ch_names, exclude=[], allow_empty=True)
        if len(picks) != len(self.ch_names):
            if isinstance(inst, BaseRaw):
                kind, do = 'Raw', "doesn't"
            elif isinstance(inst, BaseEpochs):
                kind, do = 'Epochs', "don't"
            elif isinstance(inst, Evoked):
                kind, do = 'Evoked', "doesn't"
            else:
                raise ValueError('Data input must be of Raw, Epochs or Evoked '
                                 'type')
            raise RuntimeError("%s %s match fitted data: %i channels "
                               "fitted but %i channels supplied. \nPlease "
                               "provide %s compatible with ica.ch_names"
                               % (kind, do, len(self.ch_names), len(picks),
                                  kind))
        return picks

    def get_components(self):
        """Get ICA topomap for components as numpy arrays.

        Returns
        -------
        components : array, shape (n_channels, n_components)
            The ICA components (maps).
        """
        return np.dot(self.mixing_matrix_[:, :self.n_components_].T,
                      self.pca_components_[:self.n_components_]).T

    def get_sources(self, inst, add_channels=None, start=None, stop=None):
        """Estimate sources given the unmixing matrix.

        This method will return the sources in the container format passed.
        Typical usecases:

        1. pass Raw object to use `raw.plot <mne.io.Raw.plot>` for ICA sources
        2. pass Epochs object to compute trial-based statistics in ICA space
        3. pass Evoked object to investigate time-locking in ICA space

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Object to compute sources from and to represent sources in.
        add_channels : None | list of str
            Additional channels  to be added. Useful to e.g. compare sources
            with some reference. Defaults to None.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, the entire data will be used.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, the entire data will be used.

        Returns
        -------
        sources : instance of Raw, Epochs or Evoked
            The ICA sources time series.
        """
        if isinstance(inst, BaseRaw):
            _check_compensation_grade(self.info, inst.info, 'ICA', 'Raw',
                                      ch_names=self.ch_names)
            sources = self._sources_as_raw(inst, add_channels, start, stop)
        elif isinstance(inst, BaseEpochs):
            _check_compensation_grade(self.info, inst.info, 'ICA', 'Epochs',
                                      ch_names=self.ch_names)
            sources = self._sources_as_epochs(inst, add_channels, False)
        elif isinstance(inst, Evoked):
            _check_compensation_grade(self.info, inst.info, 'ICA', 'Evoked',
                                      ch_names=self.ch_names)
            sources = self._sources_as_evoked(inst, add_channels)
        else:
            raise ValueError('Data input must be of Raw, Epochs or Evoked '
                             'type')
        return sources

    def _sources_as_raw(self, raw, add_channels, start, stop):
        """Aux method."""
        # merge copied instance and picked data with sources
        start, stop = _check_start_stop(raw, start, stop)
        data_ = self._transform_raw(raw, start=start, stop=stop)
        assert data_.shape[1] == stop - start
        if raw.preload:  # get data and temporarily delete
            data = raw._data
            del raw._data

        out = raw.copy()  # copy and reappend
        if raw.preload:
            raw._data = data

        # populate copied raw.
        if add_channels is not None and len(add_channels):
            picks = pick_channels(raw.ch_names, add_channels)
            data_ = np.concatenate([
                data_, raw.get_data(picks, start=start, stop=stop)])
        out._data = data_
        out._filenames = [None]
        out.preload = True
        out._first_samps[:] = [out.first_samp + start]
        out._last_samps[:] = [out.first_samp + data_.shape[1] - 1]
        out._projector = None
        self._export_info(out.info, raw, add_channels)

        return out

    def _sources_as_epochs(self, epochs, add_channels, concatenate):
        """Aux method."""
        out = epochs.copy()
        sources = self._transform_epochs(epochs, concatenate)
        if add_channels is not None:
            picks = [epochs.ch_names.index(k) for k in add_channels]
        else:
            picks = []
        out._data = np.concatenate([sources, epochs.get_data()[:, picks]],
                                   axis=1) if len(picks) > 0 else sources

        self._export_info(out.info, epochs, add_channels)
        out.preload = True
        out._raw = None
        out._projector = None

        return out

    def _sources_as_evoked(self, evoked, add_channels):
        """Aux method."""
        if add_channels is not None:
            picks = [evoked.ch_names.index(k) for k in add_channels]
        else:
            picks = []

        sources = self._transform_evoked(evoked)
        if len(picks) > 1:
            data = np.r_[sources, evoked.data[picks]]
        else:
            data = sources
        out = evoked.copy()
        out.data = data
        self._export_info(out.info, evoked, add_channels)

        return out

    def _export_info(self, info, container, add_channels):
        """Aux method."""
        # set channel names and info
        ch_names = []
        ch_info = []
        for ii, name in enumerate(self._ica_names):
            ch_names.append(name)
            ch_info.append(dict(
                ch_name=name, cal=1, logno=ii + 1,
                coil_type=FIFF.FIFFV_COIL_NONE,
                kind=FIFF.FIFFV_MISC_CH,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                unit=FIFF.FIFF_UNIT_NONE,
                loc=np.zeros(12, dtype='f4'),
                range=1.0, scanno=ii + 1, unit_mul=0))

        if add_channels is not None:
            # re-append additionally picked ch_names
            ch_names += add_channels
            # re-append additionally picked ch_info
            ch_info += [k for k in container.info['chs'] if k['ch_name'] in
                        add_channels]
        with info._unlock(update_redundant=True, check_after=True):
            info['chs'] = ch_info
            info['bads'] = [ch_names[k] for k in self.exclude]
            info['projs'] = []  # make sure projections are removed.

    @verbose
    def score_sources(self, inst, target=None, score_func='pearsonr',
                      start=None, stop=None, l_freq=None, h_freq=None,
                      reject_by_annotation=True, verbose=None):
        """Assign score to components based on statistic or metric.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            The object to reconstruct the sources from.
        target : array-like | str | None
            Signal to which the sources shall be compared. It has to be of
            the same shape as the sources. If str, a routine will try to find
            a matching channel name. If None, a score
            function expecting only one input-array argument must be used,
            for instance, scipy.stats.skew (default).
        score_func : callable | str
            Callable taking as arguments either two input arrays
            (e.g. Pearson correlation) or one input
            array (e. g. skewness) and returns a float. For convenience the
            most common score_funcs are available via string labels:
            Currently, all distance metrics from scipy.spatial and All
            functions from scipy.stats taking compatible input arguments are
            supported. These function have been modified to support iteration
            over the rows of a 2D array.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
        l_freq : float
            Low pass frequency.
        h_freq : float
            High pass frequency.
        %(reject_by_annotation_all)s

            .. versionadded:: 0.14.0
        %(verbose_meth)s

        Returns
        -------
        scores : ndarray
            Scores for each source as returned from score_func.
        """
        if isinstance(inst, BaseRaw):
            _check_compensation_grade(self.info, inst.info, 'ICA', 'Raw',
                                      ch_names=self.ch_names)
            sources = self._transform_raw(inst, start, stop,
                                          reject_by_annotation)
        elif isinstance(inst, BaseEpochs):
            _check_compensation_grade(self.info, inst.info, 'ICA', 'Epochs',
                                      ch_names=self.ch_names)
            sources = self._transform_epochs(inst, concatenate=True)
        elif isinstance(inst, Evoked):
            _check_compensation_grade(self.info, inst.info, 'ICA', 'Evoked',
                                      ch_names=self.ch_names)
            sources = self._transform_evoked(inst)
        else:
            raise ValueError('Data input must be of Raw, Epochs or Evoked '
                             'type')

        if target is not None:  # we can have univariate metrics without target
            target = self._check_target(target, inst, start, stop,
                                        reject_by_annotation)

            if sources.shape[-1] != target.shape[-1]:
                raise ValueError('Sources and target do not have the same '
                                 'number of time slices.')
            # auto target selection
            if isinstance(inst, BaseRaw):
                # We pass inst, not self, because the sfreq of the data we
                # use for scoring components can be different:
                sources, target = _band_pass_filter(inst, sources, target,
                                                    l_freq, h_freq)

        scores = _find_sources(sources, target, score_func)

        return scores

    def _check_target(self, target, inst, start, stop,
                      reject_by_annotation=False):
        """Aux Method."""
        if isinstance(inst, BaseRaw):
            reject_by_annotation = 'omit' if reject_by_annotation else None
            start, stop = _check_start_stop(inst, start, stop)
            if hasattr(target, 'ndim'):
                if target.ndim < 2:
                    target = target.reshape(1, target.shape[-1])
            if isinstance(target, str):
                pick = _get_target_ch(inst, target)
                target = inst.get_data(pick, start, stop, reject_by_annotation)

        elif isinstance(inst, BaseEpochs):
            if isinstance(target, str):
                pick = _get_target_ch(inst, target)
                target = inst.get_data()[:, pick]

            if hasattr(target, 'ndim'):
                if target.ndim == 3 and min(target.shape) == 1:
                    target = target.ravel()

        elif isinstance(inst, Evoked):
            if isinstance(target, str):
                pick = _get_target_ch(inst, target)
                target = inst.data[pick]

        return target

    def _find_bads_ch(self, inst, chs, threshold=3.0, start=None,
                      stop=None, l_freq=None, h_freq=None,
                      reject_by_annotation=True, prefix='chs',
                      measure='zscore'):
        """Compute ExG/ref components.

        See find_bads_ecg, find_bads_eog, and find_bads_ref for details.
        """
        scores, idx = [], []
        # some magic we need inevitably ...
        # get targets before equalizing
        targets = [self._check_target(
            ch, inst, start, stop, reject_by_annotation) for ch in chs]
        # assign names, if targets are arrays instead of strings
        target_names = []
        for ch in chs:
            if not isinstance(ch, str):
                if prefix == "ecg":
                    target_names.append('ECG-MAG')
                else:
                    target_names.append(prefix)
            else:
                target_names.append(ch)

        for ii, (ch, target) in enumerate(zip(target_names, targets)):
            scores += [self.score_sources(
                inst, target=target, score_func='pearsonr', start=start,
                stop=stop, l_freq=l_freq, h_freq=h_freq,
                reject_by_annotation=reject_by_annotation)]
            # pick last scores
            if measure == "zscore":
                this_idx = _find_outliers(scores[-1], threshold=threshold)
            elif measure == "correlation":
                this_idx = np.where(abs(scores[-1]) > threshold)[0]
            else:
                raise ValueError("Unknown measure {}".format(measure))
            idx += [this_idx]
            self.labels_['%s/%i/' % (prefix, ii) + ch] = list(this_idx)

        # remove duplicates but keep order by score, even across multiple
        # ref channels
        scores_ = np.concatenate([scores[ii][inds]
                                  for ii, inds in enumerate(idx)])
        idx_ = np.concatenate(idx)[np.abs(scores_).argsort()[::-1]]

        idx_unique = list(np.unique(idx_))
        idx = []
        for i in idx_:
            if i in idx_unique:
                idx.append(i)
                idx_unique.remove(i)
        if len(scores) == 1:
            scores = scores[0]
        labels = list(idx)

        return labels, scores

    def _get_ctps_threshold(self, pk_threshold=20):
        """Automatically decide the threshold of Kuiper index for CTPS method.

        This function finds the threshold of Kuiper index based on the
        threshold of pk. Kuiper statistic that minimizes the difference between
        pk and the pk threshold (defaults to 20 :footcite:`DammersEtAl2008`)
        is returned. It is assumed that the data are appropriately filtered and
        bad data are rejected at least based on peak-to-peak amplitude
        when/before running the ICA decomposition on data.

        References
        ----------
        .. footbibliography::
        """
        N = self.info['sfreq']
        Vs = np.arange(1, 100) / 100
        C = math.sqrt(N) + 0.155 + 0.24 / math.sqrt(N)
        # in formula (13), when k gets large, only k=1 matters for the
        # summation. k*V*C thus becomes V*C
        Pks = 2 * (4 * (Vs * C)**2 - 1) * (np.exp(-2 * (Vs * C)**2))
        # NOTE: the threshold of pk is transformed to Pk for comparison
        # pk = -log10(Pk)
        return Vs[np.argmin(np.abs(Pks - 10**(-pk_threshold)))]

    @verbose
    def find_bads_ecg(self, inst, ch_name=None, threshold='auto', start=None,
                      stop=None, l_freq=8, h_freq=16, method='ctps',
                      reject_by_annotation=True, measure='zscore',
                      verbose=None):
        """Detect ECG related components.

        Cross-trial phase statistics :footcite:`DammersEtAl2008` or Pearson
        correlation can be used for detection.

        .. note:: If no ECG channel is available, routine attempts to create
                  an artificial ECG based on cross-channel averaging.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Object to compute sources from.
        ch_name : str
            The name of the channel to use for ECG peak detection.
            The argument is mandatory if the dataset contains no ECG
            channels.
        threshold : float | 'auto'
            Value above which a feature is classified as outlier. See Notes.

            .. versionchanged:: 0.21
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
            When working with Epochs or Evoked objects, must be float or None.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
            When working with Epochs or Evoked objects, must be float or None.
        l_freq : float
            Low pass frequency.
        h_freq : float
            High pass frequency.
        method : 'ctps' | 'correlation'
            The method used for detection. If ``'ctps'``, cross-trial phase
            statistics :footcite:`DammersEtAl2008` are used to detect
            ECG-related components. See Notes.
        %(reject_by_annotation_all)s

            .. versionadded:: 0.14.0
        %(measure)s
        %(verbose_meth)s

        Returns
        -------
        ecg_idx : list of int
            The indices of ECG-related components.
        scores : np.ndarray of float, shape (``n_components_``)
            If method is 'ctps', the normalized Kuiper index scores. If method
            is 'correlation', the correlation scores.

        See Also
        --------
        find_bads_eog, find_bads_ref

        Notes
        -----
        The ``threshold``, ``method``, and ``measure`` parameters interact in
        the following ways:

        - If ``method='ctps'``, ``threshold`` refers to the significance value
          of a Kuiper statistic, and ``threshold='auto'`` will compute the
          threshold automatically based on the sampling frequency.
        - If ``method='correlation'`` and ``measure='correlation'``,
          ``threshold`` refers to the Pearson correlation value, and
          ``threshold='auto'`` sets the threshold to 0.9.
        - If ``method='correlation'`` and ``measure='zscore'``, ``threshold``
          refers to the z-score value (i.e., standard deviations) used in the
          iterative z-scoring method, and ``threshold='auto'`` sets the
          threshold to 3.0.

        References
        ----------
        .. footbibliography::
        """
        _validate_type(threshold, (str, 'numeric'), 'threshold')
        if isinstance(threshold, str):
            _check_option('threshold', threshold, ('auto',), extra='when str')
        _validate_type(method, str, 'method')
        _check_option('method', method, ('ctps', 'correlation'))
        _validate_type(measure, str, 'measure')
        _check_option('measure', measure, ('zscore', 'correlation'))

        idx_ecg = _get_ecg_channel_index(ch_name, inst)

        if idx_ecg is None:
            ecg, times = _make_ecg(inst, start, stop,
                                   reject_by_annotation=reject_by_annotation)
        else:
            ecg = inst.ch_names[idx_ecg]

        if method == 'ctps':
            if threshold == 'auto':
                threshold = self._get_ctps_threshold()
                logger.info('Using threshold: %.2f for CTPS ECG detection'
                            % threshold)
            if isinstance(inst, BaseRaw):
                sources = self.get_sources(create_ecg_epochs(
                    inst, ch_name, l_freq=l_freq, h_freq=h_freq,
                    keep_ecg=False,
                    reject_by_annotation=reject_by_annotation)).get_data()

                if sources.shape[0] == 0:
                    warn('No ECG activity detected. Consider changing '
                         'the input parameters.')
            elif isinstance(inst, BaseEpochs):
                sources = self.get_sources(inst).get_data()
            else:
                raise ValueError('With `ctps` only Raw and Epochs input is '
                                 'supported')
            _, p_vals, _ = ctps(sources)
            scores = p_vals.max(-1)
            ecg_idx = np.where(scores >= threshold)[0]
            # sort indices by scores
            ecg_idx = ecg_idx[np.abs(scores[ecg_idx]).argsort()[::-1]]

            self.labels_['ecg'] = list(ecg_idx)
            if ch_name is None:
                ch_name = 'ECG-MAG'
            self.labels_['ecg/%s' % ch_name] = list(ecg_idx)
        elif method == 'correlation':
            if threshold == 'auto' and measure == 'zscore':
                threshold = 3.0
            elif threshold == 'auto' and measure == 'correlation':
                threshold = 0.9
            self.labels_['ecg'], scores = self._find_bads_ch(
                inst, [ecg], threshold=threshold, start=start, stop=stop,
                l_freq=l_freq, h_freq=h_freq, prefix="ecg",
                reject_by_annotation=reject_by_annotation, measure=measure)

        return self.labels_['ecg'], scores

    @verbose
    def find_bads_ref(self, inst, ch_name=None, threshold=3.0, start=None,
                      stop=None, l_freq=None, h_freq=None,
                      reject_by_annotation=True, method='together',
                      measure="zscore", verbose=None):
        """Detect MEG reference related components using correlation.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Object to compute sources from. Should contain at least one channel
            i.e. component derived from MEG reference channels.
        ch_name : list of str
            Which MEG reference components to use. If None, then all channels
            that begin with REF_ICA.
        threshold : float | str
            Value above which a feature is classified as outlier.

            - If ``measure`` is ``'zscore'``, defines the threshold on the
              z-score used in the iterative z-scoring method.
            - If ``measure`` is ``'correlation'``, defines the absolute
              threshold on the correlation between 0 and 1.
            - If ``'auto'``, defaults to 3.0 if ``measure`` is ``'zscore'`` and
              0.9 if ``measure`` is ``'correlation'``.

             .. warning::
                 If ``method`` is ``'together'``, the iterative z-score method
                 is always used.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
        l_freq : float
            Low pass frequency.
        h_freq : float
            High pass frequency.
        %(reject_by_annotation_all)s
        method : 'together' | 'separate'
            Method to use to identify reference channel related components.
            Defaults to ``'together'``. See notes.

            .. versionadded:: 0.21
        %(measure)s
        %(verbose_meth)s

        Returns
        -------
        ref_idx : list of int
            The indices of MEG reference related components, sorted by score.
        scores : np.ndarray of float, shape (``n_components_``) | list of array
            The correlation scores.

        See Also
        --------
        find_bads_ecg, find_bads_eog

        Notes
        -----
        ICA decomposition on MEG reference channels is used to assess external
        magnetic noise and remove it from the MEG. Two methods are supported:

        With the ``'together'`` method, only one ICA fit is used, which
        encompasses both MEG and reference channels together. Components which
        have particularly strong weights on the reference channels may be
        thresholded and marked for removal.

        With ``'separate'`` selected components from a separate ICA
        decomposition on the reference channels are used as a ground truth for
        identifying bad components in an ICA fit done on MEG channels only. The
        logic here is similar to an EOG/ECG, with reference components
        replacing the EOG/ECG channels. Recommended procedure is to perform ICA
        separately on reference channels, extract them using
        :meth:`~mne.preprocessing.ICA.get_sources`, and then append them to the
        inst using :meth:`~mne.io.Raw.add_channels`, preferably with the prefix
        ``REF_ICA`` so that they can be automatically detected.

        With ``'together'``, thresholding is based on adaptative z-scoring.

        With ``'separate'``:

        - If ``measure`` is ``'zscore'``, thresholding is based on adaptative
          z-scoring.
        - If ``measure`` is ``'correlation'``, threshold defines the absolute
          threshold on the correlation between 0 and 1.

        Validation and further documentation for this technique can be found
        in :footcite:`HannaEtAl2020`.

        .. versionadded:: 0.18

        References
        ----------
        .. footbibliography::
        """
        _validate_type(threshold, (str, 'numeric'), 'threshold')
        if isinstance(threshold, str):
            _check_option('threshold', threshold, ('auto',), extra='when str')
        _validate_type(method, str, 'method')
        _check_option('method', method, ('together', 'separate'))
        _validate_type(measure, str, 'measure')
        _check_option('measure', measure, ('zscore', 'correlation'))

        if method == "separate":
            if threshold == 'auto' and measure == 'zscore':
                threshold = 3.0
            elif threshold == 'auto' and measure == 'correlation':
                threshold = 0.9

            if not ch_name:
                inds = pick_channels_regexp(inst.ch_names, 'REF_ICA*')
            else:
                inds = pick_channels(inst.ch_names, ch_name)
            # regexp returns list, pick_channels returns numpy
            inds = list(inds)
            if not inds:
                raise ValueError('No valid channels available.')
            ref_chs = [inst.ch_names[k] for k in inds]

            self.labels_['ref_meg'], scores = self._find_bads_ch(
                inst, ref_chs, threshold=threshold, start=start, stop=stop,
                l_freq=l_freq, h_freq=h_freq, prefix='ref_meg',
                reject_by_annotation=reject_by_annotation,
                measure=measure)
        elif method == 'together':
            if threshold == 'auto':
                threshold = 3.0
            if measure != 'zscore':
                logger.info(
                    "With method 'together', only 'zscore' measure is"
                    f"supported. Using 'zscore' instead of '{measure}'.")

            meg_picks = pick_types(self.info, meg=True, ref_meg=False)
            ref_picks = pick_types(self.info, meg=False, ref_meg=True)
            if not any(meg_picks) or not any(ref_picks):
                raise ValueError('ICA solution must contain both reference and'
                                 ' MEG channels.')
            weights = self.get_components()
            # take norm of component weights on reference channels for each
            # component, divide them by the norm on the standard channels,
            # log transform to approximate normal distribution
            normrats = np.linalg.norm(weights[ref_picks], axis=0) \
                / np.linalg.norm(weights[meg_picks], axis=0)
            scores = np.log(normrats)
            self.labels_['ref_meg'] = list(_find_outliers(scores,
                                           threshold=threshold,
                                           tail=1))

        return self.labels_['ref_meg'], scores

    @verbose
    def find_bads_eog(self, inst, ch_name=None, threshold=3.0, start=None,
                      stop=None, l_freq=1, h_freq=10,
                      reject_by_annotation=True, measure='zscore',
                      verbose=None):
        """Detect EOG related components using correlation.

        Detection is based on Pearson correlation between the
        filtered data and the filtered EOG channel.
        Thresholding is based on adaptive z-scoring. The above threshold
        components will be masked and the z-score will be recomputed
        until no supra-threshold component remains.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Object to compute sources from.
        ch_name : str
            The name of the channel to use for EOG peak detection.
            The argument is mandatory if the dataset contains no EOG
            channels.
        threshold : float | str
            Value above which a feature is classified as outlier.

            - If ``measure`` is ``'zscore'``, defines the threshold on the
              z-score used in the iterative z-scoring method.
            - If ``measure`` is ``'correlation'``, defines the absolute
              threshold on the correlation between 0 and 1.
            - If ``'auto'``, defaults to 3.0 if ``measure`` is ``'zscore'`` and
              0.9 if ``measure`` is ``'correlation'``.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
        l_freq : float
            Low pass frequency.
        h_freq : float
            High pass frequency.
        %(reject_by_annotation_all)s

            .. versionadded:: 0.14.0
        %(measure)s
        %(verbose_meth)s

        Returns
        -------
        eog_idx : list of int
            The indices of EOG related components, sorted by score.
        scores : np.ndarray of float, shape (``n_components_``) | list of array
            The correlation scores.

        See Also
        --------
        find_bads_ecg, find_bads_ref
        """
        _validate_type(threshold, (str, 'numeric'), 'threshold')
        if isinstance(threshold, str):
            _check_option('threshold', threshold, ('auto',), extra='when str')
        _validate_type(measure, str, 'measure')
        _check_option('measure', measure, ('zscore', 'correlation'))

        eog_inds = _get_eog_channel_index(ch_name, inst)
        eog_chs = [inst.ch_names[k] for k in eog_inds]

        if threshold == 'auto' and measure == 'zscore':
            threshold = 3.0
        elif threshold == 'auto' and measure == 'correlation':
            threshold = 0.9

        self.labels_['eog'], scores = self._find_bads_ch(
            inst, eog_chs, threshold=threshold, start=start, stop=stop,
            l_freq=l_freq, h_freq=h_freq, prefix="eog",
            reject_by_annotation=reject_by_annotation, measure=measure)
        return self.labels_['eog'], scores

    @verbose
    def apply(self, inst, include=None, exclude=None, n_pca_components=None,
              start=None, stop=None, verbose=None):
        """Remove selected components from the signal.

        Given the unmixing matrix, transform the data,
        zero out all excluded components, and inverse-transform the data.
        This procedure will reconstruct M/EEG signals from which
        the dynamics described by the excluded components is subtracted.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            The data to be processed (i.e., cleaned). It will be modified
            in-place.
        include : array_like of int
            The indices referring to columns in the ummixing matrix. The
            components to be kept.
        exclude : array_like of int
            The indices referring to columns in the ummixing matrix. The
            components to be zeroed out.
        %(n_pca_components_apply)s
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
        %(verbose_meth)s

        Returns
        -------
        out : instance of Raw, Epochs or Evoked
            The processed data.

        Notes
        -----
        .. note:: Applying ICA may introduce a DC shift. If you pass
                  baseline-corrected `~mne.Epochs` or `~mne.Evoked` data,
                  the baseline period of the cleaned data may not be of
                  zero mean anymore. If you require baseline-corrected
                  data, apply baseline correction again after cleaning
                  via ICA. A warning will be emitted to remind you of this
                  fact if you pass baseline-corrected data.

        .. versionchanged:: 0.23
            Warn if instance was baseline-corrected.
        """
        _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), 'inst',
                       'Raw, Epochs, or Evoked')
        kwargs = dict(include=include, exclude=exclude,
                      n_pca_components=n_pca_components)
        if isinstance(inst, BaseRaw):
            kind, meth = 'Raw', self._apply_raw
            kwargs.update(raw=inst, start=start, stop=stop)
        elif isinstance(inst, BaseEpochs):
            kind, meth = 'Epochs', self._apply_epochs
            kwargs.update(epochs=inst)
        else:  # isinstance(inst, Evoked):
            kind, meth = 'Evoked', self._apply_evoked
            kwargs.update(evoked=inst)
        _check_compensation_grade(self.info, inst.info, 'ICA', kind,
                                  ch_names=self.ch_names)

        if isinstance(inst, (BaseEpochs, Evoked)):
            if getattr(inst, 'baseline', None) is not None:
                warn('The data you passed to ICA.apply() was '
                     'baseline-corrected. Please note that ICA can introduce '
                     'DC shifts, therefore you may wish to consider '
                     'baseline-correcting the cleaned data again.')

        logger.info(f'Applying ICA to {kind} instance')
        return meth(**kwargs)

    def _check_exclude(self, exclude):
        if exclude is None:
            return list(set(self.exclude))
        else:
            # Allow both self.exclude and exclude to be array-like:
            return list(set(self.exclude).union(set(exclude)))

    def _apply_raw(self, raw, include, exclude, n_pca_components, start, stop):
        """Aux method."""
        _check_preload(raw, "ica.apply")

        start, stop = _check_start_stop(raw, start, stop)

        picks = pick_types(raw.info, meg=False, include=self.ch_names,
                           exclude='bads', ref_meg=False)

        data = raw[picks, start:stop][0]
        data = self._pick_sources(data, include, exclude, n_pca_components)

        raw[picks, start:stop] = data
        return raw

    def _apply_epochs(self, epochs, include, exclude, n_pca_components):
        """Aux method."""
        _check_preload(epochs, "ica.apply")

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

        data = np.hstack(epochs.get_data(picks))
        data = self._pick_sources(data, include, exclude, n_pca_components)

        # restore epochs, channels, tsl order
        epochs._data[:, picks] = np.array(
            np.split(data, len(epochs.events), 1))
        epochs.preload = True

        return epochs

    def _apply_evoked(self, evoked, include, exclude, n_pca_components):
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
        data = self._pick_sources(data, include, exclude, n_pca_components)

        # restore evoked
        evoked.data[picks] = data

        return evoked

    def _pick_sources(self, data, include, exclude, n_pca_components):
        """Aux function."""
        if n_pca_components is None:
            n_pca_components = self.n_pca_components
        data = self._pre_whiten(data)
        exclude = self._check_exclude(exclude)
        _n_pca_comp = self._check_n_pca_components(n_pca_components)
        n_ch, _ = data.shape

        max_pca_components = self.pca_components_.shape[0]
        if not self.n_components_ <= _n_pca_comp <= max_pca_components:
            raise ValueError(
                f'n_pca_components ({_n_pca_comp}) must be >= '
                f'n_components_ ({self.n_components_}) and <= '
                'the total number of PCA components '
                f'({max_pca_components}).')

        logger.info(f'    Transforming to ICA space ({self.n_components_} '
                    f'component{_pl(self.n_components_)})')

        # Apply first PCA
        if self.pca_mean_ is not None:
            data -= self.pca_mean_[:, None]

        sel_keep = np.arange(self.n_components_)
        if include not in (None, []):
            sel_keep = np.unique(include)
        elif exclude not in (None, []):
            sel_keep = np.setdiff1d(np.arange(self.n_components_), exclude)

        n_zero = self.n_components_ - len(sel_keep)
        logger.info(f'    Zeroing out {n_zero} ICA component{_pl(n_zero)}')

        # Mixing and unmixing should both be shape (self.n_components_, 2),
        # and we need to put these into the upper left part of larger mixing
        # and unmixing matrices of shape (n_ch, _n_pca_comp)
        pca_components = self.pca_components_[:_n_pca_comp]
        assert pca_components.shape == (_n_pca_comp, n_ch)
        assert self.unmixing_matrix_.shape == \
            self.mixing_matrix_.shape == \
            (self.n_components_,) * 2
        unmixing = np.eye(_n_pca_comp)
        unmixing[:self.n_components_, :self.n_components_] = \
            self.unmixing_matrix_
        unmixing = np.dot(unmixing, pca_components)

        logger.info(f'    Projecting back using {_n_pca_comp} '
                    f'PCA component{_pl(_n_pca_comp)}')
        mixing = np.eye(_n_pca_comp)
        mixing[:self.n_components_, :self.n_components_] = \
            self.mixing_matrix_
        mixing = pca_components.T @ mixing
        assert mixing.shape == unmixing.shape[::-1] == (n_ch, _n_pca_comp)

        # keep requested components plus residuals (if any)
        sel_keep = np.concatenate(
            (sel_keep, np.arange(self.n_components_, _n_pca_comp)))
        proj_mat = np.dot(mixing[:, sel_keep], unmixing[sel_keep, :])
        data = np.dot(proj_mat, data)
        assert proj_mat.shape == (n_ch,) * 2

        if self.pca_mean_ is not None:
            data += self.pca_mean_[:, None]

        # restore scaling
        if self.noise_cov is None:  # revert standardization
            data *= self.pre_whitener_
        else:
            data = np.linalg.pinv(self.pre_whitener_, rcond=1e-14) @ data

        return data

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Store ICA solution into a fiff file.

        Parameters
        ----------
        fname : str
            The absolute path of the file name to save the ICA solution into.
            The file name should end with -ica.fif or -ica.fif.gz.
        %(overwrite)s

            .. versionadded:: 1.0
        %(verbose_meth)s

        Returns
        -------
        ica : instance of ICA
            The object.

        See Also
        --------
        read_ica
        """
        if self.current_fit == 'unfitted':
            raise RuntimeError('No fit available. Please first fit ICA')

        check_fname(fname, 'ICA', ('-ica.fif', '-ica.fif.gz',
                                   '_ica.fif', '_ica.fif.gz'))
        fname = _check_fname(fname, overwrite=overwrite)

        logger.info('Writing ICA solution to %s...' % fname)
        fid = start_file(fname)

        try:
            _write_ica(fid, self)
            end_file(fid)
        except Exception:
            end_file(fid)
            os.remove(fname)
            raise

        return self

    def copy(self):
        """Copy the ICA object.

        Returns
        -------
        ica : instance of ICA
            The copied object.
        """
        return deepcopy(self)

    @copy_function_doc_to_method_doc(plot_ica_components)
    def plot_components(self, picks=None, ch_type=None, res=64,
                        vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                        colorbar=False, title=None, show=True, outlines='head',
                        contours=6, image_interp='bilinear',
                        inst=None, plot_std=True, topomap_args=None,
                        image_args=None, psd_args=None, reject='auto',
                        sphere=None, verbose=None):
        return plot_ica_components(self, picks=picks, ch_type=ch_type,
                                   res=res, vmin=vmin,
                                   vmax=vmax, cmap=cmap, sensors=sensors,
                                   colorbar=colorbar, title=title, show=show,
                                   outlines=outlines, contours=contours,
                                   image_interp=image_interp,
                                   inst=inst, plot_std=plot_std,
                                   topomap_args=topomap_args,
                                   image_args=image_args, psd_args=psd_args,
                                   reject=reject, sphere=sphere,
                                   verbose=verbose)

    @copy_function_doc_to_method_doc(plot_ica_properties)
    def plot_properties(self, inst, picks=None, axes=None, dB=True,
                        plot_std=True, topomap_args=None, image_args=None,
                        psd_args=None, figsize=None, show=True, reject='auto',
                        reject_by_annotation=True, *, verbose=None):
        return plot_ica_properties(self, inst, picks=picks, axes=axes,
                                   dB=dB, plot_std=plot_std,
                                   topomap_args=topomap_args,
                                   image_args=image_args, psd_args=psd_args,
                                   figsize=figsize, show=show, reject=reject,
                                   reject_by_annotation=reject_by_annotation,
                                   verbose=verbose)

    @copy_function_doc_to_method_doc(plot_ica_sources)
    def plot_sources(self, inst, picks=None, start=None,
                     stop=None, title=None, show=True, block=False,
                     show_first_samp=False, show_scrollbars=True,
                     time_format='float'):
        return plot_ica_sources(self, inst=inst, picks=picks,
                                start=start, stop=stop, title=title, show=show,
                                block=block, show_first_samp=show_first_samp,
                                show_scrollbars=show_scrollbars,
                                time_format=time_format)

    @copy_function_doc_to_method_doc(plot_ica_scores)
    def plot_scores(self, scores, exclude=None, labels=None, axhline=None,
                    title='ICA component scores', figsize=None, n_cols=None,
                    show=True):
        return plot_ica_scores(
            ica=self, scores=scores, exclude=exclude, labels=labels,
            axhline=axhline, title=title, figsize=figsize, n_cols=n_cols,
            show=show)

    @copy_function_doc_to_method_doc(plot_ica_overlay)
    def plot_overlay(self, inst, exclude=None, picks=None, start=None,
                     stop=None, title=None, show=True, n_pca_components=None):
        return plot_ica_overlay(self, inst=inst, exclude=exclude, picks=picks,
                                start=start, stop=stop, title=title, show=show,
                                n_pca_components=n_pca_components)

    @deprecated(extra='Use ICA.find_bads_eog and ICA.find_bads_ecg instead.')
    def detect_artifacts(self, raw, start_find=None, stop_find=None,
                         ecg_ch=None, ecg_score_func='pearsonr',
                         ecg_criterion=0.1, eog_ch=None,
                         eog_score_func='pearsonr',
                         eog_criterion=0.1, skew_criterion=0,
                         kurt_criterion=0, var_criterion=-1,
                         add_nodes=None):
        """Run ICA artifacts detection workflow.

        Note. This is still experimental and will most likely change over
        the next releases. For maximum control use the workflow exposed in
        the examples.

        Hints and caveats:
        - It is highly recommended to bandpass filter ECG and EOG
        data and pass them instead of the channel names as ecg_ch and eog_ch
        arguments.
        - please check your results. Detection by kurtosis and variance
        may be powerful but misclassification of brain signals as
        noise cannot be precluded.
        - Consider using shorter times for start_find and stop_find than
        for start and stop. It can save you much time.

        Example invocation (taking advantage of the defaults)::

            ica.detect_artifacts(ecg_channel='MEG 1531', eog_channel='EOG 061')

        Parameters
        ----------
        raw : instance of Raw
            Raw object to draw sources from. No components are actually removed
            here, i.e. ica is not applied to raw in this function. Use
            `ica.apply() <ICA.apply>` for this after inspection of the
            identified components.
        start_find : int | float | None
            First sample to include for artifact search. If float, data will be
            interpreted as time in seconds. If None, data will be used from the
            first sample.
        stop_find : int | float | None
            Last sample to not include for artifact search. If float, data will
            be interpreted as time in seconds. If None, data will be used to
            the last sample.
        ecg_ch : str | ndarray | None
            The ``target`` argument passed to ica.find_sources_raw. Either the
            name of the ECG channel or the ECG time series. If None, this step
            will be skipped.
        ecg_score_func : str | callable
            The ``score_func`` argument passed to ica.find_sources_raw. Either
            the name of function supported by ICA or a custom function.
        ecg_criterion : float | int | list-like | slice
            The indices of the sorted ecg scores. If float, sources with
            absolute scores greater than the criterion will be dropped. Else,
            the absolute scores sorted in descending order will be indexed
            accordingly. E.g. range(2) would return the two sources with the
            highest absolute score. If None, this step will be skipped.
        eog_ch : list | str | ndarray | None
            The ``target`` argument or the list of target arguments
            subsequently passed to ica.find_sources_raw. Either the name of the
            vertical EOG channel or the corresponding EOG time series. If None,
            this step will be skipped.
        eog_score_func : str | callable
            The ``score_func`` argument passed to ica.find_sources_raw. Either
            the name of function supported by ICA or a custom function.
        eog_criterion : float | int | list-like | slice
            The indices of the sorted eog scores. If float, sources with
            absolute scores greater than the criterion will be dropped. Else,
            the absolute scores sorted in descending order will be indexed
            accordingly. E.g. range(2) would return the two sources with the
            highest absolute score. If None, this step will be skipped.
        skew_criterion : float | int | list-like | slice
            The indices of the sorted skewness scores. If float, sources with
            absolute scores greater than the criterion will be dropped. Else,
            the absolute scores sorted in descending order will be indexed
            accordingly. E.g. range(2) would return the two sources with the
            highest absolute score. If None, this step will be skipped.
        kurt_criterion : float | int | list-like | slice
            The indices of the sorted kurtosis scores. If float, sources with
            absolute scores greater than the criterion will be dropped. Else,
            the absolute scores sorted in descending order will be indexed
            accordingly. E.g. range(2) would return the two sources with the
            highest absolute score. If None, this step will be skipped.
        var_criterion : float | int | list-like | slice
            The indices of the sorted variance scores. If float, sources with
            absolute scores greater than the criterion will be dropped. Else,
            the absolute scores sorted in descending order will be indexed
            accordingly. E.g. range(2) would return the two sources with the
            highest absolute score. If None, this step will be skipped.
        add_nodes : list of tuple
            Additional list if tuples carrying the following parameters
            of ica nodes:
            (name : str, target : str | array, score_func : callable,
            criterion : float | int | list-like | slice). This parameter is a
            generalization of the artifact specific parameters above and has
            the same structure. Example::

                add_nodes=('ECG phase lock', ECG 01',
                           my_phase_lock_function, 0.5)

        Returns
        -------
        self : instance of ICA
            The ICA object with the detected artifact indices marked for
            exclusion.
        """
        logger.info('    Searching for artifacts...')
        _detect_artifacts(self, raw=raw, start_find=start_find,
                          stop_find=stop_find, ecg_ch=ecg_ch,
                          ecg_score_func=ecg_score_func,
                          ecg_criterion=ecg_criterion,
                          eog_ch=eog_ch, eog_score_func=eog_score_func,
                          eog_criterion=eog_criterion,
                          skew_criterion=skew_criterion,
                          kurt_criterion=kurt_criterion,
                          var_criterion=var_criterion,
                          add_nodes=add_nodes)

        return self

    @verbose
    def _check_n_pca_components(self, _n_pca_comp, verbose=None):
        """Aux function."""
        if isinstance(_n_pca_comp, float):
            n, ev = _exp_var_ncomp(
                self.pca_explained_variance_, _n_pca_comp)
            logger.info(f'    Selected {n} PCA components by explained '
                        f'variance ({100 * ev}≥{100 * _n_pca_comp}%)')
            _n_pca_comp = n
        elif _n_pca_comp is None:
            _n_pca_comp = self._max_pca_components
            if _n_pca_comp is None:
                _n_pca_comp = self.pca_components_.shape[0]
        elif _n_pca_comp < self.n_components_:
            _n_pca_comp = self.n_components_

        return _n_pca_comp


def _exp_var_ncomp(var, n):
    cvar = np.asarray(var, dtype=np.float64)
    cvar = cvar.cumsum()
    cvar /= cvar[-1]
    # We allow 1., which would give us N+1
    n = min((cvar <= n).sum() + 1, len(cvar))
    return n, cvar[n - 1]


def _check_start_stop(raw, start, stop):
    """Aux function."""
    out = list()
    for st, none_ in ((start, 0), (stop, raw.n_times)):
        if st is None:
            out.append(none_)
        else:
            try:
                out.append(_ensure_int(st))
            except TypeError:  # not int-like
                out.append(raw.time_as_index(st)[0])
    return out


@verbose
def ica_find_ecg_events(raw, ecg_source, event_id=999,
                        tstart=0.0, l_freq=5, h_freq=35, qrs_threshold='auto',
                        verbose=None):
    """Find ECG peaks from one selected ICA source.

    Parameters
    ----------
    raw : instance of Raw
        Raw object to draw sources from.
    ecg_source : ndarray
        ICA source resembling ECG to find peaks from.
    event_id : int
        The index to assign to found events.
    tstart : float
        Start detection after tstart seconds. Useful when beginning
        of run is noisy.
    l_freq : float
        Low pass frequency.
    h_freq : float
        High pass frequency.
    qrs_threshold : float | str
        Between 0 and 1. qrs detection threshold. Can also be "auto" to
        automatically choose the threshold that generates a reasonable
        number of heartbeats (40-160 beats / min).
    %(verbose)s

    Returns
    -------
    ecg_events : array
        Events.
    ch_ECG : string
        Name of channel used.
    average_pulse : float.
        Estimated average pulse.
    """
    logger.info('Using ICA source to identify heart beats')

    # detecting QRS and generating event file
    ecg_events = qrs_detector(raw.info['sfreq'], ecg_source.ravel(),
                              tstart=tstart, thresh_value=qrs_threshold,
                              l_freq=l_freq, h_freq=h_freq)

    n_events = len(ecg_events)

    ecg_events = np.c_[ecg_events + raw.first_samp, np.zeros(n_events),
                       event_id * np.ones(n_events)]

    return ecg_events


@verbose
def ica_find_eog_events(raw, eog_source=None, event_id=998, l_freq=1,
                        h_freq=10, verbose=None):
    """Locate EOG artifacts from one selected ICA source.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    eog_source : ndarray
        ICA source resembling EOG to find peaks from.
    event_id : int
        The index to assign to found events.
    l_freq : float
        Low cut-off frequency in Hz.
    h_freq : float
        High cut-off frequency in Hz.
    %(verbose)s

    Returns
    -------
    eog_events : array
        Events.
    """
    eog_events = _find_eog_events(eog_source[np.newaxis], event_id=event_id,
                                  l_freq=l_freq, h_freq=h_freq,
                                  sampling_rate=raw.info['sfreq'],
                                  first_samp=raw.first_samp)
    return eog_events


def _get_target_ch(container, target):
    """Aux function."""
    # auto target selection
    picks = pick_channels(container.ch_names, include=[target])
    ref_picks = pick_types(container.info, meg=False, eeg=False, ref_meg=True)
    if len(ref_picks) > 0:
        picks = list(set(picks) - set(ref_picks))

    if len(picks) == 0:
        raise ValueError('%s not in channel list (%s)' %
                         (target, container.ch_names))
    return picks


def _find_sources(sources, target, score_func):
    """Aux function."""
    if isinstance(score_func, str):
        score_func = get_score_funcs().get(score_func, score_func)

    if not callable(score_func):
        raise ValueError('%s is not a valid score_func.' % score_func)

    scores = (score_func(sources, target) if target is not None
              else score_func(sources, 1))

    return scores


def _ica_explained_variance(ica, inst, normalize=False):
    """Check variance accounted for by each component in supplied data.

    Parameters
    ----------
    ica : ICA
        Instance of `mne.preprocessing.ICA`.
    inst : Raw | Epochs | Evoked
        Data to explain with ICA. Instance of Raw, Epochs or Evoked.
    normalize : bool
        Whether to normalize the variance.

    Returns
    -------
    var : array
        Variance explained by each component.
    """
    # check if ica is ICA and whether inst is Raw or Epochs
    if not isinstance(ica, ICA):
        raise TypeError('first argument must be an instance of ICA.')
    if not isinstance(inst, (BaseRaw, BaseEpochs, Evoked)):
        raise TypeError('second argument must an instance of either Raw, '
                        'Epochs or Evoked.')

    source_data = _get_inst_data(ica.get_sources(inst))

    # if epochs - reshape to channels x timesamples
    if isinstance(inst, BaseEpochs):
        n_epochs, n_chan, n_samp = source_data.shape
        source_data = source_data.transpose(1, 0, 2).reshape(
            (n_chan, n_epochs * n_samp))

    n_chan, n_samp = source_data.shape
    var = np.sum(ica.mixing_matrix_ ** 2, axis=0) * np.sum(
        source_data ** 2, axis=1) / (n_chan * n_samp - 1)
    if normalize:
        var /= var.sum()
    return var


def _sort_components(ica, order, copy=True):
    """Change the order of components in ica solution."""
    assert ica.n_components_ == len(order)
    if copy:
        ica = ica.copy()

    # reorder components
    ica.mixing_matrix_ = ica.mixing_matrix_[:, order]
    ica.unmixing_matrix_ = ica.unmixing_matrix_[order, :]

    # reorder labels, excludes etc.
    if isinstance(order, np.ndarray):
        order = list(order)
    if ica.exclude:
        ica.exclude = [order.index(ic) for ic in ica.exclude]
    for k in ica.labels_.keys():
        ica.labels_[k] = [order.index(ic) for ic in ica.labels_[k]]

    return ica


def _serialize(dict_, outer_sep=';', inner_sep=':'):
    """Aux function."""
    s = []
    for key, value in dict_.items():
        if callable(value):
            value = value.__name__
        elif isinstance(value, Integral):
            value = int(value)
        elif isinstance(value, dict):
            # py35 json does not support numpy int64
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    if len(subvalue) > 0:
                        if isinstance(subvalue[0], (int, np.integer)):
                            value[subkey] = [int(i) for i in subvalue]

        for cls in (np.random.RandomState, Covariance):
            if isinstance(value, cls):
                value = cls.__name__

        s.append(key + inner_sep + json.dumps(value))

    return outer_sep.join(s)


def _deserialize(str_, outer_sep=';', inner_sep=':'):
    """Aux Function."""
    out = {}
    for mapping in str_.split(outer_sep):
        k, v = mapping.split(inner_sep, 1)
        out[k] = json.loads(v)
    return out


def _write_ica(fid, ica):
    """Write an ICA object.

    Parameters
    ----------
    fid: file
        The file descriptor
    ica:
        The instance of ICA to write
    """
    ica_init = dict(noise_cov=ica.noise_cov,
                    n_components=ica.n_components,
                    n_pca_components=ica.n_pca_components,
                    max_pca_components=ica._max_pca_components,
                    current_fit=ica.current_fit,
                    allow_ref_meg=ica.allow_ref_meg)

    if ica.info is not None:
        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        if ica.info['meas_id'] is not None:
            write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, ica.info['meas_id'])

        # Write measurement info
        write_meas_info(fid, ica.info)
        end_block(fid, FIFF.FIFFB_MEAS)

    start_block(fid, FIFF.FIFFB_MNE_ICA)

    #   ICA interface params
    write_string(fid, FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS,
                 _serialize(ica_init))

    #   Channel names
    if ica.ch_names is not None:
        write_name_list(fid, FIFF.FIFF_MNE_ROW_NAMES, ica.ch_names)

    # samples on fit
    n_samples = getattr(ica, 'n_samples_', None)
    ica_misc = {'n_samples_': (None if n_samples is None else int(n_samples)),
                'labels_': getattr(ica, 'labels_', None),
                'method': getattr(ica, 'method', None),
                'n_iter_': getattr(ica, 'n_iter_', None),
                'fit_params': getattr(ica, 'fit_params', None)}

    #   ICA misc params
    write_string(fid, FIFF.FIFF_MNE_ICA_MISC_PARAMS,
                 _serialize(ica_misc))

    #   Whitener
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_WHITENER, ica.pre_whitener_)

    #   PCA components_
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_PCA_COMPONENTS,
                        ica.pca_components_)

    #   PCA mean_
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_PCA_MEAN, ica.pca_mean_)

    #   PCA explained_variance_
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_PCA_EXPLAINED_VAR,
                        ica.pca_explained_variance_)

    #   ICA unmixing
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_MATRIX, ica.unmixing_matrix_)

    #   Write bad components
    write_int(fid, FIFF.FIFF_MNE_ICA_BADS, list(ica.exclude))

    # Done!
    end_block(fid, FIFF.FIFFB_MNE_ICA)


@verbose
def read_ica(fname, verbose=None):
    """Restore ICA solution from fif file.

    Parameters
    ----------
    fname : str
        Absolute path to fif file containing ICA matrices.
        The file name should end with -ica.fif or -ica.fif.gz.
    %(verbose)s

    Returns
    -------
    ica : instance of ICA
        The ICA estimator.
    """
    check_fname(fname, 'ICA', ('-ica.fif', '-ica.fif.gz',
                               '_ica.fif', '_ica.fif.gz'))

    logger.info('Reading %s ...' % fname)
    fid, tree, _ = fiff_open(fname)

    try:
        # we used to store bads that weren't part of the info...
        info, _ = read_meas_info(fid, tree, clean_bads=True)
    except ValueError:
        logger.info('Could not find the measurement info. \n'
                    'Functionality requiring the info won\'t be'
                    ' available.')
        info = None

    ica_data = dir_tree_find(tree, FIFF.FIFFB_MNE_ICA)
    if len(ica_data) == 0:
        ica_data = dir_tree_find(tree, 123)  # Constant 123 Used before v 0.11
        if len(ica_data) == 0:
            fid.close()
            raise ValueError('Could not find ICA data')

    my_ica_data = ica_data[0]
    for d in my_ica_data['directory']:
        kind = d.kind
        pos = d.pos
        if kind == FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS:
            tag = read_tag(fid, pos)
            ica_init = tag.data
        elif kind == FIFF.FIFF_MNE_ROW_NAMES:
            tag = read_tag(fid, pos)
            ch_names = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_WHITENER:
            tag = read_tag(fid, pos)
            pre_whitener = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_COMPONENTS:
            tag = read_tag(fid, pos)
            pca_components = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_EXPLAINED_VAR:
            tag = read_tag(fid, pos)
            pca_explained_variance = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_MEAN:
            tag = read_tag(fid, pos)
            pca_mean = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_MATRIX:
            tag = read_tag(fid, pos)
            unmixing_matrix = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_BADS:
            tag = read_tag(fid, pos)
            exclude = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_MISC_PARAMS:
            tag = read_tag(fid, pos)
            ica_misc = tag.data

    fid.close()

    ica_init, ica_misc = [_deserialize(k) for k in (ica_init, ica_misc)]
    n_pca_components = ica_init.pop('n_pca_components')
    current_fit = ica_init.pop('current_fit')
    max_pca_components = ica_init.pop('max_pca_components')
    method = ica_misc.get('method', 'fastica')
    if method in _KNOWN_ICA_METHODS:
        ica_init['method'] = method
    if ica_init['noise_cov'] == Covariance.__name__:
        logger.info('Reading whitener drawn from noise covariance ...')

    logger.info('Now restoring ICA solution ...')

    # make sure dtypes are np.float64 to satisfy fast_dot
    def f(x):
        return x.astype(np.float64)

    ica_init = {k: v for k, v in ica_init.items()
                if k in _get_args(ICA.__init__)}
    ica = ICA(**ica_init)
    ica.current_fit = current_fit
    ica.ch_names = ch_names.split(':')
    if n_pca_components is not None and \
            not isinstance(n_pca_components, int_like):
        n_pca_components = np.float64(n_pca_components)
    ica.n_pca_components = n_pca_components
    ica.pre_whitener_ = f(pre_whitener)
    ica.pca_mean_ = f(pca_mean)
    ica.pca_components_ = f(pca_components)
    ica.n_components_ = unmixing_matrix.shape[0]
    ica._max_pca_components = max_pca_components
    ica._update_ica_names()
    ica.pca_explained_variance_ = f(pca_explained_variance)
    ica.unmixing_matrix_ = f(unmixing_matrix)
    ica._update_mixing_matrix()
    ica.exclude = [] if exclude is None else list(exclude)
    ica.info = info
    if 'n_samples_' in ica_misc:
        ica.n_samples_ = ica_misc['n_samples_']
    if 'labels_' in ica_misc:
        labels_ = ica_misc['labels_']
        if labels_ is not None:
            ica.labels_ = labels_
    if 'method' in ica_misc:
        ica.method = ica_misc['method']
    if 'n_iter_' in ica_misc:
        ica.n_iter_ = ica_misc['n_iter_']
    if 'fit_params' in ica_misc:
        ica.fit_params = ica_misc['fit_params']

    logger.info('Ready.')

    return ica


_ica_node = namedtuple('Node', 'name target score_func criterion')


def _detect_artifacts(ica, raw, start_find, stop_find, ecg_ch, ecg_score_func,
                      ecg_criterion, eog_ch, eog_score_func, eog_criterion,
                      skew_criterion, kurt_criterion, var_criterion,
                      add_nodes):
    """Aux Function."""
    from scipy import stats

    nodes = []
    if ecg_ch is not None:
        nodes += [_ica_node('ECG', ecg_ch, ecg_score_func, ecg_criterion)]

    if eog_ch not in [None, []]:
        if not isinstance(eog_ch, list):
            eog_ch = [eog_ch]
        for idx, ch in enumerate(eog_ch):
            nodes += [_ica_node('EOG %02d' % idx, ch, eog_score_func,
                                eog_criterion)]

    if skew_criterion is not None:
        nodes += [_ica_node('skewness', None, stats.skew, skew_criterion)]

    if kurt_criterion is not None:
        nodes += [_ica_node('kurtosis', None, stats.kurtosis, kurt_criterion)]

    if var_criterion is not None:
        nodes += [_ica_node('variance', None, np.var, var_criterion)]

    if add_nodes is not None:
        nodes.extend(add_nodes)

    for node in nodes:
        scores = ica.score_sources(raw, start=start_find, stop=stop_find,
                                   target=node.target,
                                   score_func=node.score_func)
        if isinstance(node.criterion, float):
            found = list(np.where(np.abs(scores) > node.criterion)[0])
        else:
            # Sort in descending order; use (-abs()), rather than [::-1] to
            # keep any NaN values in the end (and also keep the order of same
            # values):
            found = list(np.atleast_1d((-np.abs(scores)).argsort()
                         [node.criterion]))

        case = (len(found), _pl(found), node.name)
        logger.info('    found %s artifact%s by %s' % case)
        ica.exclude = list(ica.exclude) + found

    logger.info('Artifact indices found:\n    ' + str(ica.exclude).strip('[]'))
    if len(set(ica.exclude)) != len(ica.exclude):
        logger.info('    Removing duplicate indices...')
        ica.exclude = list(set(ica.exclude))

    logger.info('Ready.')


@verbose
def _band_pass_filter(inst, sources, target, l_freq, h_freq, verbose=None):
    """Optionally band-pass filter the data."""
    if l_freq is not None and h_freq is not None:
        logger.info('... filtering ICA sources')
        # use FIR here, steeper is better
        kw = dict(phase='zero-double', filter_length='10s', fir_window='hann',
                  l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                  fir_design='firwin2')
        sources = filter_data(sources, inst.info['sfreq'], l_freq, h_freq,
                              **kw)
        logger.info('... filtering target')
        target = filter_data(target, inst.info['sfreq'], l_freq, h_freq, **kw)
    elif l_freq is not None or h_freq is not None:
        raise ValueError('Must specify both pass bands')
    return sources, target


# #############################################################################
# CORRMAP

def _find_max_corrs(all_maps, target, threshold):
    """Compute correlations between template and target components."""
    all_corrs = [compute_corr(target, subj.T) for subj in all_maps]
    abs_corrs = [np.abs(a) for a in all_corrs]
    corr_polarities = [np.sign(a) for a in all_corrs]

    if threshold <= 1:
        max_corrs = [list(np.nonzero(s_corr > threshold)[0])
                     for s_corr in abs_corrs]
    else:
        max_corrs = [list(_find_outliers(s_corr, threshold=threshold))
                     for s_corr in abs_corrs]

    am = [l_[i] for l_, i_s in zip(abs_corrs, max_corrs)
          for i in i_s]
    median_corr_with_target = np.median(am) if len(am) > 0 else 0

    polarities = [l_[i] for l_, i_s in zip(corr_polarities, max_corrs)
                  for i in i_s]

    maxmaps = [l_[i] for l_, i_s in zip(all_maps, max_corrs)
               for i in i_s]

    if len(maxmaps) == 0:
        return [], 0, 0, []
    newtarget = np.zeros(maxmaps[0].size)
    std_of_maps = np.std(np.asarray(maxmaps))
    mean_of_maps = np.std(np.asarray(maxmaps))
    for maxmap, polarity in zip(maxmaps, polarities):
        newtarget += (maxmap / std_of_maps - mean_of_maps) * polarity

    newtarget /= len(maxmaps)
    newtarget *= std_of_maps

    sim_i_o = np.abs(np.corrcoef(target, newtarget)[1, 0])

    return newtarget, median_corr_with_target, sim_i_o, max_corrs


@verbose
def corrmap(icas, template, threshold="auto", label=None, ch_type="eeg",
            plot=True, show=True, outlines='head',
            sensors=True, contours=6, cmap=None, sphere=None, verbose=None):
    """Find similar Independent Components across subjects by map similarity.

    Corrmap (Viola et al. 2009 Clin Neurophysiol) identifies the best group
    match to a supplied template. Typically, feed it a list of fitted ICAs and
    a template IC, for example, the blink for the first subject, to identify
    specific ICs across subjects.

    The specific procedure consists of two iterations. In a first step, the
    maps best correlating with the template are identified. In the next step,
    the analysis is repeated with the mean of the maps identified in the first
    stage.

    Run with ``plot`` and ``show`` set to ``True`` and ``label=False`` to find
    good parameters. Then, run with labelling enabled to apply the
    labelling in the IC objects. (Running with both ``plot`` and ``labels``
    off does nothing.)

    Outputs a list of fitted ICAs with the indices of the marked ICs in a
    specified field.

    The original Corrmap website: www.debener.de/corrmap/corrmapplugin1.html

    Parameters
    ----------
    icas : list of mne.preprocessing.ICA
        A list of fitted ICA objects.
    template : tuple | np.ndarray, shape (n_components,)
        Either a tuple with two elements (int, int) representing the list
        indices of the set from which the template should be chosen, and the
        template. E.g., if template=(1, 0), the first IC of the 2nd ICA object
        is used.
        Or a numpy array whose size corresponds to each IC map from the
        supplied maps, in which case this map is chosen as the template.
    threshold : "auto" | list of float | float
        Correlation threshold for identifying ICs
        If "auto", search for the best map by trying all correlations between
        0.6 and 0.95. In the original proposal, lower values are considered,
        but this is not yet implemented.
        If list of floats, search for the best map in the specified range of
        correlation strengths. As correlation values, must be between 0 and 1
        If float > 0, select ICs correlating better than this.
        If float > 1, use z-scoring to identify ICs within subjects (not in
        original Corrmap)
        Defaults to "auto".
    label : None | str
        If not None, categorised ICs are stored in a dictionary ``labels_``
        under the given name. Preexisting entries will be appended to
        (excluding repeats), not overwritten. If None, a dry run is performed
        and the supplied ICs are not changed.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg'
        The channel type to plot. Defaults to 'eeg'.
    plot : bool
        Should constructed template and selected maps be plotted? Defaults
        to True.
    show : bool
        Show figures if True.
    %(topomap_outlines)s
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses). If True, a circle will be
        used (via .add_artist). Defaults to True.
    contours : int | array of float
        The number of contour lines to draw. If 0, no contours will be drawn.
        When an integer, matplotlib ticker locator is used to find suitable
        values for the contour thresholds (may sometimes be inaccurate, use
        array for accuracy). If an array, the values represent the levels for
        the contours. Defaults to 6.
    cmap : None | matplotlib colormap
        Colormap for the plot. If ``None``, defaults to 'Reds_r' for norm data,
        otherwise to 'RdBu_r'.
    %(topomap_sphere_auto)s
    %(verbose)s

    Returns
    -------
    template_fig : Figure
        Figure showing the template.
    labelled_ics : Figure
        Figure showing the labelled ICs in all ICA decompositions.
    """
    if not isinstance(plot, bool):
        raise ValueError("`plot` must be of type `bool`")

    same_chans = _check_all_same_channel_names(icas)
    if same_chans is False:
        raise ValueError("Not all ICA instances have the same channel names. "
                         "Corrmap requires all instances to have the same "
                         "montage. Consider interpolating bad channels before "
                         "running ICA.")

    threshold_extra = ''
    if threshold == 'auto':
        threshold = np.arange(60, 95, dtype=np.float64) / 100.
        threshold_extra = ' ("auto")'

    all_maps = [ica.get_components().T for ica in icas]

    # check if template is an index to one IC in one ICA object, or an array
    if len(template) == 2:
        target = all_maps[template[0]][template[1]]
        is_subject = True
    elif template.ndim == 1 and len(template) == all_maps[0].shape[1]:
        target = template
        is_subject = False
    else:
        raise ValueError("`template` must be a length-2 tuple or an array the "
                         "size of the ICA maps.")

    template_fig, labelled_ics = None, None
    if plot is True:
        if is_subject:  # plotting from an ICA object
            ttl = 'Template from subj. {}'.format(str(template[0]))
            template_fig = icas[template[0]].plot_components(
                picks=template[1], ch_type=ch_type, title=ttl,
                outlines=outlines, cmap=cmap, contours=contours,
                show=show, topomap_args=dict(sphere=sphere))
        else:  # plotting an array
            template_fig = _plot_corrmap([template], [0], [0], ch_type,
                                         icas[0].copy(), "Template",
                                         outlines=outlines, cmap=cmap,
                                         contours=contours,
                                         show=show, template=True,
                                         sphere=sphere)
        template_fig.subplots_adjust(top=0.8)
        template_fig.canvas.draw()

    # first run: use user-selected map
    threshold = np.atleast_1d(np.array(threshold, float)).ravel()
    threshold_err = ('No component detected using when z-scoring '
                     'threshold%s %s, consider using a more lenient '
                     'threshold' % (threshold_extra, threshold))
    if len(all_maps) == 0:
        raise RuntimeError(threshold_err)
    paths = [_find_max_corrs(all_maps, target, t) for t in threshold]
    # find iteration with highest avg correlation with target
    new_target, _, _, _ = paths[np.argmax([path[2] for path in paths])]

    # second run: use output from first run
    if len(all_maps) == 0 or len(new_target) == 0:
        raise RuntimeError(threshold_err)
    paths = [_find_max_corrs(all_maps, new_target, t) for t in threshold]
    del new_target
    # find iteration with highest avg correlation with target
    _, median_corr, _, max_corrs = paths[
        np.argmax([path[1] for path in paths])]

    allmaps, indices, subjs, nones = [list() for _ in range(4)]
    logger.info('Median correlation with constructed map: %0.3f' % median_corr)
    del median_corr
    if plot is True:
        logger.info('Displaying selected ICs per subject.')

    for ii, (ica, max_corr) in enumerate(zip(icas, max_corrs)):
        if len(max_corr) > 0:
            if isinstance(max_corr[0], np.ndarray):
                max_corr = max_corr[0]
            if label is not None:
                ica.labels_[label] = list(set(list(max_corr) +
                                              ica.labels_.get(label, list())))
            if plot is True:
                allmaps.extend(ica.get_components()[:, max_corr].T)
                subjs.extend([ii] * len(max_corr))
                indices.extend(max_corr)
        else:
            if (label is not None) and (label not in ica.labels_):
                ica.labels_[label] = list()
            nones.append(ii)

    if len(nones) == 0:
        logger.info('At least 1 IC detected for each subject.')
    else:
        logger.info('No maps selected for subject%s %s, '
                    'consider a more liberal threshold.'
                    % (_pl(nones), nones))

    if plot is True:
        labelled_ics = _plot_corrmap(allmaps, subjs, indices, ch_type, ica,
                                     label, outlines=outlines, cmap=cmap,
                                     contours=contours,
                                     show=show, sphere=sphere)
        return template_fig, labelled_ics
    else:
        return None


@verbose
def read_ica_eeglab(fname, *, verbose=None):
    """Load ICA information saved in an EEGLAB .set file.

    Parameters
    ----------
    fname : str
        Complete path to a .set EEGLAB file that contains an ICA object.
    %(verbose)s

    Returns
    -------
    ica : instance of ICA
        An ICA object based on the information contained in the input file.
    """
    from scipy import linalg
    eeg = _check_load_mat(fname, None)
    info, eeg_montage, _ = _get_info(eeg)
    info.set_montage(eeg_montage)
    pick_info(info, np.round(eeg['icachansind']).astype(int) - 1, copy=False)

    rank = eeg.icasphere.shape[0]
    n_components = eeg.icaweights.shape[0]

    ica = ICA(method='imported_eeglab', n_components=n_components)

    ica.current_fit = "eeglab"
    ica.ch_names = info["ch_names"]
    ica.n_pca_components = None
    ica.n_components_ = n_components

    n_ch = len(ica.ch_names)
    assert len(eeg.icachansind) == n_ch

    ica.pre_whitener_ = np.ones((n_ch, 1))
    ica.pca_mean_ = np.zeros(n_ch)

    assert eeg.icasphere.shape[1] == n_ch
    assert eeg.icaweights.shape == (n_components, rank)

    # When PCA reduction is used in EEGLAB, runica returns
    # weights= weights*sphere*eigenvectors(:,1:ncomps)';
    # sphere = eye(urchans). When PCA reduction is not used, we have:
    #
    #     eeg.icawinv == pinv(eeg.icaweights @ eeg.icasphere)
    #
    # So in either case, we can use SVD to get our square whitened
    # weights matrix (u * s) and our PCA vectors (v) back:
    use = eeg.icaweights @ eeg.icasphere
    use_check = linalg.pinv(eeg.icawinv)
    if not np.allclose(use, use_check, rtol=1e-6):
        warn('Mismatch between icawinv and icaweights @ icasphere from EEGLAB '
             'possibly due to ICA component removal, assuming icawinv is '
             'correct')
        use = use_check
    u, s, v = _safe_svd(use, full_matrices=False)
    ica.unmixing_matrix_ = u * s
    ica.pca_components_ = v
    ica.pca_explained_variance_ = s * s
    ica.info = info
    ica._update_mixing_matrix()
    ica._update_ica_names()
    return ica
