# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Juergen Dammers <j.dammers@fz-juelich.de>
#
# License: BSD (3-clause)

from inspect import isfunction
from collections import namedtuple
from copy import deepcopy

import os
import json

import numpy as np
from scipy import linalg

from .ecg import (qrs_detector, _get_ecg_channel_index, _make_ecg,
                  create_ecg_epochs)
from .eog import _find_eog_events, _get_eog_channel_index
from .infomax_ import infomax

from ..cov import compute_whitener
from .. import Covariance, Evoked
from ..io.pick import (pick_types, pick_channels, pick_info,
                       _pick_data_channels, _DATA_CH_TYPES_SPLIT)
from ..io.write import (write_double_matrix, write_string,
                        write_name_list, write_int, start_block,
                        end_block)
from ..io.tree import dir_tree_find
from ..io.open import fiff_open
from ..io.tag import read_tag
from ..io.meas_info import write_meas_info, read_meas_info
from ..io.constants import Bunch, FIFF
from ..io.base import BaseRaw
from ..epochs import BaseEpochs
from ..viz import (plot_ica_components, plot_ica_scores,
                   plot_ica_sources, plot_ica_overlay)
from ..viz.ica import plot_ica_properties
from ..viz.utils import (_prepare_trellis, tight_layout, plt_show,
                         _setup_vmin_vmax)
from ..viz.topomap import (_prepare_topo_plot, _check_outlines,
                           plot_topomap, _hide_frame)

from ..channels.channels import _contains_ch_type, ContainsMixin
from ..io.write import start_file, end_file, write_id
from ..utils import (check_version, logger, check_fname, verbose,
                     _reject_data_segments, check_random_state,
                     _get_fast_dot, compute_corr, _get_inst_data,
                     copy_function_doc_to_method_doc, _pl)
from ..fixes import _get_args
from ..filter import filter_data
from .bads import find_outliers
from .ctps_ import ctps
from ..externals.six import string_types, text_type
from ..io.pick import channel_type


__all__ = ('ICA', 'ica_find_ecg_events', 'ica_find_eog_events',
           'get_score_funcs', 'read_ica', 'run_ica')


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


# makes score funcs attr accessible for users
def get_score_funcs():
    """Get the score functions."""
    from scipy import stats
    from scipy.spatial import distance
    score_funcs = Bunch()
    xy_arg_dist_funcs = [(n, f) for n, f in vars(distance).items()
                         if isfunction(f) and not n.startswith('_')]
    xy_arg_stats_funcs = [(n, f) for n, f in vars(stats).items()
                          if isfunction(f) and not n.startswith('_')]
    score_funcs.update(dict((n, _make_xy_sfunc(f))
                            for n, f in xy_arg_dist_funcs
                            if _get_args(f) == ['u', 'v']))
    score_funcs.update(dict((n, _make_xy_sfunc(f, ndim_output=True))
                            for n, f in xy_arg_stats_funcs
                            if _get_args(f) == ['x', 'y']))
    return score_funcs


def _check_for_unsupported_ica_channels(picks, info):
    """Check for channels in picks that are not considered valid channels.

    Accepted channels are the data channels
    ('seeg','ecog','eeg', 'hbo', 'hbr', 'mag', and 'grad') and 'eog'.
    This prevents the program from crashing without
    feedback when a bad channel is provided to ICA whitening.
    """
    if picks is None:
        return
    elif len(picks) == 0:
        raise ValueError('No channels provided to ICA')
    types = _DATA_CH_TYPES_SPLIT + ['eog']
    chs = list(set([channel_type(info, j) for j in picks]))
    check = all([ch in types for ch in chs])
    if not check:
        raise ValueError('Invalid channel type(s) passed for ICA.\n'
                         'Only the following channels are supported {0}\n'
                         'Following types were passed {1}\n'
                         .format(types, chs))


class ICA(ContainsMixin):
    """M/EEG signal decomposition using Independent Component Analysis (ICA).

    This object can be used to estimate ICA components and then
    remove some from Raw or Epochs for data exploration or artifact
    correction.

    Caveat! If supplying a noise covariance keep track of the projections
    available in the cov or in the raw object. For example, if you are
    interested in EOG or ECG artifacts, EOG and ECG projections should be
    temporally removed before fitting the ICA. You can say::

        >> projs, raw.info['projs'] = raw.info['projs'], []
        >> ica.fit(raw)
        >> raw.info['projs'] = projs

    .. note:: Methods implemented are FastICA (default), Infomax and
              Extended-Infomax. Infomax can be quite sensitive to differences
              in floating point arithmetic due to exponential non-linearity.
              Extended-Infomax seems to be more stable in this respect
              enhancing reproducibility and stability of results.

    Parameters
    ----------
    n_components : int | float | None
        The number of components used for ICA decomposition. If int, it must be
        smaller then max_pca_components. If None, all PCA components will be
        used. If float between 0 and 1 components will be selected by the
        cumulative percentage of explained variance.
    max_pca_components : int | None
        The number of components used for PCA decomposition. If None, no
        dimension reduction will be applied and max_pca_components will equal
        the number of channels supplied on decomposing data. Defaults to None.
    n_pca_components : int | float
        The number of PCA components used after ICA recomposition. The ensuing
        attribute allows to balance noise reduction against potential loss of
        features due to dimensionality reduction. If greater than
        ``self.n_components_``, the next ``n_pca_components`` minus
        ``n_components_`` PCA components will be added before restoring the
        sensor space data. The attribute gets updated each time the according
        parameter for in .pick_sources_raw or .pick_sources_epochs is changed.
        If float, the number of components selected matches the number of
        components with a cumulative explained variance below
        `n_pca_components`.
    noise_cov : None | instance of mne.cov.Covariance
        Noise covariance used for whitening. If None, channels are just
        z-scored.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the FastICA estimation.
        As the estimation is non-deterministic it can be useful to
        fix the seed to have reproducible results. Defaults to None.
    method : {'fastica', 'infomax', 'extended-infomax'}
        The ICA method to use. Defaults to 'fastica'.
    fit_params : dict | None.
        Additional parameters passed to the ICA estimator chosen by `method`.
    max_iter : int, optional
        Maximum number of iterations during fit.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Attributes
    ----------
    current_fit : str
        Flag informing about which data type (raw or epochs) was used for
        the fit.
    ch_names : list-like
        Channel names resulting from initial picking.
        The number of components used for ICA decomposition.
    ``n_components_`` : int
        If fit, the actual number of components used for ICA decomposition.
    n_pca_components : int
        See above.
    max_pca_components : int
        The number of components used for PCA dimensionality reduction.
    verbose : bool, str, int, or None
        See above.
    ``pca_components_`` : ndarray
        If fit, the PCA components
    ``pca_mean_`` : ndarray
        If fit, the mean vector used to center the data before doing the PCA.
    ``pca_explained_variance_`` : ndarray
        If fit, the variance explained by each PCA component
    ``mixing_matrix_`` : ndarray
        If fit, the mixing matrix to restore observed data, else None.
    ``unmixing_matrix_`` : ndarray
        If fit, the matrix to unmix observed data, else None.
    exclude : list
        List of sources indices to exclude, i.e. artifact components identified
        throughout the ICA solution. Indices added to this list, will be
        dispatched to the .pick_sources methods. Source indices passed to
        the .pick_sources method via the 'exclude' argument are added to the
        .exclude attribute. When saving the ICA also the indices are restored.
        Hence, artifact components once identified don't have to be added
        again. To dump this 'artifact memory' say: ica.exclude = []
    info : None | instance of Info
        The measurement info copied from the object fitted.
    ``n_samples_`` : int
        the number of samples used on fit.
    ``labels_`` : dict
        A dictionary of independent component indices, grouped by types of
        independent components. This attribute is set by some of the artifact
        detection functions.
    """

    @verbose
    def __init__(self, n_components=None, max_pca_components=None,
                 n_pca_components=None, noise_cov=None, random_state=None,
                 method='fastica', fit_params=None, max_iter=200,
                 verbose=None):  # noqa: D102
        methods = ('fastica', 'infomax', 'extended-infomax')
        if method not in methods:
            raise ValueError('`method` must be "%s". You passed: "%s"' %
                             ('" or "'.join(methods), method))
        if not check_version('sklearn', '0.12'):
            raise RuntimeError('the scikit-learn package (version >= 0.12)'
                               'is required for ICA')

        self.noise_cov = noise_cov

        if (n_components is not None and
                max_pca_components is not None and
                n_components > max_pca_components):
            raise ValueError('n_components must be smaller than '
                             'max_pca_components')

        if isinstance(n_components, float) \
                and not 0 < n_components <= 1:
            raise ValueError('Selecting ICA components by explained variance '
                             'needs values between 0.0 and 1.0 ')

        self.current_fit = 'unfitted'
        self.verbose = verbose
        self.n_components = n_components
        self.max_pca_components = max_pca_components
        self.n_pca_components = n_pca_components
        self.ch_names = None
        self.random_state = random_state

        if fit_params is None:
            fit_params = {}
        fit_params = deepcopy(fit_params)  # avoid side effects
        if "extended" in fit_params:
            raise ValueError("'extended' parameter provided. You should "
                             "rather use method='extended-infomax'.")
        if method == 'fastica':
            update = {'algorithm': 'parallel', 'fun': 'logcosh',
                      'fun_args': None}
            fit_params.update(dict((k, v) for k, v in update.items() if k
                              not in fit_params))
        elif method == 'infomax':
            fit_params.update({'extended': False})
        elif method == 'extended-infomax':
            fit_params.update({'extended': True})
        if 'max_iter' not in fit_params:
            fit_params['max_iter'] = max_iter
        self.max_iter = max_iter
        self.fit_params = fit_params

        self.exclude = []
        self.info = None
        self.method = method
        self.labels_ = dict()

    def __repr__(self):
        """ICA fit information."""
        if self.current_fit == 'unfitted':
            s = 'no'
        elif self.current_fit == 'raw':
            s = 'raw data'
        else:
            s = 'epochs'
        s += ' decomposition, '
        s += 'fit (%s): %s samples, ' % (self.method,
                                         str(getattr(self, 'n_samples_', '')))
        s += ('%s components' % str(self.n_components_) if
              hasattr(self, 'n_components_') else
              'no dimension reduction')
        if self.info is not None:
            ch_fit = ['"%s"' % c for c in _DATA_CH_TYPES_SPLIT if c in self]
            s += ', channels used: {0}'.format('; '.join(ch_fit))
        if self.exclude:
            s += ', %i sources marked for exclusion' % len(self.exclude)

        return '<ICA  |  %s>' % s

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
        inst : instance of Raw, Epochs or Evoked
            Raw measurements to be decomposed.
        picks : array-like of int
            Channels to be included. This selection remains throughout the
            initialized ICA solution. If None only good data channels are used.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
        decim : int | None
            Increment for selecting each nth time slice. If None, all samples
            within ``start`` and ``stop`` are used.
        reject : dict | None
            Rejection parameters based on peak-to-peak amplitude.
            Valid keys are 'grad', 'mag', 'eeg', 'seeg', 'ecog', 'eog', 'ecg',
            'hbo', 'hbr'.
            If reject is None then no rejection is done. Example::

                reject = dict(grad=4000e-13, # T / m (gradiometers)
                              mag=4e-12, # T (magnetometers)
                              eeg=40e-6, # V (EEG channels)
                              eog=250e-6 # V (EOG channels)
                              )

            It only applies if `inst` is of type Raw.
        flat : dict | None
            Rejection parameters based on flatness of signal.
            Valid keys are 'grad', 'mag', 'eeg', 'seeg', 'ecog', 'eog', 'ecg',
            'hbo', 'hbr'.
            Values are floats that set the minimum acceptable peak-to-peak
            amplitude. If flat is None then no rejection is done.
            It only applies if `inst` is of type Raw.
        tstep : float
            Length of data chunks for artifact rejection in seconds.
            It only applies if `inst` is of type Raw.
        reject_by_annotation : bool
            Whether to omit bad segments from the data before fitting. If True,
            annotated segments with a description that starts with 'bad' are
            omitted. Has no effect if ``inst`` is an Epochs or Evoked object.
            Defaults to True.

            .. versionadded:: 0.14.0

        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        self : instance of ICA
            Returns the modified instance.
        """
        if isinstance(inst, (BaseRaw, BaseEpochs)):
            _check_for_unsupported_ica_channels(picks, inst.info)
            if isinstance(inst, BaseRaw):
                self._fit_raw(inst, picks, start, stop, decim, reject, flat,
                              tstep, reject_by_annotation, verbose)
            elif isinstance(inst, BaseEpochs):
                self._fit_epochs(inst, picks, decim, verbose)
        else:
            raise ValueError('Data input must be of Raw or Epochs type')

        # sort ICA components by explained variance
        var = _ica_explained_variance(self, inst)
        var_ord = var.argsort()[::-1]
        _sort_components(self, var_ord, copy=False)

        return self

    def _reset(self):
        """Aux method."""
        del self._pre_whitener
        del self.unmixing_matrix_
        del self.mixing_matrix_
        del self.n_components_
        del self.n_samples_
        del self.pca_components_
        del self.pca_explained_variance_
        del self.pca_mean_
        if hasattr(self, 'drop_inds_'):
            del self.drop_inds_

    def _fit_raw(self, raw, picks, start, stop, decim, reject, flat, tstep,
                 reject_by_annotation, verbose):
        """Aux method."""
        if self.current_fit != 'unfitted':
            self._reset()

        if picks is None:  # just use good data channels
            picks = _pick_data_channels(raw.info, exclude='bads',
                                        with_ref_meg=False)

        logger.info('Fitting ICA to data using %i channels. \n'
                    'Please be patient, this may take some time' % len(picks))

        if self.max_pca_components is None:
            self.max_pca_components = len(picks)
            logger.info('Inferring max_pca_components from picks.')

        self.info = pick_info(raw.info, picks)
        if self.info['comps']:
            self.info['comps'] = []
        self.ch_names = self.info['ch_names']
        start, stop = _check_start_stop(raw, start, stop)

        reject_by_annotation = 'omit' if reject_by_annotation else None
        # this will be a copy
        data = raw.get_data(picks, start, stop, reject_by_annotation)

        # this will be a view
        if decim is not None:
            data = data[:, ::decim]

        # this will make a copy
        if (reject is not None) or (flat is not None):
            data, self.drop_inds_ = _reject_data_segments(data, reject, flat,
                                                          decim, self.info,
                                                          tstep)

        self.n_samples_ = data.shape[1]
        # this may operate inplace or make a copy
        data, self._pre_whitener = self._pre_whiten(data, raw.info, picks)

        self._fit(data, self.max_pca_components, 'raw')

        return self

    def _fit_epochs(self, epochs, picks, decim, verbose):
        """Aux method."""
        if self.current_fit != 'unfitted':
            self._reset()

        if picks is None:
            picks = _pick_data_channels(epochs.info, exclude='bads',
                                        with_ref_meg=False)
        logger.info('Fitting ICA to data using %i channels. \n'
                    'Please be patient, this may take some time' % len(picks))

        # filter out all the channels the raw wouldn't have initialized
        self.info = pick_info(epochs.info, picks)
        if self.info['comps']:
            self.info['comps'] = []
        self.ch_names = self.info['ch_names']

        if self.max_pca_components is None:
            self.max_pca_components = len(picks)
            logger.info('Inferring max_pca_components from picks.')

        # this should be a copy (picks a list of int)
        data = epochs.get_data()[:, picks]
        # this will be a view
        if decim is not None:
            data = data[:, :, ::decim]

        self.n_samples_ = np.prod(data[:, 0, :].shape)

        # This will make at least one copy (one from hstack, maybe one
        # more from _pre_whiten)
        data, self._pre_whitener = \
            self._pre_whiten(np.hstack(data), epochs.info, picks)

        self._fit(data, self.max_pca_components, 'epochs')

        return self

    def _pre_whiten(self, data, info, picks):
        """Aux function."""
        fast_dot = _get_fast_dot()
        has_pre_whitener = hasattr(self, '_pre_whitener')
        if not has_pre_whitener and self.noise_cov is None:
            # use standardization as whitener
            # Scale (z-score) the data by channel type
            info = pick_info(info, picks)
            pre_whitener = np.empty([len(data), 1])
            for ch_type in _DATA_CH_TYPES_SPLIT + ['eog']:
                if _contains_ch_type(info, ch_type):
                    if ch_type == 'seeg':
                        this_picks = pick_types(info, meg=False, seeg=True)
                    elif ch_type == 'ecog':
                        this_picks = pick_types(info, meg=False, ecog=True)
                    elif ch_type == 'eeg':
                        this_picks = pick_types(info, meg=False, eeg=True)
                    elif ch_type in ('mag', 'grad'):
                        this_picks = pick_types(info, meg=ch_type)
                    elif ch_type == 'eog':
                        this_picks = pick_types(info, meg=False, eog=True)
                    elif ch_type in ('hbo', 'hbr'):
                        this_picks = pick_types(info, meg=False, fnirs=ch_type)
                    else:
                        raise RuntimeError('Should not be reached.'
                                           'Unsupported channel {0}'
                                           .format(ch_type))
                    pre_whitener[this_picks] = np.std(data[this_picks])
            data /= pre_whitener
        elif not has_pre_whitener and self.noise_cov is not None:
            pre_whitener, _ = compute_whitener(self.noise_cov, info, picks)
            assert data.shape[0] == pre_whitener.shape[1]
            data = fast_dot(pre_whitener, data)
        elif has_pre_whitener and self.noise_cov is None:
            data /= self._pre_whitener
            pre_whitener = self._pre_whitener
        else:
            data = fast_dot(self._pre_whitener, data)
            pre_whitener = self._pre_whitener

        return data, pre_whitener

    def _fit(self, data, max_pca_components, fit_type):
        """Aux function."""
        random_state = check_random_state(self.random_state)

        if not check_version('sklearn', '0.18'):
            from sklearn.decomposition import RandomizedPCA
            # XXX fix copy==True later. Bug in sklearn, see PR #2273
            pca = RandomizedPCA(n_components=max_pca_components, whiten=True,
                                copy=True, random_state=random_state)

        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=max_pca_components, copy=True, whiten=True,
                      svd_solver='randomized', random_state=random_state)

        if isinstance(self.n_components, float):
            # compute full feature variance before doing PCA
            full_var = np.var(data, axis=1).sum()

        data = pca.fit_transform(data.T)

        if isinstance(self.n_components, float):
            # compute eplained variance manually, cf. sklearn bug
            # fixed in #2664
            explained_variance_ratio_ = pca.explained_variance_ / full_var
            n_components_ = np.sum(explained_variance_ratio_.cumsum() <=
                                   self.n_components)
            if n_components_ < 1:
                raise RuntimeError('One PCA component captures most of the '
                                   'explained variance, your threshold resu'
                                   'lts in 0 components. You should select '
                                   'a higher value.')
            logger.info('Selection by explained variance: %i components' %
                        n_components_)
            sel = slice(n_components_)
        else:
            if self.n_components is not None:  # normal n case
                sel = slice(self.n_components)
                logger.info('Selection by number: %i components' %
                            self.n_components)
            else:  # None case
                logger.info('Using all PCA components: %i'
                            % len(pca.components_))
                sel = slice(len(pca.components_))

        # the things to store for PCA
        self.pca_mean_ = pca.mean_
        self.pca_components_ = pca.components_
        self.pca_explained_variance_ = exp_var = pca.explained_variance_
        if not check_version('sklearn', '0.18'):
            # unwhiten pca components and put scaling in unmixing matrix later.
            # RandomizedPCA applies the whitening to the components
            # but not the new PCA class.
            self.pca_components_ *= np.sqrt(exp_var[:, None])
        del pca
        # update number of components
        self.n_components_ = sel.stop
        if self.n_pca_components is not None:
            if self.n_pca_components > len(self.pca_components_):
                self.n_pca_components = len(self.pca_components_)

        # Take care of ICA
        if self.method == 'fastica':
            from sklearn.decomposition import FastICA  # to avoid strong dep.
            ica = FastICA(whiten=False,
                          random_state=random_state, **self.fit_params)
            ica.fit(data[:, sel])
            # get unmixing and add scaling
            self.unmixing_matrix_ = getattr(ica, 'components_',
                                            'unmixing_matrix_')
        elif self.method in ('infomax', 'extended-infomax'):
            self.unmixing_matrix_ = infomax(data[:, sel],
                                            random_state=random_state,
                                            **self.fit_params)
        self.unmixing_matrix_ /= np.sqrt(exp_var[sel])[None, :]
        self.mixing_matrix_ = linalg.pinv(self.unmixing_matrix_)
        self.current_fit = fit_type

    def _transform(self, data):
        """Compute sources from data (operates inplace)."""
        fast_dot = _get_fast_dot()
        if self.pca_mean_ is not None:
            data -= self.pca_mean_[:, None]

        # Apply first PCA
        pca_data = fast_dot(self.pca_components_[:self.n_components_], data)
        # Apply unmixing to low dimension PCA
        sources = fast_dot(self.unmixing_matrix_, pca_data)
        return sources

    def _transform_raw(self, raw, start, stop):
        """Transform raw data."""
        if not hasattr(self, 'mixing_matrix_'):
            raise RuntimeError('No fit available. Please fit ICA.')
        start, stop = _check_start_stop(raw, start, stop)

        picks = pick_types(raw.info, include=self.ch_names, exclude='bads',
                           meg=False, ref_meg=False)
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Raw doesn\'t match fitted data: %i channels '
                               'fitted but %i channels supplied. \nPlease '
                               'provide Raw compatible with '
                               'ica.ch_names' % (len(self.ch_names),
                                                 len(picks)))

        data, _ = self._pre_whiten(raw[picks, start:stop][0], raw.info, picks)
        return self._transform(data)

    def _transform_epochs(self, epochs, concatenate):
        """Aux method."""
        if not hasattr(self, 'mixing_matrix_'):
            raise RuntimeError('No fit available. Please fit ICA')

        picks = pick_types(epochs.info, include=self.ch_names, exclude='bads',
                           meg=False, ref_meg=False)
        # special case where epochs come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Epochs don\'t match fitted data: %i channels '
                               'fitted but %i channels supplied. \nPlease '
                               'provide Epochs compatible with '
                               'ica.ch_names' % (len(self.ch_names),
                                                 len(picks)))

        data = np.hstack(epochs.get_data()[:, picks])
        data, _ = self._pre_whiten(data, epochs.info, picks)
        sources = self._transform(data)

        if not concatenate:
            # Put the data back in 3D
            sources = np.array(np.split(sources, len(epochs.events), 1))

        return sources

    def _transform_evoked(self, evoked):
        """Aux method."""
        if not hasattr(self, 'mixing_matrix_'):
            raise RuntimeError('No fit available. Please first fit ICA')

        picks = pick_types(evoked.info, include=self.ch_names, exclude='bads',
                           meg=False, ref_meg=False)

        if len(picks) != len(self.ch_names):
            raise RuntimeError('Evoked doesn\'t match fitted data: %i channels'
                               ' fitted but %i channels supplied. \nPlease '
                               'provide Evoked compatible with '
                               'ica.ch_names' % (len(self.ch_names),
                                                 len(picks)))

        data, _ = self._pre_whiten(evoked.data[picks], evoked.info, picks)
        sources = self._transform(data)

        return sources

    def get_components(self):
        """Get ICA topomap for components as numpy arrays.

        Returns
        -------
        components : array, shape (n_channels, n_components)
            The ICA components (maps).
        """
        fast_dot = _get_fast_dot()
        return fast_dot(self.mixing_matrix_[:, :self.n_components_].T,
                        self.pca_components_[:self.n_components_]).T

    def get_sources(self, inst, add_channels=None, start=None, stop=None):
        """Estimate sources given the unmixing matrix.

        This method will return the sources in the container format passed.
        Typical usecases:

        1. pass Raw object to use `raw.plot` for ICA sources
        2. pass Epochs object to compute trial-based statistics in ICA space
        3. pass Evoked object to investigate time-locking in ICA space

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Object to compute sources from and to represent sources in.
        add_channels : None | list of str
            Additional channels  to be added. Useful to e.g. compare sources
            with some reference. Defaults to None
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
            sources = self._sources_as_raw(inst, add_channels, start, stop)
        elif isinstance(inst, BaseEpochs):
            sources = self._sources_as_epochs(inst, add_channels, False)
        elif isinstance(inst, Evoked):
            sources = self._sources_as_evoked(inst, add_channels)
        else:
            raise ValueError('Data input must be of Raw, Epochs or Evoked '
                             'type')

        return sources

    def _sources_as_raw(self, raw, add_channels, start, stop):
        """Aux method."""
        # merge copied instance and picked data with sources
        sources = self._transform_raw(raw, start=start, stop=stop)
        if raw.preload:  # get data and temporarily delete
            data = raw._data
            del raw._data

        out = raw.copy()  # copy and reappend
        if raw.preload:
            raw._data = data

        # populate copied raw.
        start, stop = _check_start_stop(raw, start, stop)
        if add_channels is not None:
            raw_picked = raw.copy().pick_channels(add_channels)
            data_, times_ = raw_picked[:, start:stop]
            data_ = np.r_[sources, data_]
        else:
            data_ = sources
            _, times_ = raw[0, start:stop]
        out._data = data_
        out._times = times_
        out._filenames = [None]
        out.preload = True

        # update first and last samples
        out._first_samps = np.array([raw.first_samp +
                                     (start if start else 0)])
        out._last_samps = np.array([out.first_samp + stop
                                    if stop else raw.last_samp])

        out._projector = None
        self._export_info(out.info, raw, add_channels)
        out._update_times()

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
        ch_info = info['chs'] = []
        for ii in range(self.n_components_):
            this_source = 'ICA %03d' % (ii + 1)
            ch_names.append(this_source)
            ch_info.append(dict(ch_name=this_source, cal=1,
                                logno=ii + 1, coil_type=FIFF.FIFFV_COIL_NONE,
                                kind=FIFF.FIFFV_MISC_CH,
                                coord_Frame=FIFF.FIFFV_COORD_UNKNOWN,
                                loc=np.array([0., 0., 0., 1.] * 3, dtype='f4'),
                                unit=FIFF.FIFF_UNIT_NONE,
                                range=1.0, scanno=ii + 1, unit_mul=0))

        if add_channels is not None:
            # re-append additionally picked ch_names
            ch_names += add_channels
            # re-append additionally picked ch_info
            ch_info += [k for k in container.info['chs'] if k['ch_name'] in
                        add_channels]
        info['bads'] = [ch_names[k] for k in self.exclude]
        info['projs'] = []  # make sure projections are removed.
        info._update_redundant()
        info._check_consistency()

    @verbose
    def score_sources(self, inst, target=None, score_func='pearsonr',
                      start=None, stop=None, l_freq=None, h_freq=None,
                      verbose=None):
        """Assign score to components based on statistic or metric.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            The object to reconstruct the sources from.
        target : array-like | ch_name | None
            Signal to which the sources shall be compared. It has to be of
            the same shape as the sources. If some string is supplied, a
            routine will try to find a matching channel. If None, a score
            function expecting only one input-array argument must be used,
            for instance, scipy.stats.skew (default).
        score_func : callable | str label
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
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        scores : ndarray
            scores for each source as returned from score_func
        """
        if isinstance(inst, BaseRaw):
            sources = self._transform_raw(inst, start, stop)
        elif isinstance(inst, BaseEpochs):
            sources = self._transform_epochs(inst, concatenate=True)
        elif isinstance(inst, Evoked):
            sources = self._transform_evoked(inst)
        else:
            raise ValueError('Input must be of Raw, Epochs or Evoked type')

        if target is not None:  # we can have univariate metrics without target
            target = self._check_target(target, inst, start, stop)

            if sources.shape[-1] != target.shape[-1]:
                raise ValueError('Sources and target do not have the same'
                                 'number of time slices.')
            # auto target selection
            if verbose is None:
                verbose = self.verbose
            if isinstance(inst, BaseRaw):
                sources, target = _band_pass_filter(self, sources, target,
                                                    l_freq, h_freq, verbose)

        scores = _find_sources(sources, target, score_func)

        return scores

    def _check_target(self, target, inst, start, stop):
        """Aux Method."""
        if isinstance(inst, BaseRaw):
            start, stop = _check_start_stop(inst, start, stop)
            if hasattr(target, 'ndim'):
                if target.ndim < 2:
                    target = target.reshape(1, target.shape[-1])
            if isinstance(target, string_types):
                pick = _get_target_ch(inst, target)
                target, _ = inst[pick, start:stop]

        elif isinstance(inst, BaseEpochs):
            if isinstance(target, string_types):
                pick = _get_target_ch(inst, target)
                target = inst.get_data()[:, pick]

            if hasattr(target, 'ndim'):
                if target.ndim == 3 and min(target.shape) == 1:
                    target = target.ravel()

        elif isinstance(inst, Evoked):
            if isinstance(target, string_types):
                pick = _get_target_ch(inst, target)
                target = inst.data[pick]

        return target

    @verbose
    def find_bads_ecg(self, inst, ch_name=None, threshold=None,
                      start=None, stop=None, l_freq=8, h_freq=16,
                      method='ctps', verbose=None):
        """Detect ECG related components using correlation.

        Note. If no ECG channel is available, routine attempts to create
        an artificial ECG based on cross-channel averaging.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Object to compute sources from.
        ch_name : str
            The name of the channel to use for ECG peak detection.
            The argument is mandatory if the dataset contains no ECG
            channels.
        threshold : float
            The value above which a feature is classified as outlier. If
            method is 'ctps', defaults to 0.25, else defaults to 3.0.
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
        method : {'ctps', 'correlation'}
            The method used for detection. If 'ctps', cross-trial phase
            statistics [1] are used to detect ECG related components.
            Thresholding is then based on the significance value of a Kuiper
            statistic.
            If 'correlation', detection is based on Pearson correlation
            between the filtered data and the filtered ECG channel.
            Thresholding is based on iterative z-scoring. The above
            threshold components will be masked and the z-score will
            be recomputed until no supra-threshold component remains.
            Defaults to 'ctps'.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        ecg_idx : list of int
            The indices of ECG related components.
        scores : np.ndarray of float, shape (``n_components_``)
            The correlation scores.

        See also
        --------
        find_bads_eog

        References
        ----------
        [1] Dammers, J., Schiek, M., Boers, F., Silex, C., Zvyagintsev,
            M., Pietrzyk, U., Mathiak, K., 2008. Integration of amplitude
            and phase statistics for complete artifact removal in independent
            components of neuromagnetic recordings. Biomedical
            Engineering, IEEE Transactions on 55 (10), 2353-2362.
        """
        if verbose is None:
            verbose = self.verbose

        idx_ecg = _get_ecg_channel_index(ch_name, inst)

        if idx_ecg is None:
            if verbose is not None:
                verbose = self.verbose
            ecg, times = _make_ecg(inst, start, stop, verbose)
            ch_name = 'ECG-MAG'
        else:
            ecg = inst.ch_names[idx_ecg]

        if method == 'ctps':
            if threshold is None:
                threshold = 0.25
            if isinstance(inst, BaseRaw):
                sources = self.get_sources(create_ecg_epochs(inst)).get_data()
            elif isinstance(inst, BaseEpochs):
                sources = self.get_sources(inst).get_data()
            else:
                raise ValueError('With `ctps` only Raw and Epochs input is '
                                 'supported')
            _, p_vals, _ = ctps(sources)
            scores = p_vals.max(-1)
            ecg_idx = np.where(scores >= threshold)[0]
        elif method == 'correlation':
            if threshold is None:
                threshold = 3.0
            scores = self.score_sources(inst, target=ecg,
                                        score_func='pearsonr',
                                        start=start, stop=stop,
                                        l_freq=l_freq, h_freq=h_freq,
                                        verbose=verbose)
            ecg_idx = find_outliers(scores, threshold=threshold)
        else:
            raise ValueError('Method "%s" not supported.' % method)
        # sort indices by scores
        ecg_idx = ecg_idx[np.abs(scores[ecg_idx]).argsort()[::-1]]

        self.labels_['ecg'] = list(ecg_idx)
        self.labels_['ecg/%s' % ch_name] = list(ecg_idx)
        return self.labels_['ecg'], scores

    @verbose
    def find_bads_eog(self, inst, ch_name=None, threshold=3.0,
                      start=None, stop=None, l_freq=1, h_freq=10,
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
        threshold : int | float
            The value above which a feature is classified as outlier.
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
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        eog_idx : list of int
            The indices of EOG related components, sorted by score.
        scores : np.ndarray of float, shape (``n_components_``) | list of array
            The correlation scores.

        See Also
        --------
        find_bads_ecg
        """
        if verbose is None:
            verbose = self.verbose

        eog_inds = _get_eog_channel_index(ch_name, inst)
        if len(eog_inds) > 2:
            eog_inds = eog_inds[:1]
            logger.info('Using EOG channel %s' % inst.ch_names[eog_inds[0]])
        scores, eog_idx = [], []
        eog_chs = [inst.ch_names[k] for k in eog_inds]

        # some magic we need inevitably ...
        # get targets befor equalizing
        targets = [self._check_target(k, inst, start, stop) for k in eog_chs]

        for ii, (eog_ch, target) in enumerate(zip(eog_chs, targets)):
            scores += [self.score_sources(inst, target=target,
                                          score_func='pearsonr',
                                          start=start, stop=stop,
                                          l_freq=l_freq, h_freq=h_freq,
                                          verbose=verbose)]
            # pick last scores
            this_idx = find_outliers(scores[-1], threshold=threshold)
            eog_idx += [this_idx]
            self.labels_[('eog/%i/' % ii) + eog_ch] = list(this_idx)

        # remove duplicates but keep order by score, even across multiple
        # EOG channels
        scores_ = np.concatenate([scores[ii][inds]
                                  for ii, inds in enumerate(eog_idx)])
        eog_idx_ = np.concatenate(eog_idx)[np.abs(scores_).argsort()[::-1]]

        eog_idx_unique = list(np.unique(eog_idx_))
        eog_idx = []
        for i in eog_idx_:
            if i in eog_idx_unique:
                eog_idx.append(i)
                eog_idx_unique.remove(i)
        if len(scores) == 1:
            scores = scores[0]
        self.labels_['eog'] = list(eog_idx)

        return self.labels_['eog'], scores

    def apply(self, inst, include=None, exclude=None, n_pca_components=None,
              start=None, stop=None):
        """Remove selected components from the signal.

        Given the unmixing matrix, transform data,
        zero out components, and inverse transform the data.
        This procedure will reconstruct M/EEG signals from which
        the dynamics described by the excluded components is subtracted.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            The data to be processed. The instance is modified inplace.
        include : array_like of int.
            The indices referring to columns in the ummixing matrix. The
            components to be kept.
        exclude : array_like of int.
            The indices referring to columns in the ummixing matrix. The
            components to be zeroed out.
        n_pca_components : int | float | None
            The number of PCA components to be kept, either absolute (int)
            or percentage of the explained variance (float). If None (default),
            all PCA components will be used.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
        """
        if isinstance(inst, BaseRaw):
            out = self._apply_raw(raw=inst, include=include,
                                  exclude=exclude,
                                  n_pca_components=n_pca_components,
                                  start=start, stop=stop)
        elif isinstance(inst, BaseEpochs):
            out = self._apply_epochs(epochs=inst, include=include,
                                     exclude=exclude,
                                     n_pca_components=n_pca_components)
        elif isinstance(inst, Evoked):
            out = self._apply_evoked(evoked=inst, include=include,
                                     exclude=exclude,
                                     n_pca_components=n_pca_components)
        else:
            raise ValueError('Data input must be of Raw, Epochs or Evoked '
                             'type')
        return out

    def _apply_raw(self, raw, include, exclude, n_pca_components, start, stop):
        """Aux method."""
        if not raw.preload:
            raise ValueError('Raw data must be preloaded to apply ICA')

        if exclude is None:
            exclude = list(set(self.exclude))
        else:
            exclude = list(set(self.exclude + exclude))

        if n_pca_components is not None:
            self.n_pca_components = n_pca_components

        start, stop = _check_start_stop(raw, start, stop)

        picks = pick_types(raw.info, meg=False, include=self.ch_names,
                           exclude='bads', ref_meg=False)

        data = raw[picks, start:stop][0]
        data, _ = self._pre_whiten(data, raw.info, picks)

        data = self._pick_sources(data, include, exclude)

        raw[picks, start:stop] = data
        return raw

    def _apply_epochs(self, epochs, include, exclude, n_pca_components):
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

        if n_pca_components is not None:
            self.n_pca_components = n_pca_components

        data = np.hstack(epochs.get_data()[:, picks])
        data, _ = self._pre_whiten(data, epochs.info, picks)
        data = self._pick_sources(data, include=include, exclude=exclude)

        # restore epochs, channels, tsl order
        epochs._data[:, picks] = np.array(np.split(data,
                                          len(epochs.events), 1))
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

        if n_pca_components is not None:
            self.n_pca_components = n_pca_components

        data = evoked.data[picks]
        data, _ = self._pre_whiten(data, evoked.info, picks)
        data = self._pick_sources(data, include=include,
                                  exclude=exclude)

        # restore evoked
        evoked.data[picks] = data

        return evoked

    def _pick_sources(self, data, include, exclude):
        """Aux function."""
        fast_dot = _get_fast_dot()
        if exclude is None:
            exclude = self.exclude
        else:
            exclude = list(set(self.exclude + list(exclude)))

        _n_pca_comp = self._check_n_pca_components(self.n_pca_components)

        if not(self.n_components_ <= _n_pca_comp <= self.max_pca_components):
            raise ValueError('n_pca_components must be >= '
                             'n_components and <= max_pca_components.')

        n_components = self.n_components_
        logger.info('Transforming to ICA space (%i components)' % n_components)

        # Apply first PCA
        if self.pca_mean_ is not None:
            data -= self.pca_mean_[:, None]

        sel_keep = np.arange(n_components)
        if include not in (None, []):
            sel_keep = np.unique(include)
        elif exclude not in (None, []):
            sel_keep = np.setdiff1d(np.arange(n_components), exclude)

        logger.info('Zeroing out %i ICA components'
                    % (n_components - len(sel_keep)))

        unmixing = np.eye(_n_pca_comp)
        unmixing[:n_components, :n_components] = self.unmixing_matrix_
        unmixing = np.dot(unmixing, self.pca_components_[:_n_pca_comp])

        mixing = np.eye(_n_pca_comp)
        mixing[:n_components, :n_components] = self.mixing_matrix_
        mixing = np.dot(self.pca_components_[:_n_pca_comp].T, mixing)

        if _n_pca_comp > n_components:
            sel_keep = np.concatenate(
                (sel_keep, range(n_components, _n_pca_comp)))

        proj_mat = np.dot(mixing[:, sel_keep], unmixing[sel_keep, :])

        data = fast_dot(proj_mat, data)

        if self.pca_mean_ is not None:
            data += self.pca_mean_[:, None]

        # restore scaling
        if self.noise_cov is None:  # revert standardization
            data *= self._pre_whitener
        else:
            data = fast_dot(linalg.pinv(self._pre_whitener), data)

        return data

    @verbose
    def save(self, fname):
        """Store ICA solution into a fiff file.

        Parameters
        ----------
        fname : str
            The absolute path of the file name to save the ICA solution into.
            The file name should end with -ica.fif or -ica.fif.gz.
        """
        if self.current_fit == 'unfitted':
            raise RuntimeError('No fit available. Please first fit ICA')

        check_fname(fname, 'ICA', ('-ica.fif', '-ica.fif.gz'))

        logger.info('Writing ica solution to %s...' % fname)
        fid = start_file(fname)

        try:
            _write_ica(fid, self)
        except Exception:
            os.remove(fname)
            raise
        end_file(fid)

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
    def plot_components(self, picks=None, ch_type=None, res=64, layout=None,
                        vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                        colorbar=False, title=None, show=True, outlines='head',
                        contours=6, image_interp='bilinear', head_pos=None,
                        inst=None):
        return plot_ica_components(self, picks=picks, ch_type=ch_type,
                                   res=res, layout=layout, vmin=vmin,
                                   vmax=vmax, cmap=cmap, sensors=sensors,
                                   colorbar=colorbar, title=title, show=show,
                                   outlines=outlines, contours=contours,
                                   image_interp=image_interp,
                                   head_pos=head_pos, inst=inst)

    @copy_function_doc_to_method_doc(plot_ica_properties)
    def plot_properties(self, inst, picks=None, axes=None, dB=True,
                        plot_std=True, topomap_args=None, image_args=None,
                        psd_args=None, figsize=None, show=True):
        return plot_ica_properties(self, inst, picks=picks, axes=axes,
                                   dB=dB, plot_std=plot_std,
                                   topomap_args=topomap_args,
                                   image_args=image_args, psd_args=psd_args,
                                   figsize=figsize, show=show)

    @copy_function_doc_to_method_doc(plot_ica_sources)
    def plot_sources(self, inst, picks=None, exclude=None, start=None,
                     stop=None, title=None, show=True, block=False,
                     show_first_samp=False):
        return plot_ica_sources(self, inst=inst, picks=picks, exclude=exclude,
                                start=start, stop=stop, title=title, show=show,
                                block=block, show_first_samp=show_first_samp)

    @copy_function_doc_to_method_doc(plot_ica_scores)
    def plot_scores(self, scores, exclude=None, labels=None, axhline=None,
                    title='ICA component scores', figsize=(12, 6),
                    show=True):
        return plot_ica_scores(
            ica=self, scores=scores, exclude=exclude, labels=labels,
            axhline=axhline, title=title, figsize=figsize, show=show)

    @copy_function_doc_to_method_doc(plot_ica_overlay)
    def plot_overlay(self, inst, exclude=None, picks=None, start=None,
                     stop=None, title=None, show=True):
        return plot_ica_overlay(self, inst=inst, exclude=exclude, picks=picks,
                                start=start, stop=stop, title=title, show=show)

    def detect_artifacts(self, raw, start_find=None, stop_find=None,
                         ecg_ch=None, ecg_score_func='pearsonr',
                         ecg_criterion=0.1, eog_ch=None,
                         eog_score_func='pearsonr',
                         eog_criterion=0.1, skew_criterion=-1,
                         kurt_criterion=-1, var_criterion=0,
                         add_nodes=None):
        """Run ICA artifacts detection workflow.

        Note. This is still experimental and will most likely change. Over
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
            Raw object to draw sources from.
        start_find : int | float | None
            First sample to include for artifact search. If float, data will be
            interpreted as time in seconds. If None, data will be used from the
            first sample.
        stop_find : int | float | None
            Last sample to not include for artifact search. If float, data will
            be interpreted as time in seconds. If None, data will be used to
            the last sample.
        ecg_ch : str | ndarray | None
            The `target` argument passed to ica.find_sources_raw. Either the
            name of the ECG channel or the ECG time series. If None, this step
            will be skipped.
        ecg_score_func : str | callable
            The `score_func` argument passed to ica.find_sources_raw. Either
            the name of function supported by ICA or a custom function.
        ecg_criterion : float | int | list-like | slice
            The indices of the sorted skewness scores. If float, sources with
            scores smaller than the criterion will be dropped. Else, the scores
            sorted in descending order will be indexed accordingly.
            E.g. range(2) would return the two sources with the highest score.
            If None, this step will be skipped.
        eog_ch : list | str | ndarray | None
            The `target` argument or the list of target arguments subsequently
            passed to ica.find_sources_raw. Either the name of the vertical EOG
            channel or the corresponding EOG time series. If None, this step
            will be skipped.
        eog_score_func : str | callable
            The `score_func` argument passed to ica.find_sources_raw. Either
            the name of function supported by ICA or a custom function.
        eog_criterion : float | int | list-like | slice
            The indices of the sorted skewness scores. If float, sources with
            scores smaller than the criterion will be dropped. Else, the scores
            sorted in descending order will be indexed accordingly.
            E.g. range(2) would return the two sources with the highest score.
            If None, this step will be skipped.
        skew_criterion : float | int | list-like | slice
            The indices of the sorted skewness scores. If float, sources with
            scores smaller than the criterion will be dropped. Else, the scores
            sorted in descending order will be indexed accordingly.
            E.g. range(2) would return the two sources with the highest score.
            If None, this step will be skipped.
        kurt_criterion : float | int | list-like | slice
            The indices of the sorted skewness scores. If float, sources with
            scores smaller than the criterion will be dropped. Else, the scores
            sorted in descending order will be indexed accordingly.
            E.g. range(2) would return the two sources with the highest score.
            If None, this step will be skipped.
        var_criterion : float | int | list-like | slice
            The indices of the sorted skewness scores. If float, sources with
            scores smaller than the criterion will be dropped. Else, the scores
            sorted in descending order will be indexed accordingly.
            E.g. range(2) would return the two sources with the highest score.
            If None, this step will be skipped.
        add_nodes : list of ica_nodes
            Additional list if tuples carrying the following parameters:
            (name : str, target : str | array, score_func : callable,
            criterion : float | int | list-like | slice). This parameter is a
            generalization of the artifact specific parameters above and has
            the same structure. Example:
            add_nodes=('ECG phase lock', ECG 01', my_phase_lock_function, 0.5)

        Returns
        -------
        self : instance of ICA
            The ica object with the detected artifact indices marked for
            exclusion
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
            _n_pca_comp = ((self.pca_explained_variance_ /
                           self.pca_explained_variance_.sum()).cumsum() <=
                           _n_pca_comp).sum()
            logger.info('Selected %i PCA components by explained '
                        'variance' % _n_pca_comp)
        elif _n_pca_comp is None:
            _n_pca_comp = self.max_pca_components
        elif _n_pca_comp < self.n_components_:
            _n_pca_comp = self.n_components_

        return _n_pca_comp


def _check_start_stop(raw, start, stop):
    """Aux function."""
    return [c if (isinstance(c, int) or c is None) else
            raw.time_as_index(c)[0] for c in (start, stop)]


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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    eog_events : array
        Events
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
    if isinstance(score_func, string_types):
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
    var = np.sum(ica.mixing_matrix_**2, axis=0) * np.sum(
        source_data**2, axis=1) / (n_chan * n_samp - 1)
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
        elif isinstance(value, int):
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
        vv = json.loads(v)
        out[k] = vv if not isinstance(vv, text_type) else str(vv)

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
                    max_pca_components=ica.max_pca_components,
                    current_fit=ica.current_fit)

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
                'method': getattr(ica, 'method', None)}

    write_string(fid, FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS,
                 _serialize(ica_init))

    #   ICA misct params
    write_string(fid, FIFF.FIFF_MNE_ICA_MISC_PARAMS,
                 _serialize(ica_misc))

    #   Whitener
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_WHITENER, ica._pre_whitener)

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

    write_int(fid, FIFF.FIFF_MNE_ICA_BADS, ica.exclude)

    # Done!
    end_block(fid, FIFF.FIFFB_MNE_ICA)


@verbose
def read_ica(fname):
    """Restore ICA solution from fif file.

    Parameters
    ----------
    fname : str
        Absolute path to fif file containing ICA matrices.
        The file name should end with -ica.fif or -ica.fif.gz.

    Returns
    -------
    ica : instance of ICA
        The ICA estimator.
    """
    check_fname(fname, 'ICA', ('-ica.fif', '-ica.fif.gz'))

    logger.info('Reading %s ...' % fname)
    fid, tree, _ = fiff_open(fname)

    try:
        # we used to store bads that weren't part of the info...
        info, meas = read_meas_info(fid, tree, clean_bads=True)
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
    current_fit = ica_init.pop('current_fit')
    if ica_init['noise_cov'] == Covariance.__name__:
        logger.info('Reading whitener drawn from noise covariance ...')

    logger.info('Now restoring ICA solution ...')

    # make sure dtypes are np.float64 to satisfy fast_dot
    def f(x):
        return x.astype(np.float64)

    ica_init = dict((k, v) for k, v in ica_init.items()
                    if k in _get_args(ICA.__init__))
    ica = ICA(**ica_init)
    ica.current_fit = current_fit
    ica.ch_names = ch_names.split(':')
    ica._pre_whitener = f(pre_whitener)
    ica.pca_mean_ = f(pca_mean)
    ica.pca_components_ = f(pca_components)
    ica.n_components_ = unmixing_matrix.shape[0]
    ica.pca_explained_variance_ = f(pca_explained_variance)
    ica.unmixing_matrix_ = f(unmixing_matrix)
    ica.mixing_matrix_ = linalg.pinv(ica.unmixing_matrix_)
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
            found = list(np.atleast_1d(abs(scores).argsort()[node.criterion]))

        case = (len(found), _pl(found), node.name)
        logger.info('    found %s artifact%s by %s' % case)
        ica.exclude += found

    logger.info('Artifact indices found:\n    ' + str(ica.exclude).strip('[]'))
    if len(set(ica.exclude)) != len(ica.exclude):
        logger.info('    Removing duplicate indices...')
        ica.exclude = list(set(ica.exclude))

    logger.info('Ready.')


@verbose
def run_ica(raw, n_components, max_pca_components=100,
            n_pca_components=64, noise_cov=None, random_state=None,
            picks=None, start=None, stop=None, start_find=None,
            stop_find=None, ecg_ch=None, ecg_score_func='pearsonr',
            ecg_criterion=0.1, eog_ch=None, eog_score_func='pearsonr',
            eog_criterion=0.1, skew_criterion=-1, kurt_criterion=-1,
            var_criterion=0, add_nodes=None, verbose=None):
    """Run ICA decomposition on raw data and identify artifact sources.

    This function implements an automated artifact removal work flow.

    Hints and caveats:

        - It is highly recommended to bandpass filter ECG and EOG
          data and pass them instead of the channel names as ecg_ch and eog_ch
          arguments.
        - Please check your results. Detection by kurtosis and variance
          can be powerful but misclassification of brain signals as
          noise cannot be precluded. If you are not sure set those to None.
        - Consider using shorter times for start_find and stop_find than
          for start and stop. It can save you much time.

    Example invocation (taking advantage of defaults)::

        ica = run_ica(raw, n_components=.9, start_find=10000, stop_find=12000,
                      ecg_ch='MEG 1531', eog_ch='EOG 061')

    Parameters
    ----------
    raw : instance of Raw
        The raw data to decompose.
    n_components : int | float | None
        The number of components used for ICA decomposition. If int, it must be
        smaller then max_pca_components. If None, all PCA components will be
        used. If float between 0 and 1 components can will be selected by the
        cumulative percentage of explained variance.
    max_pca_components : int | None
        The number of components used for PCA decomposition. If None, no
        dimension reduction will be applied and max_pca_components will equal
        the number of channels supplied on decomposing data.
    n_pca_components
        The number of PCA components used after ICA recomposition. The ensuing
        attribute allows to balance noise reduction against potential loss of
        features due to dimensionality reduction. If greater than
        ``self.n_components_``, the next ``'n_pca_components'`` minus
        ``'n_components_'`` PCA components will be added before restoring the
        sensor space data. The attribute gets updated each time the according
        parameter for in .pick_sources_raw or .pick_sources_epochs is changed.
    noise_cov : None | instance of mne.cov.Covariance
        Noise covariance used for whitening. If None, channels are just
        z-scored.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the FastICA estimation.
        As the estimation is non-deterministic it can be useful to
        fix the seed to have reproducible results.
    picks : array-like of int
        Channels to be included. This selection remains throughout the
        initialized ICA solution. If None only good data channels are used.
    start : int | float | None
        First sample to include for decomposition. If float, data will be
        interpreted as time in seconds. If None, data will be used from the
        first sample.
    stop : int | float | None
        Last sample to not include for decomposition. If float, data will be
        interpreted as time in seconds. If None, data will be used to the
        last sample.
    start_find : int | float | None
        First sample to include for artifact search. If float, data will be
        interpreted as time in seconds. If None, data will be used from the
        first sample.
    stop_find : int | float | None
        Last sample to not include for artifact search. If float, data will be
        interpreted as time in seconds. If None, data will be used to the last
        sample.
    ecg_ch : str | ndarray | None
        The ``target`` argument passed to ica.find_sources_raw. Either the
        name of the ECG channel or the ECG time series. If None, this step
        will be skipped.
    ecg_score_func : str | callable
        The ``score_func`` argument passed to ica.find_sources_raw. Either
        the name of function supported by ICA or a custom function.
    ecg_criterion : float | int | list-like | slice
        The indices of the sorted skewness scores. If float, sources with
        scores smaller than the criterion will be dropped. Else, the scores
        sorted in descending order will be indexed accordingly.
        E.g. range(2) would return the two sources with the highest score.
        If None, this step will be skipped.
    eog_ch : list | str | ndarray | None
        The ``target`` argument or the list of target arguments subsequently
        passed to ica.find_sources_raw. Either the name of the vertical EOG
        channel or the corresponding EOG time series. If None, this step
        will be skipped.
    eog_score_func : str | callable
        The ``score_func`` argument passed to ica.find_sources_raw. Either
        the name of function supported by ICA or a custom function.
    eog_criterion : float | int | list-like | slice
        The indices of the sorted skewness scores. If float, sources with
        scores smaller than the criterion will be dropped. Else, the scores
        sorted in descending order will be indexed accordingly.
        E.g. range(2) would return the two sources with the highest score.
        If None, this step will be skipped.
    skew_criterion : float | int | list-like | slice
        The indices of the sorted skewness scores. If float, sources with
        scores smaller than the criterion will be dropped. Else, the scores
        sorted in descending order will be indexed accordingly.
        E.g. range(2) would return the two sources with the highest score.
        If None, this step will be skipped.
    kurt_criterion : float | int | list-like | slice
        The indices of the sorted skewness scores. If float, sources with
        scores smaller than the criterion will be dropped. Else, the scores
        sorted in descending order will be indexed accordingly.
        E.g. range(2) would return the two sources with the highest score.
        If None, this step will be skipped.
    var_criterion : float | int | list-like | slice
        The indices of the sorted skewness scores. If float, sources with
        scores smaller than the criterion will be dropped. Else, the scores
        sorted in descending order will be indexed accordingly.
        E.g. range(2) would return the two sources with the highest score.
        If None, this step will be skipped.
    add_nodes : list of ica_nodes
        Additional list if tuples carrying the following parameters:
        (name : str, target : str | array, score_func : callable,
        criterion : float | int | list-like | slice). This parameter is a
        generalization of the artifact specific parameters above and has
        the same structure. Example::

            add_nodes=('ECG phase lock', ECG 01', my_phase_lock_function, 0.5)

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    ica : instance of ICA
        The ica object with detected artifact sources marked for exclusion
    """
    ica = ICA(n_components=n_components, max_pca_components=max_pca_components,
              n_pca_components=n_pca_components, noise_cov=noise_cov,
              random_state=random_state, verbose=verbose)

    ica.fit(raw, start=start, stop=stop, picks=picks)
    logger.info('%s' % ica)
    logger.info('    Now searching for artifacts...')

    _detect_artifacts(ica=ica, raw=raw, start_find=start_find,
                      stop_find=stop_find, ecg_ch=ecg_ch,
                      ecg_score_func=ecg_score_func,
                      ecg_criterion=ecg_criterion, eog_ch=eog_ch,
                      eog_score_func=eog_score_func,
                      eog_criterion=eog_criterion,
                      skew_criterion=skew_criterion,
                      kurt_criterion=kurt_criterion,
                      var_criterion=var_criterion,
                      add_nodes=add_nodes)
    return ica


@verbose
def _band_pass_filter(ica, sources, target, l_freq, h_freq, verbose=None):
    """Optionally band-pass filter the data."""
    if l_freq is not None and h_freq is not None:
        logger.info('... filtering ICA sources')
        # use FIR here, steeper is better
        kw = dict(phase='zero-double', filter_length='10s', fir_window='hann',
                  l_trans_bandwidth=0.5, h_trans_bandwidth=0.5)
        sources = filter_data(sources, ica.info['sfreq'], l_freq, h_freq, **kw)
        logger.info('... filtering target')
        target = filter_data(target, ica.info['sfreq'], l_freq, h_freq, **kw)
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
        max_corrs = [list(find_outliers(s_corr, threshold=threshold))
                     for s_corr in abs_corrs]

    am = [l[i] for l, i_s in zip(abs_corrs, max_corrs)
          for i in i_s]
    median_corr_with_target = np.median(am) if len(am) > 0 else 0

    polarities = [l[i] for l, i_s in zip(corr_polarities, max_corrs)
                  for i in i_s]

    maxmaps = [l[i] for l, i_s in zip(all_maps, max_corrs)
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


def _plot_corrmap(data, subjs, indices, ch_type, ica, label, show, outlines,
                  layout, cmap, contours, template=True):
    """Customized ica.plot_components for corrmap."""
    if not template:
        title = 'Detected components'
        if label is not None:
            title += ' of type ' + label
    else:
        title = "Supplied template"

    picks = list(range(len(data)))

    p = 20
    if len(picks) > p:  # plot components by sets of 20
        n_components = len(picks)
        figs = [_plot_corrmap(data[k:k + p], subjs[k:k + p],
                indices[k:k + p], ch_type, ica, label, show,
                outlines=outlines, layout=layout, cmap=cmap,
                contours=contours)
                for k in range(0, n_components, p)]
        return figs
    elif np.isscalar(picks):
        picks = [picks]

    data_picks, pos, merge_grads, names, _ = _prepare_topo_plot(
        ica, ch_type, layout)
    pos, outlines = _check_outlines(pos, outlines)

    data = np.atleast_2d(data)
    data = data[:, data_picks]

    # prepare data for iteration
    fig, axes = _prepare_trellis(len(picks), max_col=5)
    fig.suptitle(title)

    if merge_grads:
        from ..channels.layout import _merge_grad_data
    for ii, data_, ax, subject, idx in zip(picks, data, axes, subjs, indices):
        if template:
            ttl = 'Subj. {0}, IC {1}'.format(subject, idx)
            ax.set_title(ttl, fontsize=12)
        data_ = _merge_grad_data(data_) if merge_grads else data_
        vmin_, vmax_ = _setup_vmin_vmax(data_, None, None)
        plot_topomap(data_.flatten(), pos, vmin=vmin_, vmax=vmax_,
                     res=64, axes=ax, cmap=cmap, outlines=outlines,
                     image_mask=None, contours=contours, show=False,
                     image_interp='bilinear')[0]
        _hide_frame(ax)
    tight_layout(fig=fig)
    fig.subplots_adjust(top=0.8)
    fig.canvas.draw()
    plt_show(show)
    return fig


@verbose
def corrmap(icas, template, threshold="auto", label=None, ch_type="eeg",
            plot=True, show=True, verbose=None, outlines='head', layout=None,
            sensors=True, contours=6, cmap=None):
    """Find similar Independent Components across subjects by map similarity.

    Corrmap (Viola et al. 2009 Clin Neurophysiol) identifies the best group
    match to a supplied template. Typically, feed it a list of fitted ICAs and
    a template IC, for example, the blink for the first subject, to identify
    specific ICs across subjects.

    The specific procedure consists of two iterations. In a first step, the
    maps best correlating with the template are identified. In the step, the
    analysis is repeated with the mean of the maps identified in the first
    stage.

    Run with `plot` and `show` set to `True` and `label=False` to find
    good parameters. Then, run with labelling enabled to apply the
    labelling in the IC objects. (Running with both `plot` and `labels`
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
        If float > 1, use find_outliers to identify ICs within subjects (not in
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    outlines : 'head' | dict | None
        The outlines to be drawn. If 'head', a head scheme will be drawn. If
        dict, each key refers to a tuple of x and y positions. The values in
        'mask_pos' will serve as image mask. If None, nothing will be drawn.
        Defaults to 'head'. If dict, the 'autoshrink' (bool) field will
        trigger automated shrinking of the positions due to points outside the
        outline. Moreover, a matplotlib patch object can be passed for
        advanced masking options, either directly or as a function that returns
        patches (required for multi-axis plots).
    layout : None | Layout | list of Layout
        Layout instance specifying sensor positions (does not need to be
        specified for Neuromag data). Or a list of Layout if projections
        are from different sensor types.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses). If True, a circle will be
        used (via .add_artist). Defaults to True.
    contours : int | False | None
        The number of contour lines to draw. If 0, no contours will be drawn.
    cmap : None | matplotlib colormap
        Colormap for the plot. If ``None``, defaults to 'Reds_r' for norm data,
        otherwise to 'RdBu_r'.

    Returns
    -------
    template_fig : fig
        Figure showing the template.
    labelled_ics : fig
        Figure showing the labelled ICs in all ICA decompositions.
    """
    if not isinstance(plot, bool):
        raise ValueError("`plot` must be of type `bool`")

    if threshold == 'auto':
        threshold = np.arange(60, 95, dtype=np.float64) / 100.

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
            ttl = 'Template from subj. {0}'.format(str(template[0]))
            template_fig = icas[template[0]].plot_components(
                picks=template[1], ch_type=ch_type, title=ttl,
                outlines=outlines, cmap=cmap, contours=contours, layout=layout,
                show=show)
        else:  # plotting an array
            template_fig = _plot_corrmap([template], [0], [0], ch_type,
                                         icas[0].copy(), "Template",
                                         outlines=outlines, cmap=cmap,
                                         contours=contours, layout=layout,
                                         show=show, template=True)
        template_fig.subplots_adjust(top=0.8)
        template_fig.canvas.draw()

    # first run: use user-selected map
    if isinstance(threshold, (int, float)):
        if len(all_maps) == 0:
            logger.info('No component detected using find_outliers.'
                        ' Consider using threshold="auto"')
            return icas
        nt, mt, s, mx = _find_max_corrs(all_maps, target, threshold)
    elif len(threshold) > 1:
        paths = [_find_max_corrs(all_maps, target, t) for t in threshold]
        # find iteration with highest avg correlation with target
        nt, mt, s, mx = paths[np.argmax([path[2] for path in paths])]

    # second run: use output from first run
    if isinstance(threshold, (int, float)):
        if len(all_maps) == 0 or len(nt) == 0:
            if threshold > 1:
                logger.info('No component detected using find_outliers. '
                            'Consider using threshold="auto"')
            return icas
        nt, mt, s, mx = _find_max_corrs(all_maps, nt, threshold)
    elif len(threshold) > 1:
        paths = [_find_max_corrs(all_maps, nt, t) for t in threshold]
        # find iteration with highest avg correlation with target
        nt, mt, s, mx = paths[np.argmax([path[1] for path in paths])]

    allmaps, indices, subjs, nones = [list() for _ in range(4)]
    logger.info('Median correlation with constructed map: %0.3f' % mt)
    if plot is True:
        logger.info('Displaying selected ICs per subject.')

    for ii, (ica, max_corr) in enumerate(zip(icas, mx)):
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
        logger.info('No maps selected for subject(s) ' +
                    ', '.join([str(x) for x in nones]) +
                    ', consider a more liberal threshold.')

    if plot is True:
        labelled_ics = _plot_corrmap(allmaps, subjs, indices, ch_type, ica,
                                     label, outlines=outlines, cmap=cmap,
                                     contours=contours, layout=layout,
                                     show=show)
        return template_fig, labelled_ics
    else:
        return None
