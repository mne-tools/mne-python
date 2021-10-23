# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Daniel Strohmeier <daniel.strohmeier@gmail.com>
#
# License: Simplified BSD

import numpy as np

from ..source_estimate import SourceEstimate, _BaseSourceEstimate, _make_stc
from ..minimum_norm.inverse import (combine_xyz, _prepare_forward,
                                    _check_reference, _log_exp_var)
from ..forward import is_fixed_orient
from ..io.pick import pick_channels_evoked
from ..io.proj import deactivate_proj
from ..utils import (logger, verbose, _check_depth, _check_option, sum_squared,
                     _validate_type, check_random_state, warn)
from ..dipole import Dipole

from .mxne_optim import (mixed_norm_solver, iterative_mixed_norm_solver, _Phi,
                         tf_mixed_norm_solver, iterative_tf_mixed_norm_solver,
                         norm_l2inf, norm_epsilon_inf, groups_norm2)


def _check_ori(pick_ori, forward):
    """Check pick_ori."""
    _check_option('pick_ori', pick_ori, [None, 'vector'])
    if pick_ori == 'vector' and is_fixed_orient(forward):
        raise ValueError('pick_ori="vector" cannot be combined with a fixed '
                         'orientation forward solution.')


def _prepare_weights(forward, gain, source_weighting, weights, weights_min):
    mask = None
    if isinstance(weights, _BaseSourceEstimate):
        weights = np.max(np.abs(weights.data), axis=1)
    weights_max = np.max(weights)
    if weights_min > weights_max:
        raise ValueError('weights_min > weights_max (%s > %s)' %
                         (weights_min, weights_max))
    weights_min = weights_min / weights_max
    weights = weights / weights_max
    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    weights = np.ravel(np.tile(weights, [n_dip_per_pos, 1]).T)
    if len(weights) != gain.shape[1]:
        raise ValueError('weights do not have the correct dimension '
                         ' (%d != %d)' % (len(weights), gain.shape[1]))
    if len(source_weighting.shape) == 1:
        source_weighting *= weights
    else:
        source_weighting *= weights[:, None]
    gain *= weights[None, :]

    if weights_min is not None:
        mask = (weights > weights_min)
        gain = gain[:, mask]
        n_sources = np.sum(mask) // n_dip_per_pos
        logger.info("Reducing source space to %d sources" % n_sources)

    return gain, source_weighting, mask


def _prepare_gain(forward, info, noise_cov, pca, depth, loose, rank,
                  weights=None, weights_min=None):
    depth = _check_depth(depth, 'depth_sparse')
    forward, gain_info, gain, _, _, source_weighting, _, _, whitener = \
        _prepare_forward(forward, info, noise_cov, 'auto', loose, rank, pca,
                         use_cps=True, **depth)

    if weights is None:
        mask = None
    else:
        gain, source_weighting, mask = _prepare_weights(
            forward, gain, source_weighting, weights, weights_min)

    return forward, gain, gain_info, whitener, source_weighting, mask


def _reapply_source_weighting(X, source_weighting, active_set):
    X *= source_weighting[active_set][:, None]
    return X


def _compute_residual(forward, evoked, X, active_set, info):
    # OK, picking based on row_names is safe
    sel = [forward['sol']['row_names'].index(c) for c in info['ch_names']]
    residual = evoked.copy()
    residual = pick_channels_evoked(residual, include=info['ch_names'])
    r_tmp = residual.copy()

    r_tmp.data = np.dot(forward['sol']['data'][sel, :][:, active_set], X)

    # Take care of proj
    active_projs = list()
    non_active_projs = list()
    for p in evoked.info['projs']:
        if p['active']:
            active_projs.append(p)
        else:
            non_active_projs.append(p)

    if len(active_projs) > 0:
        with r_tmp.info._unlock():
            r_tmp.info['projs'] = deactivate_proj(active_projs, copy=True,
                                                  verbose=False)
        r_tmp.apply_proj(verbose=False)
        r_tmp.add_proj(non_active_projs, remove_existing=False, verbose=False)

    residual.data -= r_tmp.data

    return residual


@verbose
def _make_sparse_stc(X, active_set, forward, tmin, tstep,
                     active_is_idx=False, pick_ori=None, verbose=None):
    source_nn = forward['source_nn']
    vector = False
    if not is_fixed_orient(forward):
        if pick_ori != 'vector':
            logger.info('combining the current components...')
            X = combine_xyz(X)
        else:
            vector = True
            source_nn = np.reshape(source_nn, (-1, 3, 3))

    if not active_is_idx:
        active_idx = np.where(active_set)[0]
    else:
        active_idx = active_set

    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    if n_dip_per_pos > 1:
        active_idx = np.unique(active_idx // n_dip_per_pos)

    src = forward['src']
    vertices = []
    n_points_so_far = 0
    for this_src in src:
        this_n_points_so_far = n_points_so_far + len(this_src['vertno'])
        this_active_idx = active_idx[(n_points_so_far <= active_idx) &
                                     (active_idx < this_n_points_so_far)]
        this_active_idx -= n_points_so_far
        this_vertno = this_src['vertno'][this_active_idx]
        n_points_so_far = this_n_points_so_far
        vertices.append(this_vertno)
    source_nn = source_nn[active_idx]
    return _make_stc(
        X, vertices, src.kind, tmin, tstep, src[0]['subject_his_id'],
        vector=vector, source_nn=source_nn)


def _split_gof(M, X, gain):
    # parse out the variance explained using an orthogonal basis
    # assuming x is estimated using elements of gain, with residual res
    # along the first axis
    assert M.ndim == X.ndim == gain.ndim == 2, (M.ndim, X.ndim, gain.ndim)
    assert gain.shape == (M.shape[0], X.shape[0])
    assert M.shape[1] == X.shape[1]
    norm = (M * M.conj()).real.sum(0, keepdims=True)
    norm[norm == 0] = np.inf
    M_est = gain @ X
    assert M.shape == M_est.shape
    res = M - M_est
    assert gain.shape[0] == M.shape[0], (gain.shape, M.shape)
    # find an orthonormal basis for our matrices that spans the actual data
    U, s, _ = np.linalg.svd(gain, full_matrices=False)
    U = U[:, s >= s[0] * 1e-6]
    # the part that gets explained
    fit_orth = U.T @ M
    # the part that got over-explained (landed in residual)
    res_orth = U.T @ res
    # determine the weights by projecting each one onto this basis
    w = (U.T @ gain)[:, :, np.newaxis] * X
    w_norm = np.linalg.norm(w, axis=1, keepdims=True)
    w_norm[w_norm == 0] = 1.
    w /= w_norm
    # our weights are now unit-norm positive (will presrve power)
    fit_back = np.linalg.norm(fit_orth[:, np.newaxis] * w, axis=0) ** 2
    res_back = np.linalg.norm(res_orth[:, np.newaxis] * w, axis=0) ** 2
    # and the resulting goodness of fits
    gof_back = 100 * (fit_back - res_back) / norm
    assert gof_back.shape == X.shape, (gof_back.shape, X.shape)
    return gof_back


@verbose
def _make_dipoles_sparse(X, active_set, forward, tmin, tstep, M,
                         gain_active, active_is_idx=False,
                         verbose=None):
    times = tmin + tstep * np.arange(X.shape[1])

    if not active_is_idx:
        active_idx = np.where(active_set)[0]
    else:
        active_idx = active_set

    # Compute the GOF split amongst the dipoles
    assert M.shape == (gain_active.shape[0], len(times))
    assert gain_active.shape[1] == len(active_idx) == X.shape[0]
    gof_split = _split_gof(M, X, gain_active)
    assert gof_split.shape == (len(active_idx), len(times))
    assert X.shape[0] in (len(active_idx), 3 * len(active_idx))

    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    if n_dip_per_pos > 1:
        active_idx = active_idx // n_dip_per_pos
        _, keep = np.unique(active_idx, return_index=True)
        keep.sort()  # maintain old order
        active_idx = active_idx[keep]
        gof_split.shape = (len(active_idx), n_dip_per_pos, len(times))
        gof_split = gof_split.sum(1)
        assert (gof_split < 100).all()
    assert gof_split.shape == (len(active_idx), len(times))

    dipoles = []
    for k, i_dip in enumerate(active_idx):
        i_pos = forward['source_rr'][i_dip][np.newaxis, :]
        i_pos = i_pos.repeat(len(times), axis=0)
        X_ = X[k * n_dip_per_pos: (k + 1) * n_dip_per_pos]
        if n_dip_per_pos == 1:
            amplitude = X_[0]
            i_ori = forward['source_nn'][i_dip][np.newaxis, :]
            i_ori = i_ori.repeat(len(times), axis=0)
        else:
            if forward['surf_ori']:
                X_ = np.dot(forward['source_nn'][
                    i_dip * n_dip_per_pos:(i_dip + 1) * n_dip_per_pos].T, X_)
            amplitude = np.linalg.norm(X_, axis=0)
            i_ori = np.zeros((len(times), 3))
            i_ori[amplitude > 0.] = (X_[:, amplitude > 0.] /
                                     amplitude[amplitude > 0.]).T

        dipoles.append(Dipole(times, i_pos, amplitude, i_ori, gof_split[k]))

    return dipoles


@verbose
def make_stc_from_dipoles(dipoles, src, verbose=None):
    """Convert a list of spatio-temporal dipoles into a SourceEstimate.

    Parameters
    ----------
    dipoles : Dipole | list of instances of Dipole
        The dipoles to convert.
    src : instance of SourceSpaces
        The source space used to generate the forward operator.
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate
        The source estimate.
    """
    logger.info('Converting dipoles into a SourceEstimate.')
    if isinstance(dipoles, Dipole):
        dipoles = [dipoles]
    if not isinstance(dipoles, list):
        raise ValueError('Dipoles must be an instance of Dipole or '
                         'a list of instances of Dipole. '
                         'Got %s!' % type(dipoles))
    tmin = dipoles[0].times[0]
    tstep = dipoles[0].times[1] - tmin
    X = np.zeros((len(dipoles), len(dipoles[0].times)))
    source_rr = np.concatenate([_src['rr'][_src['vertno'], :] for _src in src],
                               axis=0)
    n_lh_points = len(src[0]['vertno'])
    lh_vertno = list()
    rh_vertno = list()
    for i in range(len(dipoles)):
        if not np.all(dipoles[i].pos == dipoles[i].pos[0]):
            raise ValueError('Only dipoles with fixed position over time '
                             'are supported!')
        X[i] = dipoles[i].amplitude
        idx = np.all(source_rr == dipoles[i].pos[0], axis=1)
        idx = np.where(idx)[0][0]
        if idx < n_lh_points:
            lh_vertno.append(src[0]['vertno'][idx])
        else:
            rh_vertno.append(src[1]['vertno'][idx - n_lh_points])
    vertices = [np.array(lh_vertno).astype(int),
                np.array(rh_vertno).astype(int)]
    stc = SourceEstimate(X, vertices=vertices, tmin=tmin, tstep=tstep,
                         subject=src._subject)
    logger.info('[done]')
    return stc


@verbose
def mixed_norm(evoked, forward, noise_cov, alpha='sure', loose='auto',
               depth=0.8, maxit=3000, tol=1e-4, active_set_size=10,
               debias=True, time_pca=True, weights=None, weights_min=0.,
               solver='auto', n_mxne_iter=1, return_residual=False,
               return_as_dipoles=False, dgap_freq=10, rank=None, pick_ori=None,
               sure_alpha_grid="auto", random_state=None, verbose=None):
    """Mixed-norm estimate (MxNE) and iterative reweighted MxNE (irMxNE).

    Compute L1/L2 mixed-norm solution :footcite:`GramfortEtAl2012` or L0.5/L2
    :footcite:`StrohmeierEtAl2016` mixed-norm solution on evoked data.

    Parameters
    ----------
    evoked : instance of Evoked or list of instances of Evoked
        Evoked data to invert.
    forward : dict
        Forward operator.
    noise_cov : instance of Covariance
        Noise covariance to compute whitener.
    alpha : float | str
        Regularization parameter. If float it should be in the range [0, 100):
        0 means no regularization, 100 would give 0 active dipole.
        If ``'sure'`` (default), the SURE method from
        :footcite:`DeledalleEtAl2014` will be used.

        .. versionchanged:: 0.24
          The default was changed to ``'sure'``.
    %(loose)s
    %(depth)s
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter.
    active_set_size : int | None
        Size of active set increment. If None, no active set strategy is used.
    debias : bool
        Remove coefficient amplitude bias due to L1 penalty.
    time_pca : bool or int
        If True the rank of the concatenated epochs is reduced to
        its true dimension. If is 'int' the rank is limited to this value.
    weights : None | array | SourceEstimate
        Weight for penalty in mixed_norm. Can be None, a
        1d array with shape (n_sources,), or a SourceEstimate (e.g. obtained
        with wMNE, dSPM, or fMRI).
    weights_min : float
        Do not consider in the estimation sources for which weights
        is less than weights_min.
    solver : 'prox' | 'cd' | 'bcd' | 'auto'
        The algorithm to use for the optimization. 'prox' stands for
        proximal iterations using the FISTA algorithm, 'cd' uses
        coordinate descent, and 'bcd' applies block coordinate descent.
        'cd' is only available for fixed orientation.
    n_mxne_iter : int
        The number of MxNE iterations. If > 1, iterative reweighting
        is applied.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    return_as_dipoles : bool
        If True, the sources are returned as a list of Dipole instances.
    dgap_freq : int or np.inf
        The duality gap is evaluated every dgap_freq iterations. Ignored if
        solver is 'cd'.
    %(rank_None)s

        .. versionadded:: 0.18
    %(pick_ori)s
    sure_alpha_grid : array | str
        If ``'auto'`` (default), the SURE is evaluated along 15 uniformly
        distributed alphas between alpha_max and 0.1 * alpha_max. If array, the
        grid is directly specified. Ignored if alpha is not "sure".

        .. versionadded:: 0.24
    random_state : int | None
        The random state used in a random number generator for delta and
        epsilon used for the SURE computation. Defaults to None.

        .. versionadded:: 0.24
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate | list of SourceEstimate
        Source time courses for each evoked data passed as input.
    residual : instance of Evoked
        The residual a.k.a. data not explained by the sources.
        Only returned if return_residual is True.

    See Also
    --------
    tf_mixed_norm

    References
    ----------
    .. footbibliography::
    """
    from scipy import linalg
    _validate_type(alpha, ('numeric', str), 'alpha')
    if isinstance(alpha, str):
        _check_option('alpha', alpha, ('sure',))
    elif not 0. <= alpha < 100:
        raise ValueError('If not equal to "sure" alpha must be in [0, 100). '
                         'Got alpha = %s' % alpha)
    if n_mxne_iter < 1:
        raise ValueError('MxNE has to be computed at least 1 time. '
                         'Requires n_mxne_iter >= 1, got %d' % n_mxne_iter)
    if dgap_freq <= 0.:
        raise ValueError('dgap_freq must be a positive integer.'
                         ' Got dgap_freq = %s' % dgap_freq)
    if not(isinstance(sure_alpha_grid, (np.ndarray, list)) or
           sure_alpha_grid == "auto"):
        raise ValueError('If not equal to "auto" sure_alpha_grid must be an '
                         'array. Got %s' % type(sure_alpha_grid))
    if ((isinstance(sure_alpha_grid, str) and sure_alpha_grid != "auto")
            and (isinstance(alpha, str) and alpha != "sure")):
        raise Exception('If sure_alpha_grid is manually specified, alpha must '
                        'be "sure". Got %s' % alpha)
    pca = True
    if not isinstance(evoked, list):
        evoked = [evoked]

    _check_reference(evoked[0])

    all_ch_names = evoked[0].ch_names
    if not all(all_ch_names == evoked[i].ch_names
               for i in range(1, len(evoked))):
        raise Exception('All the datasets must have the same good channels.')

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked[0].info, noise_cov, pca, depth, loose, rank,
        weights, weights_min)
    _check_ori(pick_ori, forward)

    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = np.concatenate([e.data[sel] for e in evoked], axis=1)

    # Whiten data
    logger.info('Whitening data matrix.')
    M = np.dot(whitener, M)

    if time_pca:
        U, s, Vh = linalg.svd(M, full_matrices=False)
        if not isinstance(time_pca, bool) and isinstance(time_pca, int):
            U = U[:, :time_pca]
            s = s[:time_pca]
            Vh = Vh[:time_pca]
        M = U * s

    # Scaling to make setting of tol and alpha easy
    tol *= sum_squared(M)
    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    alpha_max = norm_l2inf(np.dot(gain.T, M), n_dip_per_pos, copy=False)
    alpha_max *= 0.01
    gain /= alpha_max
    source_weighting /= alpha_max

    # Alpha selected automatically by SURE minimization
    if alpha == "sure":
        alpha_grid = sure_alpha_grid
        if isinstance(sure_alpha_grid, str) and sure_alpha_grid == "auto":
            alpha_grid = np.geomspace(100, 10, num=15)
        X, active_set, best_alpha_ = _compute_mxne_sure(
            M, gain, alpha_grid, sigma=1, random_state=random_state,
            n_mxne_iter=n_mxne_iter, maxit=maxit, tol=tol,
            n_orient=n_dip_per_pos, active_set_size=active_set_size,
            debias=debias, solver=solver, dgap_freq=dgap_freq, verbose=verbose)
        logger.info('Selected alpha: %s' % best_alpha_)
    else:
        if n_mxne_iter == 1:
            X, active_set, E = mixed_norm_solver(
                M, gain, alpha, maxit=maxit, tol=tol,
                active_set_size=active_set_size, n_orient=n_dip_per_pos,
                debias=debias, solver=solver, dgap_freq=dgap_freq,
                verbose=verbose)
        else:
            X, active_set, E = iterative_mixed_norm_solver(
                M, gain, alpha, n_mxne_iter, maxit=maxit, tol=tol,
                n_orient=n_dip_per_pos, active_set_size=active_set_size,
                debias=debias, solver=solver, dgap_freq=dgap_freq,
                verbose=verbose)

    if time_pca:
        X = np.dot(X, Vh)
        M = np.dot(M, Vh)

    gain_active = gain[:, active_set]
    if mask is not None:
        active_set_tmp = np.zeros(len(mask), dtype=bool)
        active_set_tmp[mask] = active_set
        active_set = active_set_tmp
        del active_set_tmp

    if active_set.sum() == 0:
        warn("No active dipoles found. alpha is too big.")
        M_estimate = np.zeros_like(M)
    else:
        # Reapply weights to have correct unit
        X = _reapply_source_weighting(X, source_weighting, active_set)
        source_weighting[source_weighting == 0] = 1  # zeros
        gain_active /= source_weighting[active_set]
        del source_weighting
        M_estimate = np.dot(gain_active, X)

    outs = list()
    residual = list()
    cnt = 0
    for e in evoked:
        tmin = e.times[0]
        tstep = 1.0 / e.info['sfreq']
        Xe = X[:, cnt:(cnt + len(e.times))]
        if return_as_dipoles:
            out = _make_dipoles_sparse(
                Xe, active_set, forward, tmin, tstep,
                M[:, cnt:(cnt + len(e.times))],
                gain_active)
        else:
            out = _make_sparse_stc(
                Xe, active_set, forward, tmin, tstep, pick_ori=pick_ori)
        outs.append(out)
        cnt += len(e.times)

        if return_residual:
            residual.append(_compute_residual(forward, e, Xe, active_set,
                                              gain_info))

    _log_exp_var(M, M_estimate, prefix='')
    logger.info('[done]')

    if len(outs) == 1:
        out = outs[0]
        if return_residual:
            residual = residual[0]
    else:
        out = outs

    if return_residual:
        out = out, residual

    return out


def _window_evoked(evoked, size):
    """Window evoked (size in seconds)."""
    if isinstance(size, (float, int)):
        lsize = rsize = float(size)
    else:
        lsize, rsize = size
    evoked = evoked.copy()
    sfreq = float(evoked.info['sfreq'])
    lsize = int(lsize * sfreq)
    rsize = int(rsize * sfreq)
    lhann = np.hanning(lsize * 2)[:lsize]
    rhann = np.hanning(rsize * 2)[-rsize:]
    window = np.r_[lhann, np.ones(len(evoked.times) - lsize - rsize), rhann]
    evoked.data *= window[None, :]
    return evoked


@verbose
def tf_mixed_norm(evoked, forward, noise_cov,
                  loose='auto', depth=0.8, maxit=3000,
                  tol=1e-4, weights=None, weights_min=0., pca=True,
                  debias=True, wsize=64, tstep=4, window=0.02,
                  return_residual=False, return_as_dipoles=False, alpha=None,
                  l1_ratio=None, dgap_freq=10, rank=None, pick_ori=None,
                  n_tfmxne_iter=1, verbose=None):
    """Time-Frequency Mixed-norm estimate (TF-MxNE).

    Compute L1/L2 + L1 mixed-norm solution on time-frequency
    dictionary. Works with evoked data
    :footcite:`GramfortEtAl2013b,GramfortEtAl2011`.

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked data to invert.
    forward : dict
        Forward operator.
    noise_cov : instance of Covariance
        Noise covariance to compute whitener.
    %(loose)s
    %(depth)s
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter.
    weights : None | array | SourceEstimate
        Weight for penalty in mixed_norm. Can be None or
        1d array of length n_sources or a SourceEstimate e.g. obtained
        with wMNE or dSPM or fMRI.
    weights_min : float
        Do not consider in the estimation sources for which weights
        is less than weights_min.
    pca : bool
        If True the rank of the data is reduced to true dimension.
    debias : bool
        Remove coefficient amplitude bias due to L1 penalty.
    wsize : int or array-like
        Length of the STFT window in samples (must be a multiple of 4).
        If an array is passed, multiple TF dictionaries are used (each having
        its own wsize and tstep) and each entry of wsize must be a multiple
        of 4. See :footcite:`BekhtiEtAl2016`.
    tstep : int or array-like
        Step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) (default: wsize/2).
        If an array is passed, multiple TF dictionaries are used (each having
        its own wsize and tstep), and each entry of tstep must be a multiple
        of 2 and divide the corresponding entry of wsize. See
        :footcite:`BekhtiEtAl2016`.
    window : float or (float, float)
        Length of time window used to take care of edge artifacts in seconds.
        It can be one float or float if the values are different for left
        and right window length.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    return_as_dipoles : bool
        If True, the sources are returned as a list of Dipole instances.
    alpha : float in [0, 100) or None
        Overall regularization parameter.
        If alpha and l1_ratio are not None, alpha_space and alpha_time are
        overridden by alpha * alpha_max * (1. - l1_ratio) and alpha * alpha_max
        * l1_ratio. 0 means no regularization, 100 would give 0 active dipole.
    l1_ratio : float in [0, 1] or None
        Proportion of temporal regularization.
        If l1_ratio and alpha are not None, alpha_space and alpha_time are
        overridden by alpha * alpha_max * (1. - l1_ratio) and alpha * alpha_max
        * l1_ratio. 0 means no time regularization a.k.a. MxNE.
    dgap_freq : int or np.inf
        The duality gap is evaluated every dgap_freq iterations.
    %(rank_None)s

        .. versionadded:: 0.18
    %(pick_ori)s
    n_tfmxne_iter : int
        Number of TF-MxNE iterations. If > 1, iterative reweighting is applied.
    %(verbose)s

    Returns
    -------
    stc : instance of SourceEstimate
        Source time courses.
    residual : instance of Evoked
        The residual a.k.a. data not explained by the sources.
        Only returned if return_residual is True.

    See Also
    --------
    mixed_norm

    References
    ----------
    .. footbibliography::
    """
    _check_reference(evoked)

    all_ch_names = evoked.ch_names
    info = evoked.info

    if not (0. <= alpha < 100.):
        raise ValueError('alpha must be in [0, 100). '
                         'Got alpha = %s' % alpha)

    if not (0. <= l1_ratio <= 1.):
        raise ValueError('l1_ratio must be in range [0, 1].'
                         ' Got l1_ratio = %s' % l1_ratio)
    alpha_space = alpha * (1. - l1_ratio)
    alpha_time = alpha * l1_ratio

    if n_tfmxne_iter < 1:
        raise ValueError('TF-MxNE has to be computed at least 1 time. '
                         'Requires n_tfmxne_iter >= 1, got %s' % n_tfmxne_iter)

    if dgap_freq <= 0.:
        raise ValueError('dgap_freq must be a positive integer.'
                         ' Got dgap_freq = %s' % dgap_freq)

    tstep = np.atleast_1d(tstep)
    wsize = np.atleast_1d(wsize)
    if len(tstep) != len(wsize):
        raise ValueError('The same number of window sizes and steps must be '
                         'passed. Got tstep = %s and wsize = %s' %
                         (tstep, wsize))

    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca, depth, loose, rank,
        weights, weights_min)
    _check_ori(pick_ori, forward)

    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3

    if window is not None:
        evoked = _window_evoked(evoked, window)

    sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
    M = evoked.data[sel]

    # Whiten data
    logger.info('Whitening data matrix.')
    M = np.dot(whitener, M)

    n_steps = np.ceil(M.shape[1] / tstep.astype(float)).astype(int)
    n_freqs = wsize // 2 + 1
    n_coefs = n_steps * n_freqs
    phi = _Phi(wsize, tstep, n_coefs, evoked.data.shape[1])

    # Scaling to make setting of tol and alpha easy
    tol *= sum_squared(M)
    alpha_max = norm_epsilon_inf(gain, M, phi, l1_ratio, n_dip_per_pos)
    alpha_max *= 0.01
    gain /= alpha_max
    source_weighting /= alpha_max

    if n_tfmxne_iter == 1:
        X, active_set, E = tf_mixed_norm_solver(
            M, gain, alpha_space, alpha_time, wsize=wsize, tstep=tstep,
            maxit=maxit, tol=tol, verbose=verbose, n_orient=n_dip_per_pos,
            dgap_freq=dgap_freq, debias=debias)
    else:
        X, active_set, E = iterative_tf_mixed_norm_solver(
            M, gain, alpha_space, alpha_time, wsize=wsize, tstep=tstep,
            n_tfmxne_iter=n_tfmxne_iter, maxit=maxit, tol=tol, verbose=verbose,
            n_orient=n_dip_per_pos, dgap_freq=dgap_freq, debias=debias)

    if active_set.sum() == 0:
        raise Exception("No active dipoles found. "
                        "alpha_space/alpha_time are too big.")

    # Compute estimated whitened sensor data for each dipole (dip, ch, time)
    gain_active = gain[:, active_set]

    if mask is not None:
        active_set_tmp = np.zeros(len(mask), dtype=bool)
        active_set_tmp[mask] = active_set
        active_set = active_set_tmp
        del active_set_tmp

    X = _reapply_source_weighting(X, source_weighting, active_set)
    gain_active /= source_weighting[active_set]

    if return_residual:
        residual = _compute_residual(
            forward, evoked, X, active_set, gain_info)

    if return_as_dipoles:
        out = _make_dipoles_sparse(
            X, active_set, forward, evoked.times[0], 1.0 / info['sfreq'],
            M, gain_active)
    else:
        out = _make_sparse_stc(
            X, active_set, forward, evoked.times[0], 1.0 / info['sfreq'],
            pick_ori=pick_ori)

    logger.info('[done]')

    if return_residual:
        out = out, residual

    return out


@verbose
def _compute_mxne_sure(M, gain, alpha_grid, sigma, n_mxne_iter, maxit, tol,
                       n_orient, active_set_size, debias, solver, dgap_freq,
                       random_state, verbose):
    """Stein Unbiased Risk Estimator (SURE).

    Implements the finite-difference Monte-Carlo approximation
    of the SURE for Multi-Task LASSO.

    See reference :footcite:`DeledalleEtAl2014`.

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    gain : array, shape (n_sensors, n_dipoles)
        The gain matrix a.k.a. lead field.
    alpha_grid : array, shape (n_alphas,)
        The grid of alphas used to evaluate the SURE.
    sigma : float
        The true or estimated noise level in the data. Usually 1 if the data
        has been previously whitened using MNE whitener.
    n_mxne_iter : int
        The number of MxNE iterations. If > 1, iterative reweighting is
        applied.
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    active_set_size : int
        Size of active set increase at each iteration.
    debias : bool
        Debias source estimates.
    solver : 'prox' | 'cd' | 'bcd' | 'auto'
        The algorithm to use for the optimization.
    dgap_freq : int or np.inf
        The duality gap is evaluated every dgap_freq iterations.
    random_state : int | None
        The random state used in a random number generator for delta and
        epsilon used for the SURE computation.

    Returns
    -------
    X : array, shape (n_active, n_times)
        Coefficient matrix.
    active_set : array, shape (n_dipoles,)
        Array of indices of non-zero coefficients.
    best_alpha_ : float
        Alpha that minimizes the SURE.

    References
    ----------
    .. footbibliography::
    """
    def g(w):
        return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    def gprime(w):
        return 2. * np.repeat(g(w), n_orient).ravel()

    def _run_solver(alpha, M, n_mxne_iter, as_init=None, X_init=None,
                    w_init=None):
        if n_mxne_iter == 1:
            X, active_set, _ = mixed_norm_solver(
                M, gain, alpha, maxit=maxit, tol=tol,
                active_set_size=active_set_size, n_orient=n_orient,
                debias=debias, solver=solver, dgap_freq=dgap_freq,
                active_set_init=as_init, X_init=X_init, verbose=False)
        else:
            X, active_set, _ = iterative_mixed_norm_solver(
                M, gain, alpha, n_mxne_iter, maxit=maxit, tol=tol,
                n_orient=n_orient, active_set_size=active_set_size,
                debias=debias, solver=solver, dgap_freq=dgap_freq,
                weight_init=w_init, verbose=False)
        return X, active_set

    def _fit_on_grid(gain, M, eps, delta):
        coefs_grid_1_0 = np.zeros((len(alpha_grid), gain.shape[1], M.shape[1]))
        coefs_grid_2_0 = np.zeros((len(alpha_grid), gain.shape[1], M.shape[1]))
        active_sets, active_sets_eps = [], []
        M_eps = M + eps * delta
        # warm start - first iteration (leverages convexity)
        logger.info('Warm starting...')
        for j, alpha in enumerate(alpha_grid):
            logger.info('alpha: %s' % alpha)
            X, a_set = _run_solver(alpha, M, 1)
            X_eps, a_set_eps = _run_solver(alpha, M_eps, 1)
            coefs_grid_1_0[j][a_set, :] = X
            coefs_grid_2_0[j][a_set_eps, :] = X_eps
            active_sets.append(a_set)
            active_sets_eps.append(a_set_eps)
        # next iterations
        if n_mxne_iter == 1:
            return coefs_grid_1_0, coefs_grid_2_0, active_sets
        else:
            coefs_grid_1 = coefs_grid_1_0.copy()
            coefs_grid_2 = coefs_grid_2_0.copy()
            logger.info('Fitting SURE on grid.')
            for j, alpha in enumerate(alpha_grid):
                logger.info('alpha: %s' % alpha)
                if active_sets[j].sum() > 0:
                    w = gprime(coefs_grid_1[j])
                    X, a_set = _run_solver(alpha, M, n_mxne_iter - 1,
                                           w_init=w)
                    coefs_grid_1[j][a_set, :] = X
                    active_sets[j] = a_set
                if active_sets_eps[j].sum() > 0:
                    w_eps = gprime(coefs_grid_2[j])
                    X_eps, a_set_eps = _run_solver(alpha, M_eps,
                                                   n_mxne_iter - 1,
                                                   w_init=w_eps)
                    coefs_grid_2[j][a_set_eps, :] = X_eps
                    active_sets_eps[j] = a_set_eps

            return coefs_grid_1, coefs_grid_2, active_sets

    def _compute_sure_val(coef1, coef2, gain, M, sigma, delta, eps):
        n_sensors, n_times = gain.shape[0], M.shape[1]
        dof = (gain @ (coef2 - coef1) * delta).sum() / eps
        df_term = np.linalg.norm(M - gain @ coef1) ** 2
        sure = df_term - n_sensors * n_times * sigma ** 2
        sure += 2 * dof * sigma ** 2
        return sure

    sure_path = np.empty(len(alpha_grid))

    rng = check_random_state(random_state)
    # See Deledalle et al. 20214 Sec. 5.1
    eps = 2 * sigma / (M.shape[0] ** 0.3)
    delta = rng.randn(*M.shape)

    coefs_grid_1, coefs_grid_2, active_sets = _fit_on_grid(gain, M, eps, delta)

    logger.info("Computing SURE values on grid.")
    for i, (coef1, coef2) in enumerate(zip(coefs_grid_1, coefs_grid_2)):
        sure_path[i] = _compute_sure_val(
            coef1, coef2, gain, M, sigma, delta, eps)
        if verbose:
            logger.info("alpha %s :: sure %s" % (alpha_grid[i], sure_path[i]))
    best_alpha_ = alpha_grid[np.argmin(sure_path)]

    X = coefs_grid_1[np.argmin(sure_path)]
    active_set = active_sets[np.argmin(sure_path)]

    X = X[active_set, :]

    return X, active_set, best_alpha_
