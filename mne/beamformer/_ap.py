"""compute an Alternating Projection (AP)."""

# Authors: Yuval Realpe <yuval.realpe@gmail.com>
#
# License: BSD-3-Clause

from copy import copy

import numpy as np
from numpy.linalg import pinv, multi_dot, lstsq

from ..utils import _check_info_inv, verbose, fill_doc, logger
from ._compute_beamformer import _prepare_beamformer_input
from ..io.pick import pick_channels_forward, pick_info
from ..forward.forward import convert_forward_solution, is_fixed_orient
from ..inverse_sparse.mxne_inverse import _make_dipoles_sparse
from ..minimum_norm.inverse import _log_exp_var


def _produce_data_cov(data_arr, n_sources):
    """Calculate data Covariance."""
    data_tr = data_arr.transpose()
    data_norm = np.matmul(data_arr, data_tr)
    data_cov = (
        data_norm + data_norm.trace() * np.eye(data_arr.shape[0]) * 0.001
    )  # Array Covariance Matrix
    logger.info(" alternating projection ; n_sources = {}:".format(n_sources))

    return data_cov


def _active_subspace(Mat):
    """Calculate Multi_dot with pseudo-inverse."""
    return multi_dot([Mat, pinv(Mat.T @ Mat), Mat.T])


def _argmax_of_remaining_dipoles(gain, perpend_spc, data_cov):
    """Find best fitting dipole index over remaining sub-space."""
    numerator = np.diagonal(
        multi_dot([gain.T, perpend_spc, data_cov, perpend_spc, gain])
    )
    denominator = np.diagonal(multi_dot([gain.T, perpend_spc, gain]))
    return np.argmax(numerator / denominator)


def _fixed_phase1a(data_cov, gain):
    """Calculate phase 1a of fixed oriented AP.

    Initialization: search the 1st source location over the entire
    dipoles topographies space.

    Parameters
    ----------
    data_cov : array
        Data Covariance.
    gain : array, shape (nchannels, n_dipoles)
        Gain matrix.

    Returns
    -------
    s_ap : list of int
        List of dipole indices.

    """
    s_ap = []
    s1_idx = np.argmax(
        np.diagonal(multi_dot([gain.T, data_cov, gain]))
        / np.diagonal(np.dot(gain.T, gain))
    )
    s_ap.append(s1_idx)
    return s_ap


def _fixed_phase1b(gain, s_ap, data_cov, n_sources):
    """Calculate phase 1b of fixed oriented AP.

    Adding one source at a time.

    Parameters
    ----------
    gain : array, shape (nchannels, n_dipoles)
        Gain matrix.
    s_ap : list of int
        List of dipole indices.
    data_cov : array
        Data Covariance.
    n_sources : int
        The number of dipoles to estimate.

    Returns
    -------
    s_ap : list of int
        List of dipole indices.

    """
    for _ in range(1, n_sources):
        sub_g = gain[:, s_ap]
        act_spc = _active_subspace(sub_g)
        perpend_spc = np.eye(act_spc.shape[0]) - act_spc
        s2_idx = _argmax_of_remaining_dipoles(gain, perpend_spc, data_cov)
        s_ap.append(s2_idx)
    logger.info("current s_ap = {}".format(s_ap))
    return s_ap


def _fixed_phase2(n_sources, max_iter, s_ap_2, gain, data_cov):
    """Calculate phase 2 of fixed oriented AP.

    Altering the projection of current estimated dipoles.

    Parameters
    ----------
    n_sources : int
        The number of dipoles to estimate.
    max_iter : int
        Maximal iteration number of AP.
    s_ap_2 : list of int
        List of dipole indices.
    gain : array, shape (nchannels, n_dipoles)
        Gain matrix.
    data_cov : array
        Data Covariance.

    Returns
    -------
    s_ap_2 : list of int
        List of dipole indices.

    """
    for itr in range(max_iter):
        logger.info("iteration No. {}".format(itr + 1))
        s_ap_2_prev = copy(s_ap_2)
        for src in range(n_sources):
            # AP localization of src-th source
            s_ap_temp = copy(s_ap_2)
            s_ap_temp.pop(src)
            sub_g = gain[:, s_ap_temp]
            act_spc = _active_subspace(sub_g)
            perpend_spc = np.eye(act_spc.shape[0]) - act_spc
            s2_idx = _argmax_of_remaining_dipoles(gain, perpend_spc, data_cov)
            s_ap_2[src] = s2_idx
        logger.info("current s_ap_2 = {}".format(s_ap_2))
        if (itr > 0) & (s_ap_2_prev == s_ap_2):
            # No improvement vs. previous iteration
            logger.info("Done (optimally)")
            break
        if itr == max_iter:
            logger.info("Done (max iteration)")
    return s_ap_2


def _calculate_fixed_alternating_projections(data_arr, gain, n_sources, max_iter):
    """Calculate fixed-orientation alternating projection.

    Parameters
    ----------
    data_arr : array, shape (nchannels, times)
        Filtered evoked data array.
    gain : array, shape (nchannels, n_dipoles)
        Gain matrix.
    n_sources : int
        The number of dipoles to estimate.
    max_iter : int
        Maximal iteration number of AP.

    Returns
    -------
    s_ap_2 : list of int
        List of dipole indices.

    """
    s_ap = []
    # n_dipoles = gain.shape[1]
    # n_dipoles is un-used: Number of dipoles throughout the model.
    logger.info("calculating fixed-orientation alternating projection")
    data_cov = _produce_data_cov(data_arr, n_sources)

    # ######################################
    # 1st Phase
    # (a) Initialization: search the 1st source location over the entire
    # dipoles topographies space
    # ######################################

    logger.info(" 1st phase : ")
    s_ap = _fixed_phase1a(data_cov, gain)

    # ######################################
    # (b) Now, add one source at a time
    # ######################################

    s_ap = _fixed_phase1b(gain, s_ap, data_cov, n_sources)

    # #####################################
    # 2nd phase
    # #####################################

    logger.info(" 2nd phase : ")
    s_ap_2 = copy(s_ap)
    s_ap_2 = _fixed_phase2(n_sources, max_iter, s_ap_2, gain, data_cov)

    return s_ap_2


def _solve_active_gain_eig(ind, gain, data_cov, eig, perpend_spc):
    """Eigen values and vector of the projection."""
    gain_idx = slice(ind * 3, ind * 3 + 3)
    l_p = gain[:, gain_idx]
    eig_a = multi_dot([l_p.T, perpend_spc, data_cov, perpend_spc, l_p])
    eig_b = multi_dot([l_p.T, perpend_spc, perpend_spc, l_p])
    eig_b += 1e-3 * eig_b.trace() * np.eye(3)
    eig_val, eig_vec = eig(eig_a, eig_b)

    return eig_val, eig_vec, l_p


def _free_phase1a(n_sources, n_dipoles, gain, data_cov):
    """Calculate phase 1a of free oriented AP.

    Initialization: search the 1st source location over
    the entire dipoles topographies space.

    Parameters
    ----------
    n_sources : int
        The number of dipoles to estimate.
    n_dipoles : int
        Number of dipoles throughout the model.
    gain : array, shape (nchannels, 3 * n_dipoles)
        Gain matrix
    data_cov : array
        Data Covariance.

    Returns
    -------
    s_ap : list of int
        List of dipole indices.
    oris : array, shape (n_sources, 3)
        Orientations array of estimated sources (sorted by s_ap).
    sub_g_proj : array
        Sub space projected by estimated dipoles.

    """
    from scipy import linalg

    s_ap = []
    oris = np.empty((n_sources, 3))
    ap_val1 = np.zeros(n_dipoles)
    perpend_spc = np.eye(gain.shape[0])
    for dip in range(n_dipoles):
        sol_tuple = _solve_active_gain_eig(dip, gain, data_cov, linalg.eig, perpend_spc)
        ap_val1[dip] = np.max([x.real for x in sol_tuple[0]])

    # obtain the 1st source location
    s1_idx = np.argmax(ap_val1)
    s_ap.append(s1_idx)

    # obtain the 1st source orientation
    sol_tuple = _solve_active_gain_eig(s1_idx, gain, data_cov, linalg.eig, perpend_spc)

    oris[0] = sol_tuple[1][:, [np.argmax([x.real for x in sol_tuple[0]])]][:, 0]
    sub_g_proj = sol_tuple[2] @ oris[0][:, np.newaxis]

    return s_ap, oris, sub_g_proj


def _free_phase1b(n_sources, n_dipoles, gain, data_cov, ap_temp_tuple, force_no_rep):
    """Calculate phase 1b of free oriented AP.

    Adding one source at a time.

    Parameters
    ----------
    n_sources : int
        The number of dipoles to estimate.
    n_dipoles : int
        Number of dipoles throughout the model.
    gain : array, shape (nchannels, 3 * n_dipoles)
        Gain matrix.
    data_cov : array
        Data Covariance.
    ap_temp_tuple : tuple
        See: _free_phase1a Returns.
    force_no_rep : bool
        Forces no repetition of estinated dipoles.

    Returns
    -------
    s_ap : list of int
        List of dipole indices.
    oris : array, shape (n_sources, 3)
        Orientations array of estimated sources (sorted by s_ap).
    sub_g_proj : array
        Sub space projected by estimated dipoles.

    """
    # ap_temp_tuple = (s_ap, oris, sub_g_proj)
    from scipy import linalg

    s_ap, oris, sub_g_proj = copy(ap_temp_tuple)
    for src in range(1, n_sources):
        ap_val2 = np.zeros(n_dipoles)
        act_spc = _active_subspace(sub_g_proj)
        perpend_spc = np.eye(act_spc.shape[0]) - act_spc
        for dip in range(n_dipoles):
            if force_no_rep and (dip in s_ap):
                continue
            sol_tuple = _solve_active_gain_eig(
                dip, gain, data_cov, linalg.eig, perpend_spc
            )
            ap_val2[dip] = np.max([x.real for x in sol_tuple[0]])

        s2_idx = np.argmax(ap_val2)
        s_ap.append(s2_idx)
        sol_tuple = _solve_active_gain_eig(
            s2_idx, gain, data_cov, linalg.eig, perpend_spc
        )

        oris[src] = sol_tuple[1][:, [np.argmax([x.real for x in sol_tuple[0]])]][:, 0]
        sub_g_proj = np.concatenate(
            [sub_g_proj, sol_tuple[2] @ oris[src][:, np.newaxis]], axis=1
        )

    return s_ap, oris, sub_g_proj


def _free_phase2(
    ap_temp_tuple, n_sources, n_dipoles, max_iter, data_cov, gain, force_no_rep
):
    """Calculate phase 2 of free oriented AP.

    altering the projection of current estimated dipoles

    Parameters
    ----------
    ap_temp_tuple : tuple
        See: _free_phase1b.
    n_sources : int
        The number of dipoles to estimate.
    n_dipoles : int
        Number of dipoles throughout the model.
    max_iter : int
        Maximal iteration number of AP.
    data_cov : array
        Data Covariance.
    gain : array, shape (nchannels, 3 * n_dipoles)
        Gain matrix.
    force_no_rep : bool
        Forces no repetition of estinated dipoles.

    Returns
    -------
    s_ap_2 : list of int
        List of dipole indices.
    oris : array, shape (n_sources, 3)
        Orientations array of estimated sources (sorted by s_ap_2).
    sub_g_proj : array
        Sub space projected by estimated dipoles.

    """
    # ap_temp_tuple = (s_ap, oris, sub_g_proj)
    from scipy import linalg

    s_ap_2, oris, sub_g_proj = copy(ap_temp_tuple)
    logger.info(" 2nd phase : ")
    for itr in range(max_iter):
        logger.info("iteration No. {}".format(itr + 1))
        s_ap_2_prev = copy(s_ap_2)
        for src in range(n_sources):
            # AP localization of src-th source
            ap_val2 = np.zeros(n_dipoles)
            a_tmp = copy(ap_temp_tuple[2])
            a_tmp = np.delete(a_tmp, src, 1)
            act_spc = _active_subspace(a_tmp)
            perpend_spc = np.eye(act_spc.shape[0]) - act_spc
            for dip in range(n_dipoles):
                if force_no_rep and (dip in np.delete(s_ap_2, src, 0)):
                    continue
                sol_tuple = _solve_active_gain_eig(
                    dip, gain, data_cov, linalg.eig, perpend_spc
                )
                ap_val2[dip] = np.max([x.real for x in sol_tuple[0]])

            sq_idx = np.argmax(ap_val2)
            s_ap_2[src] = sq_idx
            sol_tuple = _solve_active_gain_eig(
                sq_idx, gain, data_cov, linalg.eig, perpend_spc
            )

            oris[src] = sol_tuple[1][:, [np.argmax([x.real for x in sol_tuple[0]])]][
                :, 0
            ]
            sub_g_proj[:, src] = sol_tuple[2] @ oris[src]

        logger.info("current s_ap_2 = {}".format(s_ap_2))
        if (itr > 0) & (s_ap_2_prev == s_ap_2):
            # No improvement vs. previous iteration
            logger.info("Done (optimally)")
            break
        if itr == max_iter:
            logger.info("Done (max iteration)")

    return s_ap_2, oris, sub_g_proj


def _calculate_free_alternating_projections(
    data_arr, gain, n_sources, max_iter, force_no_rep
):
    """Calculate free-orientation alternating projection.

    Parameters
    ----------
    data_arr : array, shape (nchannels, times)
        Filtered evoked data array.
    gain : array, shape (nchannels, 3 * n_dipoles)
        Gain array.
    n_sources : int
        The number of dipoles to estimate.
    max_iter : int
        Maximal iteration number of AP.

    Returns
    -------
    ap_temp_tuple : tuple
        See: _free_phase2.

    """
    logger.info("calculating free-orientation alternating projection")
    n_dipoles = int(gain.shape[1] / 3)

    data_cov = _produce_data_cov(data_arr, n_sources)
    # ######################################
    # 1st Phase
    # (a) Initialization: search the 1st source location over the entire
    # dipoles topographies space
    # ######################################

    logger.info(" 1st phase : ")
    ap_temp_tuple = _free_phase1a(n_sources, n_dipoles, gain, data_cov)
    # ap_temp_tuple = (s_ap, oris, sub_g_proj)

    # ######################################
    # (b) Now, add one source at a time
    # ######################################

    ap_temp_tuple = _free_phase1b(
        n_sources, n_dipoles, gain, data_cov, ap_temp_tuple, force_no_rep
    )
    # ap_temp_tuple = (s_ap, oris, sub_g_proj)
    logger.info("current s_ap = {}".format(ap_temp_tuple[0]))

    # #####################################
    # 2nd phase
    # #####################################

    ap_temp_tuple = _free_phase2(
        ap_temp_tuple, n_sources, n_dipoles, max_iter, data_cov, gain, force_no_rep
    )

    return ap_temp_tuple


def _free_ori_ap(wh_data, gain, n_sources, forward, max_iter, force_no_rep):
    """Branch of calculations dedicated to freely oriented dipoles."""
    sol_tuple = _calculate_free_alternating_projections(
        wh_data, gain, n_sources, max_iter, force_no_rep
    )
    # sol_tuple = active_idx, active_orientations, active_idx_gain

    sol = lstsq(sol_tuple[2], wh_data, rcond=None)[0]

    gain_fwd = forward["sol"]["data"].copy()
    gain_fwd.shape = (gain_fwd.shape[0], -1, 3)
    gain_active = gain_fwd[:, sol_tuple[0]]
    gain_dip = (sol_tuple[1] * gain_active).sum(-1)
    idx = np.array(sol_tuple[0])
    active_set = np.array([[3 * idx, 3 * idx + 1, 3 * idx + 2]]).T.ravel()

    return (
        active_set,
        sol_tuple[1],
        forward["source_rr"][sol_tuple[0]],
        gain_active,
        gain_dip,
        sol,
        sol_tuple[0],
    )


def _fixed_ori_ap(wh_data, gain, n_sources, forward, max_iter):
    """Branch of calculations dedicated to fixed oriented dipoles."""
    idx = _calculate_fixed_alternating_projections(
        wh_data, gain, n_sources=n_sources, max_iter=max_iter
    )

    sub_g = gain[:, idx]
    sol = lstsq(sub_g, wh_data, rcond=None)[0]

    gain_fwd = forward["sol"]["data"].copy()
    gain_fwd.shape = (gain_fwd.shape[0], -1, 1)
    gain_active = gain_fwd[:, idx]
    gain_dip = gain_active[:, :, 0]

    return (
        idx,
        forward["source_nn"][idx],
        forward["source_rr"][idx],
        gain_active,
        gain_dip,
        sol,
    )


@fill_doc
def _apply_ap(
    data, info, times, forward, noise_cov, n_sources, picks, max_iter, force_no_rep
):
    """AP for evoked data.

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        Evoked data.
    %(info_not_none)s
    times : array
        Time sampling values.
    forward : instance of Forward
        Forward operator.
    noise_cov : instance of Covariance
        The noise Covariance.
    n_sources : int
        The number of dipoles to estimate.
    picks : List of int
        Channel indiecs for filtering.
    max_iter : int
        Maximal iteration number of AP.
    force_no_rep : bool
        Forces no repetition of estimated dipoles.

    Returns
    -------
    dipoles : list of instances of Dipole
        The dipole fits.
    explained_data : array
        Data explained by the dipoles using a least square fitting with the
        selected active dipoles and their estimated orientation.
    var_exp : float
        Percentile of data variation explained (see: _log_exp_var).
    dip_ind : List of int
        List of indices of dipole source estimated.
    oris : array, shape (n_sources, 3)
        Orientations array of estimated sources (sorted by dip_ind).
    poss : array, shape (n_sources, 3)
        Coordinates array of estimated sources (sorted by dip_ind).

    """
    info = pick_info(info, picks)
    del picks

    if forward["surf_ori"] and not is_fixed_orient(forward):
        forward = convert_forward_solution(forward, surf_ori=False)
    is_free_ori, info, _, _, gain, whitener, _, _ = _prepare_beamformer_input(
        info, forward, noise_cov=noise_cov, rank=None
    )
    forward = pick_channels_forward(forward, info["ch_names"], ordered=True)
    del info

    # whiten the data (leadfield already whitened)
    wh_data = np.dot(whitener, data)
    del data

    if is_free_ori:
        idx, oris, poss, gain_active, gain_dip, sol, dip_ind = _free_ori_ap(
            wh_data,
            gain,
            n_sources,
            forward,
            max_iter=max_iter,
            force_no_rep=force_no_rep,
        )
        X = sol[:, np.newaxis] * oris[:, :, np.newaxis]
        X.shape = (-1, len(times))
    else:
        idx, oris, poss, gain_active, gain_dip, sol = _fixed_ori_ap(
            wh_data, gain, n_sources, forward, max_iter=max_iter
        )
        X = sol
        dip_ind = idx

    gain_active = whitener @ gain_active.reshape(gain.shape[0], -1)
    explained_data = gain_dip @ sol
    m_estimate = whitener @ explained_data
    var_exp = _log_exp_var(wh_data, m_estimate)
    tstep = np.median(np.diff(times)) if len(times) > 1 else 1.0
    dipoles = _make_dipoles_sparse(
        X, idx, forward, times[0], tstep, wh_data, gain_active, active_is_idx=True
    )
    for dipole, ori in zip(dipoles, oris):
        signs = np.sign((dipole.ori * ori).sum(-1, keepdims=True))
        dipole.ori *= signs
        dipole.amplitude *= signs[:, 0]
    logger.info("[done]")

    return dipoles, explained_data, var_exp, dip_ind, oris, poss


def _make_explained_evoked(evoked, picks, explained_data_mat, residual=False):
    """Create a new Evoked object containing explained/residual data."""
    n_evoked = evoked.copy()
    n_evoked = n_evoked.pick(picks)
    n_evoked.data = (
        n_evoked.data - explained_data_mat if residual else explained_data_mat
    )
    active_projs = [proj for proj in n_evoked.info["projs"] if proj["active"]]
    for proj in active_projs:
        proj["active"] = False
    n_evoked.add_proj(active_projs, remove_existing=True)
    n_evoked.apply_proj()
    return n_evoked


@verbose
def alternating_projections(
    evoked,
    forward,
    n_sources,
    noise_cov=None,
    max_iter=6,
    return_residual=True,
    return_active_info=False,
    verbose=None,
    force_no_rep=False,
):
    """Alternating Projections sources localization method.

    Compute Alternating Projections (AP) on evoked data.

    The implementation follows :footcite:t:`AdlerEtAl2019,AdlerEtAl2022`.

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked object containing data to be localized.
    forward : instance of Forward
        Forward operator.
    n_sources : int
        The number of dipoles to estimate.
    noise_cov : instance of Covariance, optional
        The noise covariance matrix, used for whitening the evoked signal.
        The default is None. If None is given, the noise covariance matrix
        is created ad-hoc (See: mne.cov.make_ad_hoc_cov).
    max_iter : int, optional
        Maximal iteration over estimated dipoles. The default is 6.
    return_residual : bool, optional
        If True, appends residual, explained_data and var_exp to output.
        The default is True.
    return_active_info : bool, optional
        If True, appends estimated source's information
        (indices,coordinates,orientation). The default is False.
    %(verbose)s
    force_no_rep : bool, optional
            Forces no repetition of estinated dipoles.

    Returns
    -------
    dipoles : list of instance of Dipole
        The dipole fits.
    residual : instance of Evoked
        Data not explained by the estimated dipoles.
    explained_data : instance of Evoked
        Data explained by the estimated dipoles.
    var_exp : float
        Percentile of data variation explained
        (see: mne.minimum_norm.inverse._log_exp_var).
    idx : list of int
        List of indices of dipole source estimated.
    poss : array, shape (n_sources, 3)
        Coordinates array of estimated sources (sorted by idx).
    oris : array, shape (n_sources, 3)
        Orientations array of estimated sources (sorted by idx).

    References
    ----------
    .. footbibliography::

    .. versionadded:: 1.5
    """
    info = evoked.info
    data = evoked.data
    times = evoked.times

    picks = _check_info_inv(info, forward, data_cov=None, noise_cov=noise_cov)

    data = data[picks]

    dipoles, explained_data_mat, var_exp, idx, oris, poss = _apply_ap(
        data,
        info,
        times,
        forward,
        noise_cov,
        n_sources,
        picks,
        max_iter=max_iter,
        force_no_rep=force_no_rep,
    )

    output = [dipoles]
    if return_residual:
        # treating residual
        residual = _make_explained_evoked(
            evoked, picks, explained_data_mat, residual=True
        )

        # treating explained data
        explained_data = _make_explained_evoked(evoked, picks, explained_data_mat)

        for item in [residual, explained_data, var_exp]:
            output.append(item)

    if return_active_info:
        for item in [idx, poss, oris]:
            output.append(item)
    elif not return_residual:
        output = output[0]

    return output
