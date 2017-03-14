"""Compute a Recursively Applied and Projected MUltiple Signal Classification (RAP-MUSIC)."""  # noqa

# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..io.pick import pick_channels_evoked
from ..cov import compute_whitener
from ..utils import logger, verbose
from ..dipole import Dipole
from ._lcmv import _prepare_beamformer_input, _setup_picks


def _apply_rap_music(data, info, times, forward, noise_cov, n_dipoles=2,
                     picks=None, return_explained_data=False):
    """RAP-MUSIC for evoked data.

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        Evoked data.
    info : dict
        Measurement info.
    times : array
        Times.
    forward : instance of Forward
        Forward operator.
    noise_cov : instance of Covariance
        The noise covariance.
    n_dipoles : int
        The number of dipoles to estimate. The default value is 2.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    return_explained_data : bool
        If True, the explained data is returned as an array.

    Returns
    -------
    dipoles : list of instances of Dipole
        The dipole fits.
    explained_data : array | None
        Data explained by the dipoles using a least square fitting with the
        selected active dipoles and their estimated orientation.
        Computed only if return_explained_data is True.
    """
    is_free_ori, ch_names, proj, vertno, G = _prepare_beamformer_input(
        info, forward, label=None, picks=picks, pick_ori=None)

    gain = G.copy()

    # Handle whitening + data covariance
    whitener, _ = compute_whitener(noise_cov, info, picks)
    if info['projs']:
        whitener = np.dot(whitener, proj)

    # whiten the leadfield and the data
    G = np.dot(whitener, G)
    data = np.dot(whitener, data)

    eig_values, eig_vectors = linalg.eigh(np.dot(data, data.T))
    phi_sig = eig_vectors[:, -n_dipoles:]

    n_orient = 3 if is_free_ori else 1
    n_channels = G.shape[0]
    A = np.empty((n_channels, n_dipoles))
    gain_dip = np.empty((n_channels, n_dipoles))
    oris = np.empty((n_dipoles, 3))
    poss = np.empty((n_dipoles, 3))

    G_proj = G.copy()
    phi_sig_proj = phi_sig.copy()

    for k in range(n_dipoles):
        subcorr_max = -1.
        for i_source in range(G.shape[1] // n_orient):
            idx_k = slice(n_orient * i_source, n_orient * (i_source + 1))
            Gk = G_proj[:, idx_k]
            if n_orient == 3:
                Gk = np.dot(Gk, forward['source_nn'][idx_k])

            subcorr, ori = _compute_subcorr(Gk, phi_sig_proj)
            if subcorr > subcorr_max:
                subcorr_max = subcorr
                source_idx = i_source
                source_ori = ori
                if n_orient == 3 and source_ori[-1] < 0:
                    # make sure ori is relative to surface ori
                    source_ori *= -1  # XXX

                source_pos = forward['source_rr'][i_source]
                if n_orient == 1:
                    source_ori = forward['source_nn'][i_source]

        idx_k = slice(n_orient * source_idx, n_orient * (source_idx + 1))
        Ak = G[:, idx_k]
        if n_orient == 3:
            Ak = np.dot(Ak, np.dot(forward['source_nn'][idx_k], source_ori))

        A[:, k] = Ak.ravel()

        if return_explained_data:
            gain_k = gain[:, idx_k]
            if n_orient == 3:
                gain_k = np.dot(gain_k,
                                np.dot(forward['source_nn'][idx_k],
                                       source_ori))
            gain_dip[:, k] = gain_k.ravel()

        oris[k] = source_ori
        poss[k] = source_pos

        logger.info("source %s found: p = %s" % (k + 1, source_idx))
        if n_orient == 3:
            logger.info("ori = %s %s %s" % tuple(oris[k]))

        projection = _compute_proj(A[:, :k + 1])
        G_proj = np.dot(projection, G)
        phi_sig_proj = np.dot(projection, phi_sig)

    sol = linalg.lstsq(A, data)[0]

    gof, explained_data = [], None
    if return_explained_data:
        explained_data = np.dot(gain_dip, sol)
        gof = (linalg.norm(np.dot(whitener, explained_data)) /
               linalg.norm(data))

    return _make_dipoles(times, poss,
                         oris, sol, gof), explained_data


def _make_dipoles(times, poss, oris, sol, gof):
    """Instantiate a list of Dipoles.

    Parameters
    ----------
    times : array, shape (n_times,)
        The time instants.
    poss : array, shape (n_dipoles, 3)
        The dipoles' positions.
    oris : array, shape (n_dipoles, 3)
        The dipoles' orientations.
    sol : array, shape (n_times,)
        The dipoles' amplitudes over time.
    gof : array, shape (n_times,)
        The goodness of fit of the dipoles.
        Shared between all dipoles.

    Returns
    -------
    dipoles : list
        The list of Dipole instances.
    """
    amplitude = sol * 1e9
    oris = np.array(oris)

    dipoles = []
    for i_dip in range(poss.shape[0]):
        i_pos = poss[i_dip][np.newaxis, :].repeat(len(times), axis=0)
        i_ori = oris[i_dip][np.newaxis, :].repeat(len(times), axis=0)
        dipoles.append(Dipole(times, i_pos, amplitude[i_dip],
                              i_ori, gof))

    return dipoles


def _compute_subcorr(G, phi_sig):
    """Compute the subspace correlation."""
    Ug, Sg, Vg = linalg.svd(G, full_matrices=False)
    # Now we look at the actual rank of the forward fields
    # in G and handle the fact that it might be rank defficient
    # eg. when using MEG and a sphere model for which the
    # radial component will be truly 0.
    rank = np.sum(Sg > (Sg[0] * 1e-12))
    if rank == 0:
        return 0, np.zeros(len(G))
    rank = max(rank, 2)  # rank cannot be 1
    Ug, Sg, Vg = Ug[:, :rank], Sg[:rank], Vg[:rank]
    tmp = np.dot(Ug.T.conjugate(), phi_sig)
    Uc, Sc, _ = linalg.svd(tmp, full_matrices=False)
    X = np.dot(Vg.T / Sg[None, :], Uc[:, 0])  # subcorr
    return Sc[0], X / linalg.norm(X)


def _compute_proj(A):
    """Compute the orthogonal projection operation for a manifold vector A."""
    U, _, _ = linalg.svd(A, full_matrices=False)
    return np.identity(A.shape[0]) - np.dot(U, U.T.conjugate())


@verbose
def rap_music(evoked, forward, noise_cov, n_dipoles=5, return_residual=False,
              picks=None, verbose=None):
    """RAP-MUSIC source localization method.

    Compute Recursively Applied and Projected MUltiple SIgnal Classification
    (RAP-MUSIC) on evoked data.

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked data to localize.
    forward : instance of Forward
        Forward operator.
    noise_cov : instance of Covariance
        The noise covariance.
    n_dipoles : int
        The number of dipoles to look for. The default value is 5.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    dipoles : list of instance of Dipole
        The dipole fits.
    residual : instance of Evoked
        The residual a.k.a. data not explained by the dipoles.
        Only returned if return_residual is True.

    See Also
    --------
    mne.fit_dipole

    Notes
    -----
    The references are:

        J.C. Mosher and R.M. Leahy. 1999. Source localization using recursively
        applied and projected (RAP) MUSIC. Signal Processing, IEEE Trans. 47, 2
        (February 1999), 332-340.
        DOI=10.1109/78.740118 http://dx.doi.org/10.1109/78.740118

        Mosher, J.C.; Leahy, R.M., EEG and MEG source localization using
        recursively applied (RAP) MUSIC, Signals, Systems and Computers, 1996.
        pp.1201,1207 vol.2, 3-6 Nov. 1996
        doi: 10.1109/ACSSC.1996.599135

    .. versionadded:: 0.9.0
    """
    info = evoked.info
    data = evoked.data
    times = evoked.times

    picks = _setup_picks(picks, info, forward, noise_cov)

    data = data[picks]

    dipoles, explained_data = _apply_rap_music(data, info, times, forward,
                                               noise_cov, n_dipoles,
                                               picks, return_residual)

    if return_residual:
        residual = evoked.copy()
        selection = [info['ch_names'][p] for p in picks]

        residual = pick_channels_evoked(residual,
                                        include=selection)
        residual.data -= explained_data
        active_projs = [p for p in residual.info['projs'] if p['active']]
        for p in active_projs:
            p['active'] = False
        residual.add_proj(active_projs, remove_existing=True)
        residual.apply_proj()
        return dipoles, residual
    else:
        return dipoles
