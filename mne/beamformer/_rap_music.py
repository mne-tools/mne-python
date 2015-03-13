"""Compute a Recursively Applied and Projected MUltiple
Signal Classification (RAP-MUSIC).
"""

# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..io.pick import pick_channels_evoked
from ..minimum_norm.inverse import _check_reference
from ..cov import compute_whitener
from ..forward.forward import _block_diag
from ..utils import logger, verbose
from ..dipole import Dipole
from ._lcmv import _prepare_beamformer_input, _setup_picks


@verbose
def _apply_rap_music(data, info, times, forward, noise_cov,
                     signal_ndim=15, n_dipoles=5, picks=None,
                     return_explained_data=False, verbose=None):
    """RAP-MUSIC for evoked data

    Parameters
    ----------
    data : array or list / iterable
        Evoked data.
    info : dict
        Measurement info.
    times : float
        Times.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
    signal_ndim : int
        The dimension of the subspace spanning the signal.
        The default value is 15.
    n_dipoles : int
        The number of dipoles to estimate.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    return_explained_data : bool
        If True, the explained data is returned as an array.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    dipoles : list of instances of Dipole
        The dipole fits.
    explained_data : array
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
    phi_sig = eig_vectors[:, -signal_ndim:]

    n_orient = 3 if is_free_ori else 1
    A = np.zeros((G.shape[0], n_dipoles))
    active_set = []
    oris = []

    G_proj = G
    phi_sig_proj = phi_sig

    for k in range(n_dipoles):
        subcorr_max = -1.
        source_idx = None
        for i_source in range(G.shape[1] // n_orient):
            Gk = G_proj[:, n_orient * i_source:
                        n_orient * i_source + n_orient]

            subcorr, ori = _compute_subcorr(Gk, phi_sig_proj)
            if subcorr > subcorr_max:
                subcorr_max = subcorr
                source_idx = i_source
                if n_orient == 1:
                    ori = [forward['src'][0]['nn'][vertno[0][i_source]]
                           if i_source <= vertno[0].size else
                           forward['src'][1]['nn'][vertno[1][i_source -
                                                             vertno[0].size]]]
                else:
                    ori = np.dot(forward['source_nn'][n_orient * i_source:
                                                      n_orient * i_source +
                                                      n_orient, :], ori)

                if ori[-1] < 0: # make sure ori is relative to surface ori
                    ori *= -1.
                source_ori = ori
                A[:, k] = np.dot(Gk, ori)


        active_set.append(source_idx)
        oris.append(source_ori)

        logger.info("source %s found: p = %s" % (k + 1, active_set[-1]))
        if n_orient == 3:
            logger.info("ori = %s %s %s" % (source_ori[0], source_ori[1],
                                            source_ori[2]))

        projection = _compute_proj(A[:, :k + 1])
        G_proj = np.dot(projection, G)
        phi_sig_proj = np.dot(projection, phi_sig)

    sol = linalg.lstsq(A, data)[0]

    active_set = np.sort(active_set)
    explained_data = None
    if return_explained_data:
        explained_data = np.dot(gain[:, active_set], sol)

    vertno[1] = vertno[1][active_set[active_set > vertno[0].size] -
                          vertno[0].size]
    vertno[0] = vertno[0][active_set[active_set <= vertno[0].size]]

    tstep = 1.0 / info['sfreq']

    return _make_dipoles(times, tstep, forward['src'], vertno, active_set,
                         oris, sol), explained_data


def _make_dipoles(times, tstep, src, vertno, active_set, oris, sol):
    if vertno[0].size == 0:
        pos = np.array(src[1]['rr'][vertno[1]])
    elif vertno[1].size == 0:
        pos = np.array(src[0]['rr'][vertno[0]])
    else:
        pos = np.array((src[0]['rr'][vertno[0]][0],
                        src[1]['rr'][vertno[1]][0]))
    amplitude = sol * 1e09
    ori = np.array(oris)
    gof = []

    dipoles = []
    for i_dip in range(pos.shape[0]):
        dipoles.append(Dipole(times * 1e3, pos[i_dip], amplitude[i_dip],
                              ori[i_dip], gof))

    return dipoles


def _compute_subcorr(G, phi_sig):
    """ Compute the subspace correlation
    """
    Ug = linalg.qr(G, mode='economic')[0]
    tmp = np.dot(Ug.T.conjugate(), phi_sig)
    subcorr = np.dot(tmp, tmp.T.conjugate()).real
    eig_vals, eig_vecs = linalg.eigh(subcorr)
    return np.sqrt(eig_vals[-1]), eig_vecs[:, -1]


def _compute_proj(A):
    """ Compute the orthogonal projection operation for
    a manifold vector A.
    """
    Ah = A.T.conjugate()
    I = np.identity(A.shape[0])
    return I - np.dot(np.dot(A, linalg.pinv(np.dot(Ah, A))), Ah)


@verbose
def rap_music(evoked, forward, noise_cov, signal_ndim=15, n_dipoles=5,
              return_residual=False, picks=None, verbose=None):
    """RAP-MUSIC source localization method.

    Compute Recursively Applied and Projected MUltiple SIgnal Classification
    (RAP-MUSIC) on evoked data.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to localize.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
    signal_ndim: int
        The dimension of the subspace spanning the signal.
        The default value is 15.
    n_dipoles: int
        The number of dipoles to look for. Default value is 5.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    dipoles : list of instance of Dipole
        The dipole fits.
    residual : Evoked
        The residual a.k.a. data not explained by the dipoles.
        Only returned if return_residual is True.

    Notes
    -----
    The reference is:
    J.C. Mosher and R.M. Leahy. 1999. Source localization using recursively
    applied and projected (RAP) MUSIC. Signal Processing, IEEE Trans. 47, 2
    (February 1999), 332-340.
    DOI=10.1109/78.740118 http://dx.doi.org/10.1109/78.740118
    """
    _check_reference(evoked)

    info = evoked.info
    data = evoked.data
    times = evoked.times

    picks = _setup_picks(picks, info, forward, noise_cov)

    data = data[picks]

    dipoles, explained_data = _apply_rap_music(data, info, times, forward,
                                               noise_cov, signal_ndim,
                                               n_dipoles, picks,
                                               return_residual)

    if return_residual:
        residual = evoked.copy()
        selection = np.array(info['ch_names'])[picks]

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
