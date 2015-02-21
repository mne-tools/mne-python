"""Compute a Recursively Applied and Projected MUltiple
Signal Classification (RAP-MUSIC).
"""

# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..io.pick import pick_channels_evoked
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import _check_reference
from ..cov import compute_whitener
from ..source_estimate import _make_stc
from ..utils import logger, verbose
from ._lcmv import _prepare_beamformer_input


@verbose
def _apply_rap_music(data, info, tmin, forward, noise_cov, label=None,
                     r=15, n_sources=5, picks=None, pick_ori=None,
                     return_residual=False, verbose=None):
    """RAP-MUSIC for evoked data

    Parameters
    ----------
    data : array or list / iterable
        Evoked data.
    info : dict
        Measurement info.
    tmin : float
        Time of first sample.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
    label : Label
        Restricts the rap-music solution to a given label.
        XXX: not implemented yet.
    r: int
        The dimension of the subspace spanning the signal.
        The default value is 15.
    n_sources: int
        The number of sources to estimate.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate
        Source time courses.
    residual_data : array
        Data explained by the sources. Computed only if return_residual
        is True.
    """
    is_free_ori, picks, ch_names, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)
    gain = G.copy()

    # Handle whitening + data covariance
    whitener, _ = compute_whitener(noise_cov, info, picks)
    if info['projs']:
        whitener = np.dot(whitener, proj)

    # whiten the leadfield and the data
    G = np.dot(whitener, G)
    data = np.dot(whitener, data)

    eig_values, eig_vectors = linalg.eigh(np.dot(data, data.T))
    phi_sig = eig_vectors[:, -r:]

    n_orient = 3 if is_free_ori else 1
    A = np.zeros((G.shape[0], n_sources))
    active_set = []

    G_proj = G
    phi_sig_proj = phi_sig

    for k in range(n_sources):
        subcorr_max = -1.
        source_idx = None
        for i_source in range(G.shape[1] // n_orient):
            Gk = G_proj[:, n_orient * i_source:
                        n_orient * i_source + n_orient]

            subcorr, ori = _compute_subcorr(Gk, phi_sig_proj)
            if subcorr > subcorr_max:
                subcorr_max = subcorr
                source_idx = i_source
                A[:, k] = np.dot(Gk, ori)

        active_set.append(source_idx)
        logger.info("source %s found: p = %s" % (k + 1, active_set[-1]))
        if n_orient == 3:
            logger.info("ori = %s %s %s" % (ori[0], ori[1], ori[2]))

        projection = _compute_proj(A[:, :k + 1])
        G_proj = np.dot(projection, G)
        phi_sig_proj = np.dot(projection, phi_sig)

    subject = _subject_from_forward(forward)
    sol = linalg.lstsq(A, data)[0]

    active_set = np.sort(active_set)
    residual_data = None
    if return_residual:
        residual_data = np.dot(gain[:, active_set], sol)

    # Pick source orientation normal to cortical surface
    # XXX : still wrong:
    # if pick_ori == 'normal':
    #     sol = sol[:, 2::3]

    vertno[1] = vertno[1][active_set[active_set > vertno[0].size]
                          - vertno[0].size]
    vertno[0] = vertno[0][active_set[active_set <= vertno[0].size]]

    tstep = 1.0 / info['sfreq']

    return _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                     subject=subject), residual_data


def _compute_subcorr(G, phi_sig):
    """ Compute the subspace correlation
    """
    # if G.shape[1] == 1:
    #     Gh = G.T.conjugate()
    #     phi_sigh = phi_sig.T.conjugate()
    #     subcorr = np.dot(np.dot(Gh, phi_sig), np.dot(phi_sigh, G))
    #     return np.sqrt(subcorr / np.dot(Gh, G)), np.ones(1)
    # else:
    Ug = linalg.qr(G, mode='economic')[0]
    Ugh = Ug.T.conjugate()
    phi_sigh = phi_sig.T.conjugate()
    subcorr = np.dot(np.dot(Ugh, phi_sig), np.dot(phi_sigh, Ug))

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
def rap_music(evoked, forward, noise_cov, label=None, r=15,
              n_sources=5, pick_ori=None, return_residual=False,
              verbose=None):
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
    label : Label
        Restricts the RAP-MUSIC solution to a given label.
        XXX: not implemented yet.
    r: int
        The dimension of the subspace spanning the signal.
        The default value is 15.
    n_sources: int
        The number of sources to look for. Default value is 5.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate
        Source time courses
    residual : Evoked
        The residual a.k.a. data not explained by the sources.
        Only returned if return_residual is True.

    Notes
    -----
    The reference is:
    J.C. Mosher and R.M. Leahy. 1999. Source localization using recursively
    applied and projected (RAP) MUSIC. Trans. Sig. Proc. 47, 2
    (February 1999), 332-340.
    DOI=10.1109/78.740118 http://dx.doi.org/10.1109/78.740118
    """
    _check_reference(evoked)

    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    stc, D = _apply_rap_music(data, info, tmin, forward, noise_cov,
                              label, r, n_sources, pick_ori=pick_ori,
                              return_residual=return_residual)

    if return_residual:
        residual = evoked.copy()
        residual = pick_channels_evoked(residual,
                                        include=info['ch_names'])
        residual.data -= D

        return stc, residual
    else:
        return stc
