import warnings

import numpy as np
from scipy import linalg

from ..io.constants import FIFF
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
    """ RAP-MUSIC for evoked data

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
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
        XXX: 'max-power' not implemented
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate
        Source time courses.
    D : array
        Data explained by the sources. Computed only if return_residual
        is True.
    """
    is_free_ori, picks, ch_names, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    # Handle whitening + data covariance
    whitener, _ = compute_whitener(noise_cov, info, picks)

    # whiten the leadfield
    G = np.dot(whitener, G)
    data = np.dot(whitener, data)

    # Pick source orientation normal to cortical surface
    if pick_ori == 'normal':
        G = G[:, 2::3]
        is_free_ori = False

    eig_values, eig_vectors = linalg.eigh(np.dot(data, data.T))
    phi_sig = eig_vectors[:, -r:]

    n_orient = 3 if is_free_ori else 1
    A = np.zeros((G.shape[0], n_sources))
    active_set = np.concatenate(-np.ones((n_sources, 1), dtype='int'))

    G_proj = np.dot(np.identity(G.shape[0]), G)
    phi_sig_proj = np.dot(np.identity(phi_sig.shape[0]), phi_sig)

    for k in range(n_sources):
        subcorr_max = -1
        for i_source in range(G.shape[1] // n_orient):
            if n_orient == 1:
                Gk = G_proj[:, i_source]
            else:
                Gk = G_proj[:, n_orient * i_source:
                            n_orient * i_source + n_orient]

            subcorr, ori = _compute_subcorr(Gk, phi_sig_proj)
            if subcorr > subcorr_max:
                subcorr_max, active_set[k] = subcorr, i_source
                A[:, k] = np.dot(Gk, ori)

        logger.info("source %s found" % (k + 1))
        if n_orient == 3:
            logger.info("p = %s, ori = %s %s %s" % (active_set[k],
                                                    ori[0], ori[1],
                                                    ori[2]))

        projection = _compute_proj(A[:, :k + 1])
        G_proj = np.dot(projection, G)
        phi_sig_proj = np.dot(projection, phi_sig)

    subject = _subject_from_forward(forward)
    sol = np.dot(linalg.pinv(A), data)

    active_set = np.sort(active_set)
    if return_residual:
        _, _, _, _, _, G = _prepare_beamformer_input(info, forward, label,
                                                     picks, pick_ori)
        D = np.dot(G[:, active_set], sol)
    else:
        D = []

    vertno[1] = vertno[1][active_set[active_set > vertno[0].size]
                          - vertno[0].size]
    vertno[0] = vertno[0][active_set[active_set <= vertno[0].size]]

    tstep = 1.0 / info['sfreq']

    return _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                     subject=subject), D


def _compute_subcorr(G, phi_sig):
    """ Compute the subspace correlation
    """
    if len(G.shape) == 1:
        Gh = G.T.conjugate()
        phi_sigh = phi_sig.T.conjugate()
        subcorr = np.dot(np.dot(Gh, phi_sig), np.dot(phi_sigh, G))
        return np.sqrt(subcorr / np.dot(Gh, G)), 1
    else:
        Ug = np.linalg.qr(G, mode='reduced')[0]
        Ugh = Ug.T.conjugate()
        phi_sigh = phi_sig.T.conjugate()
        subcorr = np.dot(np.dot(Ugh, phi_sig), np.dot(phi_sigh, Ug))

        eig = linalg.eigh(subcorr)
        return np.sqrt(eig[0][-1]), eig[1][:, -1]


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
    """Recursively Applied and Projected MUltiple SIgnal Classification.
    RAP-MUSIC

    Compute RAP-MUSIC on evoked data.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to invert.
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
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
        XXX: 'max_power' not implemented
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate
        Source time courses
    residual : instance of Evoked
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
