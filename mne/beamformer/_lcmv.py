"""Compute Linearly constrained minimum variance (LCMV) beamformer.
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..fiff.constants import FIFF
from ..fiff.proj import make_projector
from ..fiff.pick import pick_types, pick_channels_forward
from ..minimum_norm.inverse import _make_stc, _get_vertno, combine_xyz
from ..cov import compute_whitener


def lcmv(evoked, forward, noise_cov, data_cov, reg=0.01):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on evoked data.

    NOTE : This implementation is heavilly tested so please
    report any issue or suggestions.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to invert
    forward : dict
        Forward operator
    noise_cov : Covariance
        The noise covariance
    data_cov : Covariance
        The data covariance
    reg : float
        The regularization for the whitened data covariance.

    Returns
    -------
    stc : dict
        Source time courses

    Notes
    -----
    The original reference is:
    Van Veen et al. Localization of brain electrical activity via linearly
    constrained minimum variance spatial filtering.
    Biomedical Engineering (1997) vol. 44 (9) pp. 867--880
    """
    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    picks = pick_types(evoked.info, meg=True, eeg=True,
                       exclude=evoked.info['bads'])
    ch_names = [evoked.ch_names[k] for k in picks]

    forward = pick_channels_forward(forward, include=ch_names)

    M = evoked.data
    G = forward['sol']['data']

    # Handle SSPs
    proj, ncomp, _ = make_projector(evoked.info['projs'], evoked.ch_names)
    M = np.dot(proj, M)
    G = np.dot(proj, G)

    # Handle whitening + data covariance
    W, _ = compute_whitener(noise_cov, evoked.info, picks)

    # whiten data and leadfield
    M = np.dot(W, M)
    G = np.dot(W, G)

    # Apply SSPs + whitener to data covariance
    Cm = data_cov['data']
    Cm = np.dot(proj, np.dot(Cm, proj.T))
    Cm = np.dot(W, np.dot(Cm, W.T))

    # Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
    Cm_inv = linalg.pinv(Cm, reg)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)
        Wk[:] = np.dot(linalg.pinv(Ck, 0.01), Wk)

    # noise normalization
    noise_norm = np.sum(W ** 2, axis=1)
    if is_free_ori:
        noise_norm = np.sum(np.reshape(noise_norm, (-1, 3)), axis=1)

    noise_norm = np.sqrt(noise_norm)

    sol = np.dot(W, M)

    if is_free_ori:
        print 'combining the current components...',
        sol = combine_xyz(sol)

    sol /= noise_norm[:, None]

    tstep = 1.0 / evoked.info['sfreq']
    tmin = float(evoked.first) / evoked.info['sfreq']
    vertno = _get_vertno(forward['src'])
    stc = _make_stc(sol, tmin, tstep, vertno)
    print '[done]'

    return stc
