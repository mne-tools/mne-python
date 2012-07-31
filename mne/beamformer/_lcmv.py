"""Compute Linearly constrained minimum variance (LCMV) beamformer.
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..fiff.constants import FIFF
from ..fiff.proj import make_projector
from ..fiff.pick import pick_types, pick_channels_forward, pick_channels_cov
from ..minimum_norm.inverse import _make_stc, _get_vertno, combine_xyz
from ..cov import compute_whitener


def _apply_lcmv(data, info, tmin, forward, noise_cov, data_cov, reg,
                label=None, picks=None):
    """ LCMV beamformer for evoked or raw data

    Parameters
    ----------
    data : array
        Sensor space data
    info : dict
        Measurement info
    tmin : float
        Time of first sample
    forward : dict
        Forward operator
    noise_cov : Covariance
        The noise covariance
    data_cov : Covariance
        The data covariance
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label
    picks : array of int
        Indices (in info) of data channels

    Returns
    -------
    stc : dict
        Source time courses
    """

    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, exclude=info['bads'])

    if len(data) != len(picks):
        raise ValueError('data and picks must have the same length')

    ch_names = [info['ch_names'][k] for k in picks]

    # restrict forward solution to selected channels
    forward = pick_channels_forward(forward, include=ch_names)

    # get gain matrix (forward operator)
    if label is not None:
        if forward['src'][0]['type'] != 'surf':
            return Exception('Labels are only supported with surface '
                             'source spaces')

        vertno_fwd = _get_vertno(forward['src'])
        if label['hemi'] == 'lh':
            vertno_sel = np.intersect1d(vertno_fwd[0], label['vertices'])
            idx_sel = np.searchsorted(vertno_fwd[0], vertno_sel)
            vertno = [vertno_sel, np.empty(0, dtype=vertno_sel.dtype)]
        elif label['hemi'] == 'rh':
            vertno_sel = np.intersect1d(vertno_fwd[1], label['vertices'])
            idx_sel = len(vertno_fwd[0]) + np.searchsorted(vertno_fwd[1],
                                                           vertno_sel)
            vertno = [np.empty(0, dtype=vertno_sel.dtype), vertno_sel]
        else:
            raise Exception("Unknown hemisphere type")

        if is_free_ori:
            idx_sel_free = np.zeros(3 * len(idx_sel), dtype=idx_sel.dtype)
            for i in range(3):
                idx_sel_free[i::3] = 3 * idx_sel + i
            idx_sel = idx_sel_free

        G = forward['sol']['data'][:, idx_sel]
    else:
        vertno = _get_vertno(forward['src'])
        G = forward['sol']['data']

    # Handle SSPs
    proj, ncomp, _ = make_projector(info['projs'], ch_names)
    M = np.dot(proj, data)
    G = np.dot(proj, G)

    # Handle whitening + data covariance
    W, _ = compute_whitener(noise_cov, info, picks)

    # whiten data and leadfield
    M = np.dot(W, M)
    G = np.dot(W, G)

    # Apply SSPs + whitener to data covariance
    data_cov = pick_channels_cov(data_cov, include=ch_names)
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

    tstep = 1.0 / info['sfreq']
    stc = _make_stc(sol, tmin, tstep, vertno)
    print '[done]'

    return stc


def lcmv(evoked, forward, noise_cov, data_cov, reg=0.01, label=None):
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
    label : Label
        Restricts the LCMV solution to a given label

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

    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    stc = _apply_lcmv(data, info, tmin, forward, noise_cov, data_cov, reg,
                      label)

    return stc


def lcmv_raw(raw, forward, noise_cov, data_cov, reg=0.01, label=None,
             start=None, stop=None, picks=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on raw data.

    NOTE : This implementation is heavilly tested so please
    report any issue or suggestions.

    Parameters
    ----------
    raw : mne.fiff.Raw
        Raw data to invert
    forward : dict
        Forward operator
    noise_cov : Covariance
        The noise covariance
    data_cov : Covariance
        The data covariance
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label
    start : int
        Index of first time sample (index not time is seconds)
    stop : int
        Index of first time sample not to include (index not time is seconds)
    picks: array of int
        Channel indices in raw to use for beamforming (if None all channels
        are used)

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

    info = raw.info

    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, exclude=info['bads'])

    data, times = raw[picks, start:stop]
    tmin = times[0]

    stc = _apply_lcmv(data, info, tmin, forward, noise_cov, data_cov, reg,
                      label, picks)

    return stc

