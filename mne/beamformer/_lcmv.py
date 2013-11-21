"""Compute Linearly constrained minimum variance (LCMV) beamformer.
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

import logging
logger = logging.getLogger('mne')

from ..fiff.constants import FIFF
from ..fiff.proj import make_projector
from ..fiff.pick import pick_types, pick_channels_forward, pick_channels_cov
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import _get_vertno, combine_xyz
from ..cov import compute_whitener
from ..source_estimate import SourceEstimate
from ..source_space import label_src_vertno_sel
from .. import verbose


@verbose
def _apply_lcmv(data, info, tmin, forward, noise_cov, data_cov, reg,
                label=None, picks=None, pick_ori=None, verbose=None):
    """ LCMV beamformer for evoked data, single epochs, and raw data

    Parameters
    ----------
    data : array or list / iterable
        Sensor space data. If data.ndim == 2 a single observation is assumed
        and a single stc is returned. If data.ndim == 3 or if data is
        a list / iterable, a list of stc's is returned.
    info : dict
        Measurement info.
    tmin : float
        Time of first sample.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label.
    picks : array of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    pick_ori : None | 'max-power'
        If 'max-power', the source orientation that maximizes output source
        power is chosen.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate (or list of SourceEstimate)
        Source time courses.
    """

    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    if pick_ori in ['max-power'] and not is_free_ori:
        raise ValueError('Max-power orientation can only be picked '
                         'when a forward operator with free orientation is '
                         'used.')

    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, exclude='bads')

    ch_names = [info['ch_names'][k] for k in picks]

    # restrict forward solution to selected channels
    forward = pick_channels_forward(forward, include=ch_names)

    # get gain matrix (forward operator)
    if label is not None:
        vertno, src_sel = label_src_vertno_sel(label, forward['src'])

        if is_free_ori:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        G = forward['sol']['data'][:, src_sel]
    else:
        vertno = _get_vertno(forward['src'])
        G = forward['sol']['data']

    # Handle SSPs
    proj, ncomp, _ = make_projector(info['projs'], ch_names)
    G = np.dot(proj, G)

    # Handle whitening + data covariance
    whitener, _ = compute_whitener(noise_cov, info, picks)

    # whiten the leadfield
    G = np.dot(whitener, G)

    # Apply SSPs + whitener to data covariance
    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']
    Cm = np.dot(proj, np.dot(Cm, proj.T))
    Cm = np.dot(whitener, np.dot(Cm, whitener.T))

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

        # Find source orientation maximizing output source power
        if pick_ori == 'max-power':
            eig_vals, eig_vecs = linalg.eigh(Ck)

            # Choosing the eigenvector associated with the middle eigenvalue.
            # The middle and not the minimal eigenvalue is used because MEG is
            # insensitive to one (radial) of the three dipole orientations and
            # therefore the smallest eigenvalue reflects mostly noise.
            for i in range(3):
                if i != eig_vals.argmax() and i != eig_vals.argmin():
                    idx_middle = i

            # TODO: The eigenvector associated with the smallest eigenvalue
            # should probably be used when using combined EEG and MEG data
            max_ori = eig_vecs[:, idx_middle]

            Wk[:] = np.dot(max_ori, Wk)
            Ck = np.dot(max_ori, np.dot(Ck, max_ori))
            is_free_ori = False

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

    # Pick source orientation maximizing output source power
    if pick_ori == 'max-power':
        W = W[0::3]

    # noise normalization
    noise_norm = np.sum(W ** 2, axis=1)
    if is_free_ori:
        noise_norm = np.sum(np.reshape(noise_norm, (-1, 3)), axis=1)
    noise_norm = np.sqrt(noise_norm)

    if not is_free_ori:
        W /= noise_norm[:, None]

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        return_single = True
    else:
        return_single = False

    subject = _subject_from_forward(forward)
    for i, M in enumerate(data):
        if len(M) != len(picks):
            raise ValueError('data and picks must have the same length')

        if not return_single:
            logger.info("Processing epoch : %d" % (i + 1))

        # SSP and whitening
        M = np.dot(proj, M)
        M = np.dot(whitener, M)

        # project to source space using beamformer weights

        if is_free_ori:
            sol = np.dot(W, M)
            logger.info('combining the current components...')
            sol = combine_xyz(sol)
            sol /= noise_norm[:, None]
        else:
            # Linear inverse: do computation here or delayed
            if M.shape[0] < W.shape[0] and pick_ori != 'max-power':
                sol = (W, M)
            else:
                sol = np.dot(W, M)
            if pick_ori == 'max-power':
                sol = np.abs(sol)

        tstep = 1.0 / info['sfreq']
        yield SourceEstimate(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                             subject=subject)

    logger.info('[done]')


@verbose
def lcmv(evoked, forward, noise_cov, data_cov, reg=0.01, label=None,
         pick_ori=None, verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on evoked data.

    NOTE : This implementation has not been heavilly tested so please
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
    pick_ori : None | 'max-power'
        If 'max-power', the source orientation that maximizes output source
        power is chosen.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate
        Source time courses

    Notes
    -----
    The original reference is:
    Van Veen et al. Localization of brain electrical activity via linearly
    constrained minimum variance spatial filtering.
    Biomedical Engineering (1997) vol. 44 (9) pp. 867--880

    The reference for finding the max-power orientation is:
    Sekihara et al. Asymptotic SNR of scalar and vector minimum-variance
    beamformers for neuromagnetic source reconstruction.
    Biomedical Engineering (2004) vol. 51 (10) pp. 1726--34
    """

    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    stc = _apply_lcmv(data, info, tmin, forward, noise_cov, data_cov, reg,
                      label, pick_ori=pick_ori).next()

    return stc


@verbose
def lcmv_epochs(epochs, forward, noise_cov, data_cov, reg=0.01, label=None,
                pick_ori=None, return_generator=False, verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on single trial data.

    NOTE : This implementation has not been heavilly tested so please
    report any issue or suggestions.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label.
    pick_ori : None | 'max-power'
        If 'max-power', the source orientation that maximizes output source
        power is chosen.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc: list | generator of SourceEstimate
        The source estimates for all epochs

    Notes
    -----
    The original reference is:
    Van Veen et al. Localization of brain electrical activity via linearly
    constrained minimum variance spatial filtering.
    Biomedical Engineering (1997) vol. 44 (9) pp. 867--880

    The reference for finding the max-power orientation is:
    Sekihara et al. Asymptotic SNR of scalar and vector minimum-variance
    beamformers for neuromagnetic source reconstruction.
    Biomedical Engineering (2004) vol. 51 (10) pp. 1726--34
    """

    info = epochs.info
    tmin = epochs.times[0]

    # use only the good data channels
    picks = pick_types(info, meg=True, eeg=True, exclude='bads')
    data = epochs.get_data()[:, picks, :]

    stcs = _apply_lcmv(data, info, tmin, forward, noise_cov, data_cov, reg,
                       label, pick_ori=pick_ori)

    if not return_generator:
        stcs = [s for s in stcs]

    return stcs


@verbose
def lcmv_raw(raw, forward, noise_cov, data_cov, reg=0.01, label=None,
             start=None, stop=None, picks=None, pick_ori=None, verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on raw data.

    NOTE : This implementation has not been heavilly tested so please
    report any issue or suggestions.

    Parameters
    ----------
    raw : mne.fiff.Raw
        Raw data to invert.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label.
    start : int
        Index of first time sample (index not time is seconds).
    stop : int
        Index of first time sample not to include (index not time is seconds).
    picks : array of int
        Channel indices in raw to use for beamforming (if None all channels
        are used except bad channels).
    pick_ori : None | 'max-power'
        If 'max-power', the source orientation that maximizes output source
        power is chosen.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate
        Source time courses

    Notes
    -----
    The original reference is:
    Van Veen et al. Localization of brain electrical activity via linearly
    constrained minimum variance spatial filtering.
    Biomedical Engineering (1997) vol. 44 (9) pp. 867--880

    The reference for finding the max-power orientation is:
    Sekihara et al. Asymptotic SNR of scalar and vector minimum-variance
    beamformers for neuromagnetic source reconstruction.
    Biomedical Engineering (2004) vol. 51 (10) pp. 1726--34
    """

    info = raw.info

    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, exclude='bads')

    data, times = raw[picks, start:stop]
    tmin = times[0]

    stc = _apply_lcmv(data, info, tmin, forward, noise_cov, data_cov, reg,
                      label, picks, pick_ori).next()

    return stc
