"""Dynamic Imaging of Coherent Sources (DICS).
"""

# Authors: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

import logging
logger = logging.getLogger('mne')

from ..fiff.constants import FIFF
from ..fiff.proj import make_projector
from ..fiff.pick import pick_types, pick_channels_forward
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import _get_vertno, combine_xyz
from ..source_estimate import SourceEstimate
from ..source_space import label_src_vertno_sel
from ..time_frequency import CrossSpectralDensity
from .. import verbose


@verbose
def _apply_dics(data, info, tmin, forward, noise_csd, data_csd, reg=0.1,
                label=None, picks=None, pick_ori=None, verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Calculate the DICS spatial filter based on a given cross-spectral
    density object and return estimates of source activity based on given data.

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
    noise_csd : CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : CrossSpectralDensity
        The data cross-spectral density.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    picks : array of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate (or list of SourceEstimate)
        Source time courses.
    """
    # TODO: DICS, in the original 2001 paper, used a free orientation
    # beamformer, however selection of the max-power orientation was also
    # employed depending on whether a dominant component was present

    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    # DIFF: LCMV has 'max-power' here in addition to 'normal'
    if pick_ori == 'normal' and not is_free_ori:
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with free orientation is used.')
    if pick_ori == 'normal' and not forward['surf_ori']:
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator oriented in surface coordinates is '
                         'used.')
    if pick_ori == 'normal' and not forward['src'][0]['type'] == 'surf':
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with a surface-based source space '
                         'is used.')

    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, exclude='bads')

    ch_names = [info['ch_names'][k] for k in picks]

    # Restrict forward solution to selected channels
    forward = pick_channels_forward(forward, include=ch_names)

    # Get gain matrix (forward operator)
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

    # Apply SSPs
    proj = None
    if 'projs' in info and len(info['projs']) > 0:
        proj, ncomp, _ = make_projector(info['projs'], ch_names)
        G = np.dot(proj, G)

    # DIFF: LCMV applies SSPs and whitener to data covariance at this point,
    # here we only read in the cross-spectral density matrix - the data used in
    # its calculation already had SSPs applied and we will use the noise CSD
    # matrix in noise normalization instead of whitening the data
    Cm = data_csd.data

    # Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
    Cm_inv = linalg.pinv(Cm, reg)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient

    for k in xrange(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)

        # DIFF: LCMV calculates 'max-power' orientation here
        # TODO: max-power is not used in this example, however DICS does employ
        # orientation picking when one eigen value is much larger than the
        # other

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

        # Noise normalization
        # DIFF: LCMV prepares noise normalization outside of the loop
        # DIFF: noise_norm is not complex in LCMV
        noise_norm = np.dot(np.dot(Wk.conj(), noise_csd.data), Wk.T)
        noise_norm = np.abs(noise_norm).trace()
        Wk /= np.sqrt(noise_norm)

    # DIFF: LCMV picks 'max-power' orientation here

    # DIFF: LCMV prepares noise normalization here and it doesn't involve the
    # noise covariance

    # Pick source orientation normal to cortical surface
    if pick_ori == 'normal':
        W = W[2::3]
        is_free_ori = False

    # DIFF: LCMV applies noise normalization for fixed orientation here

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

        # DIFF: LCMV applies data whitening here
        # Apply SSPs
        if proj is not None:
            M = np.dot(proj, M)

        # project to source space using beamformer weights

        if is_free_ori:
            sol = np.dot(W, M)
            logger.info('combining the current components...')
            sol = combine_xyz(sol)
            # DIFF: LCMV applies noise normalization for free orientation here
        else:
            # Linear inverse: do not delay compuation due to non-linear abs
            sol = np.dot(W, M)

        tstep = 1.0 / info['sfreq']
        if np.iscomplexobj(sol):
            sol = np.abs(sol)  # XXX : STC cannot contain (yet?) complex values
        yield SourceEstimate(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                             subject=subject)

    logger.info('[done]')


@verbose
def dics(evoked, forward, noise_csd, data_csd, reg=0.01, label=None,
         pick_ori=None, verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Compute a Dynamic Imaging of Coherent Sources (DICS) beamformer
    on evoked data and return estimates of source time courses.

    NOTE : Fixed orientation forward operators will result in complex time
    courses in which case absolute values will be  returned. Therefore the
    orientation will no longer be fixed.

    NOTE : This implementation has not been heavilly tested so please
    report any issues or suggestions.

    Parameters
    ----------
    evoked : Evooked
        Evoked data.
    forward : dict
        Forward operator.
    noise_csd : CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : CrossSpectralDensity
        The data cross-spectral density.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc: SourceEstimate
        Source time courses

    Notes
    -----
    The original reference is:
    Gross et al. Dynamic imaging of coherent sources: Studying neural
    interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    """
    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    stc = _apply_dics(data, info, tmin, forward, noise_csd, data_csd, reg=reg,
                      label=label, pick_ori=pick_ori)
    return stc.next()


@verbose
def dics_epochs(epochs, forward, noise_csd, data_csd, reg=0.01, label=None,
                pick_ori=None, return_generator=False, verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Compute a Dynamic Imaging of Coherent Sources (DICS) beamformer
    on single trial data and return estimates of source time courses.

    NOTE : Fixed orientation forward operators will result in complex time
    courses in which case absolute values will be  returned. Therefore the
    orientation will no longer be fixed.

    NOTE : This implementation has not been heavilly tested so please
    report any issues or suggestions.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    forward : dict
        Forward operator.
    noise_csd : CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : CrossSpectralDensity
        The data cross-spectral density.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
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
    Gross et al. Dynamic imaging of coherent sources: Studying neural
    interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    """

    info = epochs.info
    tmin = epochs.times[0]

    # use only the good data channels
    picks = pick_types(info, meg=True, eeg=True, exclude='bads')
    data = epochs.get_data()[:, picks, :]

    stcs = _apply_dics(data, info, tmin, forward, noise_csd, data_csd, reg=reg,
                       label=label, pick_ori=pick_ori)

    if not return_generator:
        stcs = list(stcs)

    return stcs


@verbose
def dics_source_power(info, forward, noise_csds, data_csds, freqs, reg=0.01,
                      label=None, pick_ori=None, verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Calculate source power in time and frequency windows specified in the
    calculation of the data cross-spectral density matrix or matrices. Source
    power is normalized by noise power.

    NOTE : This implementation has not been heavilly tested so please
    report any issues or suggestions.

    Parameters
    ----------
    info : dict
        Measurement info, e.g. epochs.info.
    forward : dict
        Forward operator.
    noise_csds : CrossSpectralDensity or list of CrossSpectralDensity
        The noise cross-spectral density matrix for a single frequency or a
        list of matrices for multiple frequencies.
    data_csds : CrossSpectralDensity or list of CrossSpectralDensity
        The data cross-spectral density matrix for a single frequecy or a list
        of matrices for multiple frequencies.
    freqs : array of float
        Frequencies of interest for which the CSD matrices have been computed.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc: SourceEstimate
        Source power with frequency instead of time.

    Notes
    -----
    The original reference is:
    Gross et al. Dynamic imaging of coherent sources: Studying neural
    interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    """
    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    if pick_ori == 'normal' and not is_free_ori:
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with free orientation is used.')
    if pick_ori == 'normal' and not forward['surf_ori']:
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator oriented in surface coordinates is '
                         'used.')
    if pick_ori == 'normal' and not forward['src'][0]['type'] == 'surf':
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with a surface-based source space '
                         'is used.')

    if np.array(data_csds).shape != np.array(noise_csds).shape:
        raise ValueError('One noise CSD matrix should be provided for each '
                         'data CSD matrix and vice versa. The CSD matrices '
                         'should have identical shape.')

    picks = pick_types(info, meg=True, eeg=True, exclude='bads')
    ch_names = [info['ch_names'][k] for k in picks]

    # Restrict forward solution to selected channels
    forward = pick_channels_forward(forward, include=ch_names)

    # Get gain matrix (forward operator)
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

    # Apply SSPs
    proj = None
    if 'projs' in info and len(info['projs']) > 0:
        proj, ncomp, _ = make_projector(info['projs'], ch_names)
        G = np.dot(proj, G)

    if isinstance(data_csds, CrossSpectralDensity) and\
            data_csds.data.ndim == 2:
        data_csds = [data_csds]
        noise_csds = [noise_csds]

    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    source_power = np.zeros((n_sources, len(data_csds)))

    for i, (data_csd, noise_csd) in enumerate(zip(data_csds, noise_csds)):
        Cm = data_csd.data

        # Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
        Cm_inv = linalg.pinv(Cm, reg)

        # Compute spatial filters
        W = np.dot(G.T, Cm_inv)
        for k in xrange(n_sources):
            Wk = W[n_orient * k: n_orient * k + n_orient]
            Gk = G[:, n_orient * k: n_orient * k + n_orient]
            Ck = np.dot(Wk, Gk)

            if is_free_ori:
                # Free source orientation
                Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
            else:
                # Fixed source orientation
                Wk /= Ck

            # Noise normalization
            noise_norm = np.dot(np.dot(Wk.conj(), noise_csd.data), Wk.T)
            noise_norm = np.abs(noise_norm).trace()

            # Calculating source power
            sp_temp = np.dot(np.dot(Wk.conj(), data_csd.data), Wk.T)
            sp_temp /= noise_norm
            if pick_ori == 'normal':
                source_power[k, i] = np.abs(sp_temp)[2, 2]
            else:
                source_power[k, i] = np.abs(sp_temp).trace()

    subject = _subject_from_forward(forward)
    if len(freqs) > 1:
        fstep = freqs[1] - freqs[0]
    else:
        fstep = 1  # dummy value
    fmin = freqs[0]
    return SourceEstimate(source_power, vertices=vertno, tmin=fmin,
                          tstep=fstep, subject=subject)

    logger.info('[done]')
