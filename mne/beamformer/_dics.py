"""Dynamic Imaging of Coherent Sources (DICS).
"""

# Authors: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import warnings

import numpy as np
from scipy import linalg

from ..utils import logger, verbose
from ..fiff.pick import pick_types
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import combine_xyz
from ..source_estimate import SourceEstimate
from ..time_frequency import CrossSpectralDensity
from ._lcmv import _prepare_beamformer_input


@verbose
def _apply_dics(data, info, tmin, forward, noise_csd, data_csd, reg,
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
    noise_csd : instance of CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : instance of CrossSpectralDensity
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

    is_free_ori, picks, _, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    Cm = data_csd.data

    # Calculating regularized inverse, equivalent to an inverse operation after
    # regularization: Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
    Cm_inv = linalg.pinv(Cm, reg)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient

    for k in xrange(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)

        # TODO: max-power is not implemented yet, however DICS does employ
        # orientation picking when one eigen value is much larger than the
        # other

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

        # Noise normalization
        noise_norm = np.dot(np.dot(Wk.conj(), noise_csd.data), Wk.T)
        noise_norm = np.abs(noise_norm).trace()
        Wk /= np.sqrt(noise_norm)

    # Pick source orientation normal to cortical surface
    if pick_ori == 'normal':
        W = W[2::3]
        is_free_ori = False

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

        # Apply SSPs
        if info['projs']:
            M = np.dot(proj, M)

        # project to source space using beamformer weights
        if is_free_ori:
            sol = np.dot(W, M)
            logger.info('combining the current components...')
            sol = combine_xyz(sol)
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
    noise_csd : instance of CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : instance of CrossSpectralDensity
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
    noise_csd : instance of CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : instance of CrossSpectralDensity
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
def dics_source_power(info, forward, noise_csds, data_csds, reg=0.01,
                      label=None, picks=None, pick_ori=None, verbose=None):
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
    noise_csds : instance or list of instances of CrossSpectralDensity
        The noise cross-spectral density matrix for a single frequency or a
        list of matrices for multiple frequencies.
    data_csds : instance or list of instances of CrossSpectralDensity
        The data cross-spectral density matrix for a single frequency or a list
        of matrices for multiple frequencies.
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

    if isinstance(data_csds, CrossSpectralDensity):
        data_csds = [data_csds]

    if isinstance(noise_csds, CrossSpectralDensity):
        noise_csds = [noise_csds]

    csd_shapes = lambda x: tuple(c.data.shape for c in x)
    if (csd_shapes(data_csds) != csd_shapes(noise_csds) or
       any([len(set(csd_shapes(c))) > 1 for c in [data_csds, noise_csds]])):
        raise ValueError('One noise CSD matrix should be provided for each '
                         'data CSD matrix and vice versa. All CSD matrices '
                         'should have identical shape.')

    frequencies = []
    for data_csd, noise_csd in zip(data_csds, noise_csds):
        # TODO: Is this check too restrictive? I.e. thick check won't fail only
        # when noise and data CSDs are calculated for exactly identical numbers
        # of samples, do we really need to require this?
        #if (data_csd.frequencies != noise_csd.frequencies).any:
        #    raise ValueError('Data and noise CSDs should be calculated at '
        #                     'identical frequencies')

        # If CSD is summed over multiple frequencies, take the average
        # frequency
        if(len(data_csd.frequencies) > 1):
            frequencies.append(np.mean(data_csd.frequencies))
        else:
            frequencies.append(data_csd.frequencies[0])
    fmin = frequencies[0]

    if len(frequencies) > 2:
        fstep = []
        for i in range(len(frequencies) - 1):
            fstep.append(frequencies[i+1] - frequencies[i])
        if not np.allclose(fstep, np.mean(fstep), 1e-5):
            warnings.warn('Uneven frequency spacing in CSD object, '
                          'frequencies in the resulting stc file will be '
                          'inaccurate.')
        fstep = fstep[0]
    elif len(frequencies) > 1:
        fstep = frequencies[1] - frequencies[0]
    else:
        fstep = 1  # dummy value

    is_free_ori, picks, _, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks=None,
                                  pick_ori=pick_ori)

    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    source_power = np.zeros((n_sources, len(data_csds)))
    n_csds = len(data_csds)

    logger.info('Computing DICS source power...')
    for i, (data_csd, noise_csd) in enumerate(zip(data_csds, noise_csds)):
        if n_csds > 1:
            logger.info('    computing DICS spatial filter %d out of %d' %
                        (i + 1, n_csds))

        Cm = data_csd.data

        # Calculating regularized inverse, equivalent to an inverse operation
        # after the following regularization:
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

    logger.info('[done]')

    subject = _subject_from_forward(forward)
    return SourceEstimate(source_power, vertices=vertno, tmin=fmin / 1000.,
                          tstep=fstep / 1000., subject=subject)
