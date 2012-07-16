# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import signal

import copy

from ..fiff.pick import pick_channels_cov
from ..minimum_norm.inverse import _make_stc
from ..utils import check_random_state


def gaboratomr(timesamples, sigma, mu, k, phase):
    """Computes a real-valued Gabor atom

    Parameters
    ----------
    timesamples : array
        samples in seconds
    sigma : float
        the variance of the gauss function.
    mu : float
        the mean of the gauss function.
    mu : float
        number of modulation of the cosine function.
    phase : float
        the phase of the modulated cosine function.

    Returns
    -------
    gnorm : array
        real_valued gabor atom with amplitude = 1
    """
    N = len(timesamples)
    g = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((timesamples - mu) / sigma) ** 2) *\
            np.cos(2 * np.pi * k / N * np.arange(0, N) + phase)
    gnorm = g / np.max(np.abs(g))
    return gnorm


def source_signal(mus, sigmas, amps, freqs, phis, timesamples):
    """Simulates source signal as sum of Gabor atoms

    Parameters
    ----------
    mu : list
        the means of the gauss functions.
    sigma : list
        the variances of the gauss functions.
    amps : list
        amplitudes of the Gabor atoms.
    freqs : list
        numbers of modulation of the cosine function.
    phase : list
        the phases of the modulated cosine function.
    timesamples : array
        samples in seconds

    Returns
    -------
    signal : array
        simulated source signal
    """
    data = np.zeros((len(mus), len(timesamples)))
    for k in range(len(mus)):
        for m, s, a, f, p in zip(mus[k], sigmas[k], amps[k], freqs[k], phis[k]):
            data[k] += gaboratomr(timesamples, s, m, f, p) * a
    return data


def generate_noise_evoked(evoked, noise_cov, n_samples, fir_filter=None, random_state=None):
    """Creates noise as a multivariate random process with specified cov matrix.

    Parameters
    ----------
    evoked : evoked object
        an instance of evoked used as template
    noise_cov : cov object
        an instance of cov
    n_samples : int
        number of time samples to generate
    fir_filter : None | array
        FIR filter coefficients
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    noise : evoked object
        an instance of evoked
    """
    noise = copy.deepcopy(evoked)
    noise_cov = pick_channels_cov(noise_cov, include=noise.info['ch_names'])
    rng = check_random_state(random_state)
    n_channels = np.zeros(noise.info['nchan'])
    noise.data = rng.multivariate_normal(n_channels, noise_cov.data, n_samples).T
    if fir_filter is not None:
        noise.data = signal.lfilter([1], fir_filter, noise.data, axis=-1)
    return noise


def add_noise(evoked, noise, SNR, timesamples, tmin=None, tmax=None, dB=False):
    """Adds noise to evoked object with specified SNR. SNR is computed in the
    interval from tmin to tmax. No deepcopy of evoked applied.

    Parameters
    ----------
    evoked : evoked object
        an instance of evoked with signal
    noise : evoked object
        an instance of evoked with noise
    SNR : float
        signal to noise ratio
    timesamples : array
        samples in seconds
    tmin : float
        start time before event
    tmax : float
        end time after event
    dB : bool
        SNR in dB or not

    Returns
    -------
    evoked : evoked object
        an instance of evoked
    """
    if tmin is None:
        tmin = np.min(timesamples)
    if tmax is None:
        tmax = np.max(timesamples)
    tmask = (timesamples >= tmin) & (timesamples <= tmax)
    if dB:
        SNRtemp = 20 * np.log10(np.sqrt(np.mean((evoked.data[:, tmask] ** 2).ravel()) / \
                                         np.mean((noise.data ** 2).ravel())))
        noise.data = 10 ** ((SNRtemp - float(SNR)) / 20) * noise.data
    else:
        SNRtemp = np.sqrt(np.mean((evoked.data[:, tmask] ** 2).ravel()) / \
                                         np.mean((noise.data ** 2).ravel()))
        noise.data = SNRtemp / SNR * noise.data
    evoked.data += noise.data
    return evoked


def select_source_in_label(fwd, label, random_state=None):
    """Select source positions using a label

    Parameters
    ----------
    fwd : dict
        a forward solution
    label : dict
        the label (read with mne.read_label)
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    lh_vertno : list
        selected source coefficients on the left hemisphere
    rh_vertno : list
        selected source coefficients on the right hemisphere
    """
    lh_vertno = list()
    rh_vertno = list()

    rng = check_random_state(random_state)

    if label['hemi'] == 'lh':
        src_sel_lh = np.intersect1d(fwd['src'][0]['vertno'], label['vertices'])
        idx_select = rng.randint(0, len(src_sel_lh), 1)
        lh_vertno.append(src_sel_lh[idx_select][0])
    else:
        src_sel_rh = np.intersect1d(fwd['src'][1]['vertno'], label['vertices'])
        idx_select = rng.randint(0, len(src_sel_rh), 1)
        rh_vertno.append(src_sel_rh[idx_select][0])

    return lh_vertno, rh_vertno


def generate_stc(fwd, labels, stc_data, tmin, tstep, random_state=0):
    rng = check_random_state(random_state)
    vertno = [[], []]
    for label in labels:
        lh_vertno, rh_vertno = select_source_in_label(fwd, label, rng)
        vertno[0] += lh_vertno
        vertno[1] += rh_vertno
    vertno = map(np.array, vertno)
    stc = _make_stc(stc_data, tmin, tstep, vertno)
    return stc
