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
from ..forward import apply_forward


def generate_evoked(fwd, stc, evoked, cov, snr=3, tmin=None, tmax=None,
                    fir_filter=None, random_state=None):
    """Generate noisy evoked data

    Parameters
    ----------
    fwd : dict
        a forward solution
    stc : SourceEstimate object
        The source time courses
    evoked : Evoked object
        An instance of evoked used as template
    cov : Covariance object
        The noise covariance
    snr : float
        signal to noise ratio in dB. It corresponds to
        10 * log10( var(signal) / var(noise) )
    tmin : float | None
        start of time interval to estimate SNR. If None first time point
        is used.
    tmax : float
        start of time interval to estimate SNR. If None last time point
        is used.
    fir_filter : None | array
        FIR filter coefficients e.g. [1, -1, 0.2]
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    evoked : Evoked object
        The simulated evoked data
    """
    evoked = apply_forward(fwd, stc, evoked)
    noise = generate_noise_evoked(evoked, cov, fir_filter)
    evoked_noise = add_noise_evoked(evoked, noise, snr, tmin=tmin, tmax=tmax)
    return evoked_noise


def generate_noise_evoked(evoked, noise_cov, fir_filter=None,
                          random_state=None):
    """Creates noise as a multivariate Gaussian

    The spatial covariance of the noise is given from the cov matrix.

    Parameters
    ----------
    evoked : evoked object
        an instance of evoked used as template
    cov : Covariance object
        The noise covariance
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
    n_samples = evoked.data.shape[1]
    noise.data = rng.multivariate_normal(n_channels, noise_cov.data,
                                         n_samples).T
    if fir_filter is not None:
        noise.data = signal.lfilter([1], fir_filter, noise.data, axis=-1)
    return noise


def add_noise_evoked(evoked, noise, snr, tmin=None, tmax=None):
    """Adds noise to evoked object with specified SNR.

    SNR is computed in the interval from tmin to tmax.

    Parameters
    ----------
    evoked : Evoked object
        An instance of evoked with signal
    noise : Evoked object
        An instance of evoked with noise
    snr : float
        signal to noise ratio in dB. It corresponds to
        10 * log10( var(signal) / var(noise) )
    tmin : float
        start time before event
    tmax : float
        end time after event

    Returns
    -------
    evoked_noise : Evoked object
        An instance of evoked corrupted by noise
    """
    evoked = copy.deepcopy(evoked)
    times = evoked.times
    if tmin is None:
        tmin = np.min(times)
    if tmax is None:
        tmax = np.max(times)
    tmask = (times >= tmin) & (times <= tmax)
    tmp = np.mean((evoked.data[:, tmask] ** 2).ravel()) / \
                                     np.mean((noise.data ** 2).ravel())
    tmp = 10 * np.log10(tmp)
    noise.data = 10 ** ((tmp - float(snr)) / 20) * noise.data
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


def generate_sparse_stc(fwd, labels, stc_data, tmin, tstep, random_state=0):
    """Generate sources time courses from waveforms and labels

    Parameters
    ----------
    fwd : dict
        a forward solution
    labels : list of dict
        The labels
    stc_data : array
        The waveforms
    tmin : float
        The beginning of the timeseries
    tstep : float
        The time step (1 / sampling frequency)
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.
    """
    rng = check_random_state(random_state)
    vertno = [[], []]
    for label in labels:
        lh_vertno, rh_vertno = select_source_in_label(fwd, label, rng)
        vertno[0] += lh_vertno
        vertno[1] += rh_vertno
    vertno = map(np.array, vertno)
    stc = _make_stc(stc_data, tmin, tstep, vertno)
    return stc
