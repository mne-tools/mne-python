import pdb
import copy

import numpy as np
import pylab as pl

from scipy import signal

import mne
from mne.fiff.pick import pick_types_evoked, pick_types_forward, pick_channels_cov
from mne.forward import apply_forward
from mne.label import read_label
from mne.datasets import sample
from mne.minimum_norm.inverse import _make_stc
from mne.viz import plot_evoked, plot_sparse_source_estimates
from mne.time_frequency import ar_raw


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
    signal = np.zeros(len(timesamples))
    for m, s, a, f, p in zip(mus, sigmas, amps, freqs, phis):
        signal += gaboratomr(timesamples, s, m, f, p) * a
    return signal


def generate_fir_from_raw(raw, picks, order, tmin, tmax, proj=None):
    """Fits an AR model to raw data and creates FIR filter

    Parameters
    ----------
    raw : Raw object
        an instance of Raw
    picks : array of int
        indices of selected channels
    order : int
        order of the FIR filter
    tmin : float
        start time before event
    tmax : float
        end time after event
    projs : None | list
        The list of projection vectors

    Returns
    -------
    FIR : array
        filter coefficients
    """
    if proj is not None:
        raw.info['projs'] += proj
    picks = picks[:5]
    coefs = ar_raw(raw, order=order, picks=picks, tmin=tmin, tmax=tmax)
    mean_coefs = np.mean(coefs, axis=0)  # mean model accross channels
    FIR = np.r_[1, -mean_coefs]  # filter coefficient
    return FIR


def generate_noise(noise, noise_cov, nsamp, FIR=None):
    """Creates noise as a multivariate random process
    with specified cov matrix. No deepcopy of noise applied

    Parameters
    ----------
    noise : evoked object
        an instance of evoked
    noise_cov : cov object
        an instance of cov
    nsamp : int
        number of samples to generate
    FIR : None | array
        FIR filter coefficients

    Returns
    -------
    noise : evoked object
        an instance of evoked
    """
    noise_cov = pick_channels_cov(noise_cov, include=noise_template.info['ch_names'])
    rng = np.random.RandomState(0)
    noise.data = rng.multivariate_normal(np.zeros(noise.info['nchan']), noise_cov.data, nsamp).T
    if FIR is not None:
        noise.data = signal.lfilter([1], FIR, noise.data, axis=-1)
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
        SNRtemp = 20 * np.log10(np.sqrt(np.mean((evoked.data[:,tmask] ** 2).ravel()) / \
                                         np.mean((noise.data ** 2).ravel())))
        noise.data = 10 ** ((SNRtemp - float(SNR)) / 20) * noise.data
    else:
        SNRtemp = np.sqrt(np.mean((evoked.data[:,tmask] ** 2).ravel()) / \
                                         np.mean((noise.data ** 2).ravel()))
        noise.data = SNRtemp / SNR * noise.data
    evoked.data += noise.data
    return evoked


def select_source_idxs(fwd, label_fname):
    """Select source positions using a label

    Parameters
    ----------
    fwd : dict
        a forward solution
    label_fname : str
        filename of the freesurfer label to read

    Returns
    -------
    lh_vertno : list
        selected source coefficients on the left hemisphere
    rh_vertno : list
        selected source coefficients on the right hemisphere
    """
    lh_vertno = list()
    rh_vertno = list()

    label = read_label(label_fname)
    rng = np.random.RandomState(0)

    if label['hemi']=='lh':
        src_sel_lh = np.intersect1d(fwd['src'][0]['vertno'], label['vertices'])
        idx_select = rng.randint(0, len(src_sel_lh), 1)
        lh_vertno.append(src_sel_lh[idx_select][0])
    else:
        src_sel_rh = np.intersect1d(fwd['src'][1]['vertno'], label['vertices'])
        idx_select = rng.randint(0, len(src_sel_rh), 1)
        rh_vertno.append(src_sel_rh[idx_select][0])

    return lh_vertno, rh_vertno


## load data_sets from mne-sample-data ##
data_path = sample.data_path('.')

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
exclude = ['MEG 2443', 'EEG 053']
meg_include = True
eeg_include = True
fwd = pick_types_forward(fwd, meg=meg_include, eeg=eeg_include, exclude=exclude)

cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
noise_cov = mne.read_cov(cov_fname)

tmin = -0.1
#sfreq
tstep = 0.001
n_samples = 300
timesamples = np.linspace(tmin, tmin + n_samples * tstep, n_samples)

label = ['Aud-lh', 'Aud-rh']
amps = [[40 * 1e-9, 40 * 1e-9, 30 * 1e-9], [30 * 1e-9, 40 * 1e-9, 40 * 1e-9]]
mus = [[0.030, 0.060, 0.120], [0.040, 0.060, 0.140]]
sigmas = [[0.01, 0.02, 0.03], [0.01, 0.02, 0.03]]
freqs = [[0, 0, 0], [0, 0, 0]]
phis = [[0, 0, 0], [0, 0, 0]]

SNR = 6
dB = True

signals = list()
vertno = [[], []]
for k in range(len(label)):
    label_fname = data_path + '/MEG/sample/labels/%s.label' % label[k]
    lh_vertno, rh_vertno = select_source_idxs(fwd, label_fname)
    vertno[0] += lh_vertno
    vertno[1] += rh_vertno
    signals.append(source_signal(mus[k], sigmas[k], amps[k], freqs[k], phis[k], timesamples))
signals = np.vstack(signals)
stc = _make_stc(signals, tmin, tstep, vertno)
plot_sparse_source_estimates(fwd['src'], stc, bgcolor=(1, 1, 1),
                                opacity=0.5, high_resolution=True)

ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
evoked_template = mne.fiff.read_evoked(ave_fname, setno=0, baseline=None)
evoked_template = pick_types_evoked(evoked_template, meg=meg_include, eeg=eeg_include, exclude=exclude)
evoked = apply_forward(fwd, stc, evoked_template, start=None, stop=None)

noise_template = copy.deepcopy(evoked_template)
raw = mne.fiff.Raw(data_path + '/MEG/sample/sample_audvis_raw.fif')
proj = mne.read_proj(data_path + '/MEG/sample/ecg_proj.fif')
raw.info['projs'] += proj
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels
picks = mne.fiff.pick_types(raw.info, meg=True)
FIR = generate_fir_from_raw(raw, picks, 5, tmin=60, tmax=180, proj=proj)
noise = generate_noise(noise_template, noise_cov, n_samples, FIR=FIR)
pl.figure()
pl.psd(noise.data[0])
evoked = add_noise(evoked, noise, SNR, timesamples, tmin=0.0, tmax=0.2, dB=dB)
pl.figure()
plot_evoked(evoked)
