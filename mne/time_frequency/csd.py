# Author: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import warnings
import copy as cp

import numpy as np
from scipy.fftpack import fftfreq

import nitime.algorithms as tsa

import logging
logger = logging.getLogger('mne')

from ..fiff.pick import pick_types
from .. import verbose
from ..time_frequency.multitaper import _mt_spectra, _csd_from_mt


class CrossSpectralDensity(dict):
    """Cross-spectral density

    Attributes
    ----------
    data : array of shape (n_channels, n_channels)
        The cross-spectral density.
    ch_names : list of string
        List of channels' names.
    """
    @property
    def data(self):
        return self['data']

    @property
    def ch_names(self):
        return self['names']

    def __repr__(self):
        # TODO: This will have to be updated when the CSD object will be
        # expected to hold CSDs for different frequencies
        s = "size : %s x %s" % self.data.shape
        s += ", data : %s" % self.data
        return "<CrossSpectralDensity  |  %s>" % s


@verbose
def compute_csd(epochs, tmin=None, tmax=None, fmin=0, fmax=np.inf, projs=None,
                verbose=None):
    """Estimate cross-spectral density from epochs

    Note: Baseline correction should be used when creating the Epochs.
          Otherwise the computed cross-spectral density will be inaccurate.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs
    tmin : float | None
        Min time instant to consider. If None start at first sample.
    tmax : float | None
        Max time instant to consider. If None end at last sample.
    projs : list of Projection | None
        List of projectors to use in CSD calculation, or None to indicate that
        the projectors from the epochs should be inherited.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.
    """

    # check for baseline correction
    if epochs.baseline is None:
        warnings.warn('Epochs are not baseline corrected, cross-spectral '
                      'density may be inaccurate')

    if projs is None:
        projs = cp.deepcopy(epochs.info['projs'])
    else:
        projs = cp.deepcopy(projs)

    picks_meeg = pick_types(epochs[0].info, meg=True, eeg=True, eog=False,
                            exclude=[])
    ch_names = [epochs.ch_names[k] for k in picks_meeg]

    tstart, tend = None, None
    if tmin is not None:
        tstart = np.where(epochs.times >= tmin)[0][0]
    if tmax is not None:
        tend = np.where(epochs.times <= tmax)[0][-1] + 1
    tslice = slice(tstart, tend, None)

    csd_mean = np.zeros((len(ch_names), len(ch_names)), dtype=complex)

    # Compute CSD for each epoch
    n_epochs = 0
    for epoch in epochs:
        epoch = epoch[picks_meeg][:, tslice]

        # Implementation using nitime's Welch
        #csd_method = {}
        #csd_method['Fs'] = epochs.info['sfreq']
        #csd_method['this_method'] = 'welch'  # see comment below
        #csd_method['NFFT'] = tend - tstart
        #frequencies_nt, csds_epoch_nt = tsa.get_spectra(epoch,
        #                                                method=csd_method)

        # Implementation using nitime's periodogram_csd
        #csd_method = {}
        #csd_method['Fs'] = epochs.info['sfreq']
        #csd_method['normalize'] = False
        #csd_method['this_method'] = 'periodogram_csd'
        #csd_method['sides'] = 'onesided'
        #frequencies_nt, csds_epoch_nt = tsa.get_spectra(epoch,
        #                                                method=csd_method)
        #frequencies_nt /= 100 # Weird frequencies

        # This is code copied directly from nitime for further comparison and
        # it gives identical results to below code using the multitaper module
        # when it is used without a window function.
        #Fs = epochs.info['sfreq']
        #s = epoch
        #N = s.shape[-1]
        #from scipy import fftpack
        #Sk_loc = fftpack.fft(s, n=N)
        #M = Sk_loc.shape[0]
        # putative Nyquist freq
        #Fn = N / 2 + 1
        # last duplicate freq
        #Fl = (N + 1) / 2
        #csds_epoch_nt_cp = np.empty((M, M, Fn), 'D')
        #freqs = np.linspace(0, Fs / 2, Fn)
        #for i in xrange(M):
        #    for j in xrange(i + 1):
        #        csds_epoch_nt_cp[i, j, 0] = Sk_loc[i, 0] * Sk_loc[j, 0].conj()
        #        csds_epoch_nt_cp[i, j, 1:Fl] = 2 * (Sk_loc[i, 1:Fl] *
        #                                       Sk_loc[j, 1:Fl].conj())
        #        if Fn > Fl:
        #            csds_epoch_nt_cp[i, j, Fn - 1] = (Sk_loc[i, Fn - 1] *
        #                                              Sk_loc[j, Fn -
        #                                              1].conj())

        # Picking frequencies of interest in the nitime results
        #freq_mask = (frequencies_nt > fmin) & (frequencies_nt < fmax)
        #csd_epoch_nt = np.mean(csds_epoch_nt[:, :, freq_mask], 2)
        #csd_epoch_nt_cp = np.mean(csds_epoch_nt_cp[:, :, freq_mask], 2)

        # Making the nitime CSD matrices symmetrical
        #csd_epoch_nt = csd_epoch_nt + csd_epoch_nt.conj().T -\
        #               np.diag(csd_epoch_nt.diagonal())
        #csd_epoch_nt_cp = csd_epoch_nt_cp + csd_epoch_nt_cp.conj().T -\
        #                  np.diag(csd_epoch_nt_cp.diagonal())

        # There is a problem with nitime's resulsts.
        # The documentation states that the results should be a semi-filled
        # matrix (i.e. just the upper triangle should be filled, not the
        # lower), but what is returned is a fully filled matrix, for which,
        # csd_epoch_nt[i,j] is different than csd_epoch_nt[j,i].conj()
        # That's the case for the periodogram_csd method, for Welch's method, a
        # semi-filled matrix is returned.

        # Calculating Fourier transform using multitaper module
        window_fun = np.hanning(epoch.shape[1])
        x_mt, frequencies = _mt_spectra(epoch, window_fun,
                                        epochs.info['sfreq'])

        # Picking frequencies of interest
        freq_mask = (frequencies > fmin) & (frequencies < fmax)
        x_mt = x_mt[:, :, freq_mask]

        # Calculating CSD using the multitaper module
        #x_mt_2 = np.tile(x_mt, [1, x_mt.shape[0], 1])
        #x_mt_2 = x_mt_2[:, :, np.newaxis, :] # _csd_from_mt sums over axis=-2
        #y_mt_2 = np.transpose(x_mt_2, axes=[1, 0, 2, 3])
        #weights = np.array([1.])[:, None, None]
        #csds_epoch_mt = _csd_from_mt(x_mt_2, y_mt_2, weights, weights)

        # Calculating CSD without using the multitaper module
        x_mt = np.tile(x_mt, [1, x_mt.shape[0], 1])
        y_mt = np.transpose(x_mt, axes=[1, 0, 2])
        csds_epoch = 2 * x_mt * y_mt.conj()

        # Averaging over frequencies of interest
        csd_epoch = np.mean(csds_epoch, 2)

        csd_mean += csd_epoch
        n_epochs += 1

    csd_mean /= n_epochs

    csd = CrossSpectralDensity()
    csd.update(dim=len(csd_mean), names=cp.deepcopy(ch_names), data=csd_mean,
               projs=cp.deepcopy(projs), bads=cp.deepcopy(epochs.info['bads']))

    return csd
