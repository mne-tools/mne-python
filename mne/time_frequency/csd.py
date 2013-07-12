# Author: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import warnings
import copy as cp

import numpy as np

import logging
logger = logging.getLogger('mne')

from ..fiff.pick import pick_types
from .. import verbose
from ..time_frequency.multitaper import _mt_spectra


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

        # Calculating Fourier transform using multitaper module
        window_fun = np.hanning(epoch.shape[1])
        x_mt, frequencies = _mt_spectra(epoch, window_fun,
                                        epochs.info['sfreq'])

        # Picking frequencies of interest
        freq_mask = (frequencies > fmin) & (frequencies < fmax)
        x_mt = x_mt[:, :, freq_mask]

        # Calculating CSD
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
