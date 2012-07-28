"""
=========================================================
Compute power spectrum densities of the sources with dSPM
=========================================================

Returns an STC file containing the PSD (in dB) of each of the sources.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne import fiff
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, compute_source_psd

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_label = data_path + '/MEG/sample/labels/Aud-lh.label'

# Setup for reading the raw data
raw = fiff.Raw(raw_fname, verbose=False)
events = mne.find_events(raw)
inverse_operator = read_inverse_operator(fname_inv)
raw.info['bads'] = ['MEG 2443', 'EEG 053']

# picks MEG gradiometers
picks = fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                            stim=False, include=[], exclude=raw.info['bads'])

tmin, tmax = 0, 120  # use the first 120s of data
fmin, fmax = 4, 100  # look at frequencies between 4 and 100Hz
NFFT = 2048  # the FFT size (NFFT). Ideally a power of 2
label = mne.read_label(fname_label)

stc = compute_source_psd(raw, inverse_operator, lambda2=1. / 9., method="dSPM",
                         tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                         pick_normal=True, NFFT=NFFT, label=label)

stc.save('psd_dSPM')

###############################################################################
# View PSD of sources in label
import pylab as pl
pl.plot(1e3 * stc.times, stc.data.T)
pl.xlabel('Frequency (Hz)')
pl.ylabel('PSD (dB)')
pl.title('Source Power Spectrum (PSD)')
pl.show()
