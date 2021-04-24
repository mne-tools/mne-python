"""
=========================================================
Compute power and phase lock in label of the source space
=========================================================

Compute time-frequency maps of power and phase lock in the source space.
The inverse method is linear based on dSPM inverse operator.

The example also shows the difference in the time-frequency maps
when they are computed with and without subtracting the evoked response
from each epoch. The former results in induced activity only while the
latter also includes evoked (stimulus-locked) activity.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, source_induced_power

print(__doc__)

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
label_name = 'Aud-rh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

tmin, tmax, event_id = -0.2, 0.5, 2

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')
inverse_operator = read_inverse_operator(fname_inv)

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# Picks MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

# Load epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject,
                    preload=True)

# Compute a source estimate per frequency band including and excluding the
# evoked response
freqs = np.arange(7, 30, 2)  # define frequencies of interest
label = mne.read_label(fname_label)
n_cycles = freqs / 3.  # different number of cycle per frequency

# subtract the evoked response in order to exclude evoked activity
epochs_induced = epochs.copy().subtract_evoked()

plt.close('all')

for ii, (this_epochs, title) in enumerate(zip([epochs, epochs_induced],
                                              ['evoked + induced',
                                               'induced only'])):
    # compute the source space power and the inter-trial coherence
    power, itc = source_induced_power(
        this_epochs, inverse_operator, freqs, label, baseline=(-0.1, 0),
        baseline_mode='percent', n_cycles=n_cycles, n_jobs=1)

    power = np.mean(power, axis=0)  # average over sources
    itc = np.mean(itc, axis=0)  # average over sources
    times = epochs.times

    ##########################################################################
    # View time-frequency plots
    plt.subplots_adjust(0.1, 0.08, 0.96, 0.94, 0.2, 0.43)
    plt.subplot(2, 2, 2 * ii + 1)
    plt.imshow(20 * power,
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=0., vmax=30., cmap='RdBu_r')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Power (%s)' % title)
    plt.colorbar()

    plt.subplot(2, 2, 2 * ii + 2)
    plt.imshow(itc,
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=0, vmax=0.7,
               cmap='RdBu_r')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('ITC (%s)' % title)
    plt.colorbar()

plt.show()
