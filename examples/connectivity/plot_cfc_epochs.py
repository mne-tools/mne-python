"""
==============================================================
Compute cross-frequency coupling measures between signals
==============================================================

Computes the normalized amplitude traces for a cross frequency coupled
signal across a given range of frequencies and displays it along with
the event related average response.

References
----------

[1] Canolty RT, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
    Berger MS, Barbaro NM, Knight RT. "High gamma power is phase-locked to
    theta oscillations in human neocortex." Science. 2006.

[2] Tort ABL, Komorowski R, Eichenbaum H, Kopell N. Measuring phase-amplitude
    coupling between neuronal oscillations of different frequencies. Journal of
    Neurophysiology. 2010.

"""
# Author: Chris Holdgraf <choldgraf@berkeley.edu>
#         Praveen Sripad <praveen.sripad@rwth-aachen.de>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne import io
from mne.connectivity import phase_amplitude_coupling
from mne.viz import plot_phase_locked_amplitude, plot_phase_binned_amplitude
from mne.datasets import sample
import matplotlib.pyplot as plt

print(__doc__)

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

# Add a bad channel
raw.info['bads'] += ['MEG 2443']

# Pick MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                       exclude='bads')

# Define a pair of indices
ixs = [(4, 10)]

# First we can simply calculate a PAC statistic for these signals
f_range_phase = (8, 10)
f_range_amp = (40, 60)
pac = phase_amplitude_coupling(
    raw, f_range_phase, f_range_amp, ixs, ev=events[:, 0], tmin=-.1, tmax=.5)
pac = pac.mean()  # Average across events

# We can also visualize these relationships
# Create epochs for left-visual condition
event_id, tmin, tmax = 3, -1, 4
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True)
ph_range = np.linspace(8, 10, 6)
amp_range = np.linspace(40, 60, 20)

# Show the amplitude for a range of frequencies, phase-locked to a low-freq
ax = plot_phase_locked_amplitude(epochs, ph_range, amp_range, ixs[0][0],
                                 ixs[0][1], normalize=True)
ax[0].set_title('Phase Locked Amplitude, PAC = {0}'.format(pac))

# Show the avg amplitude of the high freqs for bins of phase in the low freq
ax = plot_phase_binned_amplitude(epochs, ph_range, amp_range,
                                 ixs[0][0], ixs[0][1], normalize=True,
                                 n_bins=20)
ax.set_title('Phase Binned Amplitude, PAC = {0}'.format(pac))
plt.show(block=True)
