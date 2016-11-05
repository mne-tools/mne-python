"""
======================================
Compute LCMV beamformer on evoked data
======================================

Compute LCMV beamformer solutions on evoked dataset for three different choices
of source orientation and stores the solutions in stc files for visualisation.

"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample
from mne.beamformer import lcmv

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name
subjects_dir = data_path + '/subjects'

###############################################################################
# Get epochs
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
left_temporal_channels = mne.read_selection('Left-temporal')
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                       exclude='bads', selection=left_temporal_channels)

# Pick the channels of interest
raw.pick_channels([raw.ch_names[pick] for pick in picks])
# Re-normalize our empty-room projectors, so they are fine after subselection
raw.info.normalize_proj()

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), preload=True, proj=True,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()

forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Compute regularized noise and data covariances
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0, method='shrunk')
data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                  method='shrunk')

plt.close('all')

pick_oris = [None, 'normal', 'max-power']
names = ['free', 'normal', 'max-power']
descriptions = ['Free orientation', 'Normal orientation', 'Max-power '
                'orientation']
colors = ['b', 'k', 'r']

for pick_ori, name, desc, color in zip(pick_oris, names, descriptions, colors):
    stc = lcmv(evoked, forward, noise_cov, data_cov, reg=0.01,
               pick_ori=pick_ori)

    # View activation time-series
    label = mne.read_label(fname_label)
    stc_label = stc.in_label(label)
    plt.plot(1e3 * stc_label.times, np.mean(stc_label.data, axis=0), color,
             hold=True, label=desc)

plt.xlabel('Time (ms)')
plt.ylabel('LCMV value')
plt.ylim(-0.8, 2.2)
plt.title('LCMV in %s' % label_name)
plt.legend()
plt.show()

# Plot last stc in the brain in 3D with PySurfer if available
brain = stc.plot(hemi='lh', subjects_dir=subjects_dir,
                 initial_time=0.1, time_unit='s')
brain.show_view('lateral')
