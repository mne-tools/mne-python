"""
======================================
Compute LCMV beamformer on evoked data
======================================

Compute LCMV beamformer solutions on an evoked dataset for three different
choices of source orientation and store the solutions in stc files for
visualisation.
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
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                       exclude='bads')

# Pick the channels of interest
raw.pick_channels([raw.ch_names[pick] for pick in picks])
# Re-normalize our empty-room projectors, so they are fine after subselection
raw.info.normalize_proj()

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), preload=True, proj=True,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()

forward = mne.read_forward_solution(fname_fwd)
forward = mne.convert_forward_solution(forward, surf_ori=True)

# Compute regularized noise and data covariances
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0, method='shrunk')
data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                  method='shrunk')

plt.close('all')

pick_oris = [None, 'normal', 'max-power']
names = ['free', 'normal', 'max-power']
descriptions = ['Free orientation, voxel: %i', 'Normal orientation, voxel: %i',
                'Max-power orientation, voxel: %i']
colors = ['b', 'k', 'r']

for pick_ori, name, desc, color in zip(pick_oris, names, descriptions, colors):
    # compute unit-noise-gain beamformer with whitening of the leadfield and
    # data (enabled by passing a noise covariance matrix)
    stc = lcmv(evoked, forward, noise_cov, data_cov, reg=0.05,
               pick_ori=pick_ori, weight_norm='unit-noise-gain',
               max_ori_out='signed')

    # View activation time-series in maximum voxel at 100 ms:
    time_idx = stc.time_as_index(0.1)
    max_vox = np.argmax(stc.data[:, time_idx])
    plt.plot(stc.times, stc.data[max_vox, :], color, hold=True,
             label=desc % max_vox)

plt.xlabel('Time (ms)')
plt.ylabel('LCMV value')
plt.ylim(-0.8, 2.2)
plt.title('LCMV in maximum voxel')
plt.legend()
plt.show()


# take absolute value for plotting
stc.data[:, :] = np.abs(stc.data)

# Plot last stc in the brain in 3D with PySurfer if available
brain = stc.plot(hemi='lh', subjects_dir=subjects_dir,
                 initial_time=0.1, time_unit='s')
brain.show_view('lateral')
