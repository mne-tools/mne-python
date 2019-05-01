"""
======================================
Compute LCMV beamformer on evoked data
======================================

Compute LCMV beamformer on an evoked dataset for three different choices of
source orientation and store the solutions in stc files for visualization.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample
from mne.beamformer import make_lcmv, apply_lcmv

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
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0, method='shrunk',
                                   rank=None)
data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                  method='shrunk', rank=None)
evoked.plot(time_unit='s')

###############################################################################
# Run beamformers and look at maximum outputs

pick_oris = [None, 'normal', 'max-power', None]
descriptions = ['Free', 'Normal', 'Max-power', 'Fixed']

fig, ax = plt.subplots(1)
max_voxs = list()
colors = list()
for pick_ori, desc in zip(pick_oris, descriptions):
    # compute unit-noise-gain beamformer with whitening of the leadfield and
    # data (enabled by passing a noise covariance matrix)
    if desc == 'Fixed':
        use_forward = mne.convert_forward_solution(forward, force_fixed=True)
    else:
        use_forward = forward
    filters = make_lcmv(evoked.info, use_forward, data_cov, reg=0.05,
                        noise_cov=noise_cov, pick_ori=pick_ori,
                        weight_norm='unit-noise-gain', rank=None)
    print(filters)
    # apply this spatial filter to source-reconstruct the evoked data
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')

    # View activation time-series in maximum voxel at 100 ms:
    time_idx = stc.time_as_index(0.1)
    max_idx = np.argmax(np.abs(stc.data[:, time_idx]))
    # we know these are all left hemi, so we can just use vertices[0]
    max_voxs.append(stc.vertices[0][max_idx])
    h = ax.plot(stc.times, stc.data[max_idx, :],
                label='%s, voxel: %i' % (desc, max_idx))[0]
    colors.append(h.get_color())
    if pick_ori == 'max-power':
        max_stc = stc
ax.axhline(0, color='k')

ax.set(xlabel='Time (ms)', ylabel='LCMV value',
       title='LCMV in maximum voxel')
ax.legend(loc='lower right')
mne.viz.utils.plt_show()

###############################################################################
# We can also look at the spatial distribution

# Plot last stc in the brain in 3D with PySurfer if available
brain = max_stc.plot(hemi='lh', views='lat', subjects_dir=subjects_dir,
                     initial_time=0.1, time_unit='s', smoothing_steps=5)
for color, vertex in zip(colors, max_voxs):
    brain.add_foci([vertex], coords_as_verts=True, scale_factor=0.5,
                   hemi='lh', color=color)
