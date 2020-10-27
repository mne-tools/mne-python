"""
===============================================================================
Compute mixed source space connectivity and visualize it using a circular graph
===============================================================================

This example computes the all-to-all connectivity between 75 regions in a
mixed source space based on dSPM inverse solutions and a FreeSurfer cortical
parcellation. The connectivity is visualized using a circular graph which
is ordered based on the locations of the regions in the axial plane.
"""
# Author: Annalisa Pascarella <a.pascarella@iac.cnr.it>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt

from mne.datasets import sample
from mne import setup_volume_source_space, setup_source_space
from mne import make_forward_solution
from mne.io import read_raw_fif
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle

# Set directories
data_path = sample.data_path()
subject = 'sample'
data_dir = op.join(data_path, 'MEG', subject)
subjects_dir = op.join(data_path, 'subjects')
bem_dir = op.join(subjects_dir, subject, 'bem')

# Set file names
fname_aseg = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')

fname_model = op.join(bem_dir, '%s-5120-bem.fif' % subject)
fname_bem = op.join(bem_dir, '%s-5120-bem-sol.fif' % subject)

fname_raw = data_dir + '/sample_audvis_filt-0-40_raw.fif'
fname_trans = data_dir + '/sample_audvis_raw-trans.fif'
fname_cov = data_dir + '/ernoise-cov.fif'
fname_event = data_dir + '/sample_audvis_filt-0-40_raw-eve.fif'

# List of sub structures we are interested in. We select only the
# sub structures we want to include in the source space
labels_vol = ['Left-Amygdala',
              'Left-Thalamus-Proper',
              'Left-Cerebellum-Cortex',
              'Brain-Stem',
              'Right-Amygdala',
              'Right-Thalamus-Proper',
              'Right-Cerebellum-Cortex']

# Setup a surface-based source space, oct5 is not very dense (just used
# to speed up this example; we recommend oct6 in actual analyses)
src = setup_source_space(subject, subjects_dir=subjects_dir,
                         spacing='oct5', add_dist=False)

# Setup a volume source space
# set pos=10.0 for speed, not very accurate; we recommend something smaller
# like 5.0 in actual analyses:
vol_src = setup_volume_source_space(
    subject, mri=fname_aseg, pos=10.0, bem=fname_model,
    add_interpolator=False,  # just for speed, usually use True
    volume_label=labels_vol, subjects_dir=subjects_dir)
# Generate the mixed source space
src += vol_src

# Load data
raw = read_raw_fif(fname_raw)
raw.pick_types(meg=True, eeg=False, eog=True, stim=True).load_data()
events = mne.find_events(raw)
noise_cov = mne.read_cov(fname_cov)

# compute the fwd matrix
fwd = make_forward_solution(raw.info, fname_trans, src, fname_bem,
                            mindist=5.0)  # ignore sources<=5mm from innerskull
del src

# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
reject = dict(mag=4e-12, grad=4000e-13, eog=150e-6)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    reject=reject, preload=False)
del raw

# Compute inverse solution and for each epoch
snr = 1.0           # use smaller SNR for raw data
inv_method = 'dSPM'
parc = 'aparc'      # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'

lambda2 = 1.0 / snr ** 2

# Compute inverse operator
inverse_operator = make_inverse_operator(
    epochs.info, fwd, noise_cov, depth=None, fixed=False)
del fwd

stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, inv_method,
                            pick_ori=None, return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels_parc = mne.read_labels_from_annot(subject, parc=parc,
                                         subjects_dir=subjects_dir)

# Average the source estimates within each label of the cortical parcellation
# and each sub structures contained in the src space
# If mode = 'mean_flip' this option is used only for the cortical label
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(
    stcs, labels_parc, src, mode='mean_flip', allow_empty=True,
    return_generator=True)

# We compute the connectivity in the alpha band and plot it using a circular
# graph layout
fmin = 8.
fmax = 13.
sfreq = epochs.info['sfreq']  # the sampling frequency
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    label_ts, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

# We create a list of Label containing also the sub structures
labels_aseg = mne.get_volume_labels_from_src(src, subject, subjects_dir)
labels = labels_parc + labels_aseg

# read colors
node_colors = [label.color for label in labels]

# We reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]
lh_labels = [name for name in label_names if name.endswith('lh')]
rh_labels = [name for name in label_names if name.endswith('rh')]

# Get the y-location of the label
label_ypos_lh = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos_lh.append(ypos)
try:
    idx = label_names.index('Brain-Stem')
except ValueError:
    pass
else:
    ypos = np.mean(labels[idx].pos[:, 1])
    lh_labels.append('Brain-Stem')
    label_ypos_lh.append(ypos)


# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels
             if label != 'Brain-Stem' and label[:-2] + 'rh' in rh_labels]

# Save the plot order
node_order = lh_labels[::-1] + rh_labels

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) // 2])


# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
conmat = con[:, :, 0]
fig = plt.figure(num=None, figsize=(8, 8), facecolor='black')
plot_connectivity_circle(conmat, label_names, n_lines=300,
                         node_angles=node_angles, node_colors=node_colors,
                         title='All-to-All Connectivity left-Auditory '
                         'Condition (PLI)', fig=fig)

###############################################################################
# Save the figure (optional)
# --------------------------
#
# By default matplotlib does not save using the facecolor, even though this was
# set when the figure was generated. If not set via savefig, the labels, title,
# and legend will be cut off from the output png file::
#
#     >>> fname_fig = data_path + '/MEG/sample/plot_mixed_connect.png'
#     >>> plt.savefig(fname_fig, facecolor='black')
