# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:03:39 2016

@author: pasca
"""

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne

from mne.datasets import sample
from mne.io import Raw
from mne import write_source_spaces, setup_source_space
from mne import setup_volume_source_space
from mne import make_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle

# Set dir
data_path = sample.data_path()
sbj_id = 'sample'
data_dir = op.join(data_path, 'MEG', sbj_id)
sbj_dir = op.join(data_path, 'subjects')
bem_dir = op.join(sbj_dir, sbj_id, 'bem')

# Set file names
fname_mixed_src = op.join(bem_dir, '%s-oct-6-mixed-src.fif' % sbj_id)
fname_aseg = op.join(sbj_dir, sbj_id, 'mri/aseg.mgz')

fname_model = op.join(bem_dir, '%s-5120-bem.fif' % sbj_id)
fname_bem = op.join(bem_dir, '%s-5120-bem-sol.fif' % sbj_id)

fname_raw = data_dir + '/sample_audvis_filt-0-40_raw.fif'
fname_trans = data_dir + '/sample_audvis_raw-trans.fif'
fname_fwd = data_dir + '/sample_audvis-meg-oct-6-mixed-fwd.fif'
fname_cov = data_dir + '/ernoise-cov.fif'
fname_event = data_dir + '/sample_audvis_filt-0-40_raw-eve.fif'

# List of sub structures we are interested in
'''
labels_vol = ['Left-Accumbens-area',
             'Left-Amygdala',
             'Left-Caudate',
             'Left-Hippocampus',
             'Left-Pallidum',
             'Left-Putamen',
             'Left-Thalamus-Proper',
             'Left-Cerebellum-Cortex',
             'Brain-Stem',
             'Right-Accumbens-area',
             'Right-Amygdala',
             'Right-Caudate',
             'Right-Hippocampus',
             'Right-Pallidum',
             'Right-Putamen',
             'Right-Thalamus-Proper',
             'Right-Cerebellum-Cortex']
'''
# Here we select only these sub structures we want to include in the source
# space
labels_vol = ['Left-Amygdala',
              'Left-Thalamus-Proper',
              'Left-Cerebellum-Cortex',
              'Right-Amygdala',
              'Right-Thalamus-Proper',
              'Right-Cerebellum-Cortex']

# Setup a surface-based source space
src = setup_source_space(sbj_id, subjects_dir=sbj_dir,
                         spacing='oct6', add_dist=False, overwrite=True)

# We create a mixed src space adding to the surface src space the volume
# regions specified in the list labels_vol. First, read the aseg file and the
# source space bounds using the inner skull surface

# Generate the mixed source space
for l in labels_vol:
    # setup a volume source space of the label l
    vol_src = setup_volume_source_space(sbj_id, mri=fname_aseg,
                                        pos=5.0,
                                        bem=fname_model,
                                        volume_label=l,
                                        subjects_dir=sbj_dir)
    # combine the source spaces
    src += vol_src

n = sum(src[i]['nuse'] for i in range(len(src)))
print('the src space contains %d spaces and %d points' % (len(src), n))

# Write the mixed source space
write_source_spaces(fname_mixed_src, src)

# Export source positions to nift file
nii_fname = op.join(bem_dir, '%s-mixed-src.nii' % sbj_id)
src.export_volume(nii_fname, mri_resolution=True)

# Uncomment the following lines to display source positions in freeview.
'''
# display image in freeview
from mne.utils import run_subprocess
mri_fname = op.join(sbj_dir, sbj_id, 'mri/brain.mgz')
run_subprocess(['freeview', '-v', mri_fname, '-v',
                '%s:colormap=lut:opacity=0.5' % aseg_fname, '-v',
                '%s:colormap=jet:colorscale=0,2' % nii_fname, '-slice',
                '157 75 105'])
'''

# Now we compute the lead field matrix for the mixed source space and the
# connectivity btw 74 regions in src space (68 in the surface src space and 6
# in the sub structures) based on dSPM inverse solutions

# Compute the fwd matrix
fwd = make_forward_solution(fname_raw, fname_trans, src, fname_bem,
                            fname_fwd,
                            mindist=5.0,  # ignore sources<=5mm from innerskull
                            meg=True, eeg=False,
                            n_jobs=2,
                            overwrite=True)

leadfield = fwd['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

src_fwd = fwd['src']
n = sum(src_fwd[i]['nuse'] for i in range(len(src_fwd)))
print('the fwd src space contains %d spaces and %d points' % (len(src_fwd), n))

# Load data
raw = Raw(fname_raw, preload=True)
noise_cov = mne.read_cov(fname_cov)
events = mne.read_events(fname_event)

# Add a bad channel
raw.info['bads'] += ['MEG 2443']

# Pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# Compute inverse solution and for each epoch
snr = 1.0           # use smaller SNR for raw data
inv_method = 'MNE'  # sLORETA, MNE, dSPM
parc = 'aparc'      # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'

lambda2 = 1.0 / snr ** 2

# Compute inverse operator
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov,
                                         loose=None, depth=None,
                                         fixed=False)

stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, inv_method,
                            pick_ori=None, return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels_parc = mne.read_labels_from_annot(sbj_id, parc=parc,
                                         subjects_dir=sbj_dir)

# Average the source estimates within each label of the cortical parcellation
# and each sub structure contained in the src space
# If mode = 'mean_flip' this option is used only for the surface cortical label
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(stcs, labels_parc, src,
                                         mode='mean_flip',
                                         allow_empty=True,
                                         return_generator=False)

# We compute the connectivity in the alpha band and plot it using a circular
# graph layout
fmin = 8.
fmax = 13.
sfreq = raw.info['sfreq']  # the sampling frequency
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    label_ts, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

# We create a list of Label containing also the sub structures
labels_aseg = mne.get_volume_labels_from_src(src, sbj_dir, sbj_id)
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
    ypos = np.mean(labels[idx].pos[:, 1])
    lh_labels.append('Brain-Stem')
    label_ypos_lh.append(ypos)
except ValueError:
    pass


# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels
             if label != 'Brain-Stem' and label[:-2] + 'rh' in rh_labels]

# Save the plot order
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])


# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
conmat = con[:, :, 0]
plot_connectivity_circle(conmat, label_names, n_lines=300,
                         node_angles=node_angles, node_colors=node_colors,
                         title='All-to-All Connectivity left-Auditory '
                               'Condition (PLI)')
plt.savefig('circle.png', facecolor='black')
