# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:03:39 2016

@author: pasca
"""

import os.path as op
import numpy as np
import mne

from mne.datasets import sample
from mne.io import Raw
from mne import write_source_spaces, setup_source_space
from mne import setup_volume_source_space
from mne import make_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from nipype.utils.filemanip import split_filename as split_f


def get_aseg_labels(src, sbj_dir, sbj_id):
    import os.path as op
    import numpy as np

    from mne import Label
    from mne import get_volume_labels_from_aseg_AP

    # read the aseg file
    aseg_fname = op.join(sbj_dir, sbj_id, 'mri/aseg.mgz')
    all_labels_aseg = get_volume_labels_from_aseg_AP(aseg_fname)  # unnecessary

    # creo una lista di label per aseg
    labels_aseg = list()
    for nr in range(2, len(src)):
        vertices = src[nr]['vertno']

        pos = src[nr]['rr'][src[nr]['vertno'], :]
        roi_str = src[nr]['seg_name']
        try:
            ind = all_labels_aseg[0].index(roi_str)
            color = np.array(all_labels_aseg[1][ind])/255
        except ValueError:
            pass

        if 'left' in roi_str.lower():
            hemi = 'lh'
            roi_str = roi_str.replace('Left-', '') + '-lh'
        elif 'right' in roi_str.lower():
            hemi = 'rh'
            roi_str = roi_str.replace('Right-', '') + '-rh'
        else:
            hemi = 'both'

        label = Label(vertices=vertices, pos=pos, hemi=hemi,
                      name=roi_str, color=color,
                      subject=sbj_id)
        labels_aseg.append(label)

    print labels_aseg
    return labels_aseg

# set dir
data_path = '/home/pasca/Science/sw/mne/MNE-sample-data/'
# data_path = sample.data_path()
sbj_dir = op.join(data_path, 'subjects')
sbj_id = 'sample'
bem_dir = op.join(sbj_dir, sbj_id, 'bem')

# list of the sub structures we are interested in
'''
labels = ['Left-Accumbens-area',
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

labels = ['Left-Amygdala',
          'Left-Thalamus-Proper',
          'Left-Cerebellum-Cortex',
          'Right-Amygdala',
          'Right-Thalamus-Proper',
          'Right-Cerebellum-Cortex']

# setup a cortical surface source space
src = setup_source_space(sbj_id, subjects_dir=sbj_dir,
                         spacing='oct6', add_dist=False, overwrite=True)


# read the aseg file
aseg_fname = op.join(sbj_dir, sbj_id, 'mri/aseg.mgz')

model_fname = op.join(bem_dir, '%s-5120-bem.fif' % sbj_id)

# generate the mixed source space
for l in labels:
    print l
    # setup a volume source space of the label l
    vol_label = setup_volume_source_space(sbj_id, mri=aseg_fname,
                                          pos=5.0,
                                          bem=model_fname,
                                          volume_label=l,
                                          subjects_dir=sbj_dir)
    # Combine the source spaces
    src += vol_label

n = sum(src[i]['nuse'] for i in range(len(src)))
print('the src space contains %d spaces and %d points' % (len(src), n))

labels_cortex = mne.read_labels_from_annot(sbj_id, parc='aparc',
                                           subjects_dir=sbj_dir)

labels_aseg = get_aseg_labels(src, sbj_dir, sbj_id)
labels = labels_cortex + labels_aseg

# reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]
lh_labels = [name for name in label_names if name.endswith('lh')]

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

try:
    idx = label_names.index('Brain-Stem')
    ypos = np.mean(labels[idx].pos[:, 1])
    lh_labels.append('Brain-Stem')
    label_ypos.append(ypos)
except ValueError:
    pass

# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels if label != 'Brain-Stem']

# Save the plot order
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

# write the mixed source space
src_aseg_fname = op.join(bem_dir, '%s-oct6-aseg-src.fif' % sbj_id)
write_source_spaces(src_aseg_fname, src)

# Export source positions to nift file
nii_fname = op.join(bem_dir, '%s-aseg-src.nii' % sbj_id)

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

# load the co-registration file
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

# read bem solution
bem_fname = op.join(bem_dir, '%s-5120-bem-sol.fif' % sbj_id)

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
_, basename, _ = split_f(raw_fname)
fwd_filename = op.join(data_path, '%s-oct6-aseg-fwd.fif' % basename)

# computes the fwd matrix
fwd = make_forward_solution(raw_fname, trans_fname, src, bem_fname,
                            fwd_filename,
                            mindist=5.0,  # ignore sources<=5mm from innerskull
                            meg=True, eeg=False,
                            n_jobs=2,
                            overwrite=True)

leadfield = fwd['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

src_fwd = fwd['src']
n = sum(src_fwd[i]['nuse'] for i in range(len(src_fwd)))
print('the fwd src space contains %d spaces and %d points' % (len(src_fwd), n))

# compute inverse solution 
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
_, basename, _ = split_f(raw_fname)
cov_fname = data_path + '/MEG/sample/ernoise-cov.fif'

snr = 1.0           # use smaller SNR for raw data
inv_method = 'MNE'  # sLORETA, MNE, dSPM
parc = 'aparc'      # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'

lambda2 = 1.0 / snr ** 2

# Load raw data, noise cov and fwd operator
raw = Raw(raw_fname, preload=True)
noise_cov = mne.read_cov(cov_fname)
forward = mne.read_forward_solution(fwd_filename)

start, stop = raw.time_as_index([0, 15])  # read the first 15s of data

# compute inverse operator
inverse_operator = make_inverse_operator(raw.info, forward, noise_cov,
                                         loose=None, depth=None,
                                             fixed=False)


# apply inverse operator to the time windows [t_start, t_stop]s
stc = apply_inverse_raw(raw, inverse_operator, lambda2, inv_method,
                        label=None,
                        start=start, stop=stop,
                        buffer_size=1000,
                        pick_ori=None)

labels = mne.read_labels_from_annot(sbj_id, parc=parc, subjects_dir=sbj_dir)

src = inverse_operator['src']

label_ts = mne.extract_label_time_course_AP(stc, labels, src, mode='mean_flip',
                                            allow_empty=True,
                                            return_generator=False)

len(label_ts)

# TODO plot some label

