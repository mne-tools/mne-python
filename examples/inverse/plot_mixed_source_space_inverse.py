"""
=======================================================================
Compute MNE inverse solution on evoked data in a mixed source space
=======================================================================

Create a mixed source space and compute MNE inverse solution on evoked dataset

"""
# Author: Annalisa Pascarella <a.pascarella@iac.cnr.it>
#
# License: BSD (3-clause)

import os.path as op
import matplotlib.pyplot as plt
import mne

from mne.datasets import sample
from mne import write_source_spaces, setup_source_space
from mne import setup_volume_source_space
from mne import make_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse

from nilearn import plotting

# Set dir
data_path = sample.data_path()
subject = 'sample'
data_dir = op.join(data_path, 'MEG', subject)
subjects_dir = op.join(data_path, 'subjects')
bem_dir = op.join(subjects_dir, subject, 'bem')

# Set file names
fname_mixed_src = op.join(bem_dir, '%s-oct-6-mixed-src.fif' % subject)
fname_aseg = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')

fname_model = op.join(bem_dir, '%s-5120-bem.fif' % subject)
fname_bem = op.join(bem_dir, '%s-5120-bem-sol.fif' % subject)

fname_evoked = data_dir + '/sample_audvis-ave.fif'
fname_trans = data_dir + '/sample_audvis_raw-trans.fif'
fname_fwd = data_dir + '/sample_audvis-meg-oct-6-mixed-fwd.fif'
fname_cov = data_dir + '/sample_audvis-shrunk-cov.fif'

# List of sub structures we are interested in. We select only the
# sub structures we want to include in the source space
labels_vol = ['Left-Amygdala',
              'Left-Thalamus-Proper',
              'Left-Cerebellum-Cortex',
              'Brain-Stem',
              'Right-Amygdala',
              'Right-Thalamus-Proper',
              'Right-Cerebellum-Cortex']

# Setup a surface-based source space
src = setup_source_space(subject, subjects_dir=subjects_dir,
                         spacing='oct6', add_dist=False)

# We create a mixed src space adding to the surface src space the volume
# regions specified in the list labels_vol. First, read the aseg file and the
# source space bounds using the inner skull surface

# Setup a volume source space
# set pos=7.0 for speed issue
vol_src = setup_volume_source_space(subject, mri=fname_aseg,
                                    pos=7.0,
                                    bem=fname_model,
                                    volume_label=labels_vol,
                                    subjects_dir=subjects_dir)
# Generate the mixed source space
src += vol_src

n = sum(src[i]['nuse'] for i in range(len(src)))
print('the src space contains %d spaces and %d points' % (len(src), n))

# Write the mixed source space
write_source_spaces(fname_mixed_src, src)

# Export source positions to nift file
nii_fname = op.join(bem_dir, '%s-mixed-src.nii' % subject)
src.export_volume(nii_fname, mri_resolution=True)

plotting.plot_img(nii_fname, cmap=plt.cm.spectral)
plt.show()

# Compute the fwd matrix
fwd = make_forward_solution(fname_evoked, fname_trans, src, fname_bem,
                            mindist=5.0,  # ignore sources<=5mm from innerskull
                            meg=True, eeg=False,
                            n_jobs=1,
                            overwrite=True)

leadfield = fwd['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

src_fwd = fwd['src']
n = sum(src_fwd[i]['nuse'] for i in range(len(src_fwd)))
print('the fwd src space contains %d spaces and %d points' % (len(src_fwd), n))

# Load data
condition = 'Left Auditory'
evoked = mne.read_evokeds(fname_evoked, condition=condition,
                          baseline=(None, 0))
noise_cov = mne.read_cov(fname_cov)

# Compute inverse solution and for each epoch
snr = 3.0           # use smaller SNR for raw data
inv_method = 'MNE'  # sLORETA, MNE, dSPM
parc = 'aparc'      # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'

lambda2 = 1.0 / snr ** 2

# Compute inverse operator
inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov,
                                         loose=None, depth=None,
                                         fixed=False)

stcs = apply_inverse(evoked, inverse_operator, lambda2, inv_method,
                     pick_ori=None)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels_parc = mne.read_labels_from_annot(subject, parc=parc,
                                         subjects_dir=subjects_dir)

# Average the source estimates within each label of the cortical parcellation
# and each sub structure contained in the src space
# If mode = 'mean_flip' this option is used only for the surface cortical label
src = inverse_operator['src']

label_ts = mne.extract_label_time_course([stcs], labels_parc, src,
                                         mode='mean',
                                         allow_empty=True,
                                         return_generator=False)

# TODO some plot
