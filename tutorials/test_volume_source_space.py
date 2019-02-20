import numpy as np  # noqa
from mayavi import mlab  # noqa

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

data_path = sample.data_path()

# the raw file containing the channel location + types
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
# The paths to Freesurfer reconstructions
subjects_dir = data_path + '/subjects'
subject = 'sample'
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
fname_cov = data_path + '/MEG/sample/ernoise-cov.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

raw = mne.io.read_raw_fif(raw_fname)
info = raw.info

volume_labels = ['Left-Cerebral-Cortex', 'Right-Cerebral-Cortex']

src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir,
                                    mri='aseg.mgz')#, volume_label=volume_labels)

conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0)

# find inverse operator
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
evoked_total = epochs.average()
# Compute inverse solution and for each epoch
snr = 1.0           # use smaller SNR for raw data
inv_method = 'dSPM'
lambda2 = 1.0 / snr ** 2

# Compute inverse operator
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov,
                                         depth=None, fixed=False)
# Compute inverse solution
stc = apply_inverse(evoked_total, inverse_operator, lambda2, inv_method)
stc.crop(0.0, 0.2)

img = stc.as_volume(src,
                    mri_resolution=False)  # set True for full MRI resolution
