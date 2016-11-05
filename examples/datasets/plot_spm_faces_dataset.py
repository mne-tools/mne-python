# doc:slow-example
"""
==========================================
From raw data to dSPM on SPM Faces dataset
==========================================

Runs a full pipeline using MNE-Python:

    - artifact removal
    - averaging Epochs
    - forward model computation
    - source reconstruction using dSPM on the contrast : "faces - scrambled"

.. note:: This example does quite a bit of processing, so even on a
          fast machine it can take several minutes to complete.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import matplotlib.pyplot as plt

import mne
from mne.datasets import spm_face
from mne.preprocessing import ICA, create_eog_epochs
from mne import io, combine_evoked
from mne.minimum_norm import make_inverse_operator, apply_inverse

print(__doc__)

data_path = spm_face.data_path()
subjects_dir = data_path + '/subjects'

###############################################################################
# Load and filter data, set up epochs

raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces%d_3D.ds'

raw = io.read_raw_ctf(raw_fname % 1, preload=True)  # Take first run
# Here to save memory and time we'll downsample heavily -- this is not
# advised for real data as it can effectively jitter events!
raw.resample(120., npad='auto')

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
raw.filter(1, 30, method='iir')

events = mne.find_events(raw, stim_channel='UPPT001')

# plot the events to get an idea of the paradigm
mne.viz.plot_events(events, raw.info['sfreq'])

event_ids = {"faces": 1, "scrambled": 2}

tmin, tmax = -0.2, 0.6
baseline = None  # no baseline as high-pass is applied
reject = dict(mag=5e-12)

epochs = mne.Epochs(raw, events, event_ids, tmin, tmax,  picks=picks,
                    baseline=baseline, preload=True, reject=reject)

# Fit ICA, find and remove major artifacts
ica = ICA(n_components=0.95, random_state=0).fit(raw, decim=1, reject=reject)

# compute correlation scores, get bad indices sorted by score
eog_epochs = create_eog_epochs(raw, ch_name='MRT31-2908', reject=reject)
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name='MRT31-2908')
ica.plot_scores(eog_scores, eog_inds)  # see scores the selection is based on
ica.plot_components(eog_inds)  # view topographic sensitivity of components
ica.exclude += eog_inds[:1]  # we saw the 2nd ECG component looked too dipolar
ica.plot_overlay(eog_epochs.average())  # inspect artifact removal
ica.apply(epochs)  # clean data, default in place

evoked = [epochs[k].average() for k in event_ids]

contrast = combine_evoked(evoked, weights=[-1, 1])  # Faces - scrambled

evoked.append(contrast)

for e in evoked:
    e.plot(ylim=dict(mag=[-400, 400]))

plt.show()

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs, tmax=0, method='shrunk')

###############################################################################
# Visualize fields on MEG helmet

trans_fname = data_path + ('/MEG/spm/SPM_CTF_MEG_example_faces1_3D_'
                           'raw-trans.fif')

maps = mne.make_field_map(evoked[0], trans_fname, subject='spm',
                          subjects_dir=subjects_dir, n_jobs=1)

evoked[0].plot_field(maps, time=0.170)


###############################################################################
# Compute forward model

# Make source space
src_fname = data_path + '/subjects/spm/bem/spm-oct-6-src.fif'
if not op.isfile(src_fname):
    src = mne.setup_source_space('spm', src_fname, spacing='oct6',
                                 subjects_dir=subjects_dir, overwrite=True)
else:
    src = mne.read_source_spaces(src_fname)

bem = data_path + '/subjects/spm/bem/spm-5120-5120-5120-bem-sol.fif'
forward = mne.make_forward_solution(contrast.info, trans_fname, src, bem)
forward = mne.convert_forward_solution(forward, surf_ori=True)

###############################################################################
# Compute inverse solution

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'

inverse_operator = make_inverse_operator(contrast.info, forward, noise_cov,
                                         loose=0.2, depth=0.8)

# Compute inverse solution on contrast
stc = apply_inverse(contrast, inverse_operator, lambda2, method, pick_ori=None)
# stc.save('spm_%s_dSPM_inverse' % constrast.comment)

# Plot contrast in 3D with PySurfer if available
brain = stc.plot(hemi='both', subjects_dir=subjects_dir, initial_time=0.170,
                 views=['ven'])
# brain.save_image('dSPM_map.png')
