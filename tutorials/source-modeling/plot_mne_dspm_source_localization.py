"""
.. _tut-inverse-methods:

Source localization with MNE/dSPM/sLORETA/eLORETA
=================================================

The aim of this tutorial is to teach you how to compute and apply a linear
minimum-norm inverse method on evoked/raw/epochs data.
"""

# sphinx_gallery_thumbnail_number = 10

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

###############################################################################
# Process MEG data

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname)  # already has an average reference
events = mne.find_events(raw, stim_channel='STI 014')

event_id = dict(aud_l=1)  # event trigger and conditions
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
raw.info['bads'] = ['MEG 2443', 'EEG 053']
baseline = (None, 0)  # means from the first instant to t = 0
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=('meg', 'eog'), baseline=baseline, reject=reject)

###############################################################################
# Compute regularized noise covariance
# ------------------------------------
#
# For more details see :ref:`tut_compute_covariance`.

noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

###############################################################################
# Compute the evoked response
# ---------------------------
# Let's just use the MEG channels for simplicity.

evoked = epochs.average().pick('meg')
evoked.plot(time_unit='s')
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='mag',
                    time_unit='s')

# Show whitening
evoked.plot_white(noise_cov, time_unit='s')

del epochs  # to save memory

###############################################################################
# Inverse modeling: MNE/dSPM on evoked and raw data
# -------------------------------------------------

# Read the forward solution and compute the inverse operator
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd)

# make an MEG inverse operator
info = evoked.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)
del fwd

# You can write it to disk with::
#
#     >>> from mne.minimum_norm import write_inverse_operator
#     >>> write_inverse_operator('sample_audvis-meg-oct-6-inv.fif',
#                                inverse_operator)

###############################################################################
# Compute inverse solution
# ------------------------

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

###############################################################################
# Visualization
# -------------
# View activation time-series

plt.figure()
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.show()

###############################################################################
# Examine the original data and the residual after fitting:

fig, axes = plt.subplots(2, 1)
evoked.plot(axes=axes)
for ax in axes:
    ax.texts = []
    for line in ax.lines:
        line.set_color('#98df81')
residual.plot(axes=axes)

###############################################################################
# Here we use peak getter to move visualization to the time point of the peak
# and draw a marker at the maximum peak vertex.

vertno_max, time_max = stc.get_peak(hemi='rh')

subjects_dir = data_path + '/subjects'
surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=5)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)

###############################################################################
# Morph data to average brain
# ---------------------------

# setup source morph
morph = mne.compute_source_morph(
    src=inverse_operator['src'], subject_from=stc.subject,
    subject_to='fsaverage', spacing=5,  # to ico-5
    subjects_dir=subjects_dir)
# morph data
stc_fsaverage = morph.apply(stc)

brain = stc_fsaverage.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=20)
del stc_fsaverage

###############################################################################
# Dipole orientations
# -------------------
# The ``pick_ori`` parameter of the
# :func:`mne.minimum_norm.apply_inverse` function controls
# the orientation of the dipoles. One useful setting is ``pick_ori='vector'``,
# which will return an estimate that does not only contain the source power at
# each dipole, but also the orientation of the dipoles.

stc_vec = apply_inverse(evoked, inverse_operator, lambda2,
                        method=method, pick_ori='vector')
brain = stc_vec.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Vector solution', 'title', font_size=20)
del stc_vec

###############################################################################
# Note that there is a relationship between the orientation of the dipoles and
# the surface of the cortex. For this reason, we do not use an inflated
# cortical surface for visualization, but the original surface used to define
# the source space.
#
# For more information about dipole orientations, see
# :ref:`tut-dipole-orientations`.

###############################################################################
# Now let's look at each solver:

for mi, (method, lims) in enumerate((('dSPM', [8, 12, 15]),
                                     ('sLORETA', [3, 5, 7]),
                                     ('eLORETA', [0.75, 1.25, 1.75]),)):
    surfer_kwargs['clim']['lims'] = lims
    stc = apply_inverse(evoked, inverse_operator, lambda2,
                        method=method, pick_ori=None)
    brain = stc.plot(figure=mi, **surfer_kwargs)
    brain.add_text(0.1, 0.9, method, 'title', font_size=20)
    del stc
