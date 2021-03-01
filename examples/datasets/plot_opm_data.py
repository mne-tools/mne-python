"""
.. _ex-opm-somatosensory:

Optically pumped magnetometer (OPM) data
========================================

In this dataset, electrical median nerve stimulation was delivered to the
left wrist of the subject. Somatosensory evoked fields were measured using
nine QuSpin SERF OPMs placed over the right-hand side somatomotor area.
Here we demonstrate how to localize these custom OPM data in MNE.
"""

# sphinx_gallery_thumbnail_number = 4

import os.path as op

import numpy as np
import mne

data_path = mne.datasets.opm.data_path()
subject = 'OPM_sample'
subjects_dir = op.join(data_path, 'subjects')
raw_fname = op.join(data_path, 'MEG', 'OPM', 'OPM_SEF_raw.fif')
bem_fname = op.join(subjects_dir, subject, 'bem',
                    subject + '-5120-5120-5120-bem-sol.fif')
fwd_fname = op.join(data_path, 'MEG', 'OPM', 'OPM_sample-fwd.fif')
coil_def_fname = op.join(data_path, 'MEG', 'OPM', 'coil_def.dat')

###############################################################################
# Prepare data for localization
# -----------------------------
# First we filter and epoch the data:

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(None, 90, h_trans_bandwidth=10.)
raw.notch_filter(50., notch_widths=1)


# Set epoch rejection threshold a bit larger than for SQUIDs
reject = dict(mag=2e-10)
tmin, tmax = -0.5, 1

# Find median nerve stimulator trigger
event_id = dict(Median=257)
events = mne.find_events(raw, stim_channel='STI101', mask=257, mask_type='and')
picks = mne.pick_types(raw.info, meg=True, eeg=False)
# We use verbose='error' to suppress warning about decimation causing aliasing,
# ideally we would low-pass and then decimate instead
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, verbose='error',
                    reject=reject, picks=picks, proj=False, decim=10,
                    preload=True)
evoked = epochs.average()
evoked.plot()
cov = mne.compute_covariance(epochs, tmax=0.)
del epochs, raw

###############################################################################
# Examine our coordinate alignment for source localization and compute a
# forward operator:
#
# .. note:: The Head<->MRI transform is an identity matrix, as the
#           co-registration method used equates the two coordinate
#           systems. This mis-defines the head coordinate system
#           (which should be based on the LPA, Nasion, and RPA)
#           but should be fine for these analyses.

bem = mne.read_bem_solution(bem_fname)
trans = None

# To compute the forward solution, we must
# provide our temporary/custom coil definitions, which can be done as::
#
# with mne.use_coil_def(coil_def_fname):
#     fwd = mne.make_forward_solution(
#         raw.info, trans, src, bem, eeg=False, mindist=5.0,
#         n_jobs=1, verbose=True)

fwd = mne.read_forward_solution(fwd_fname)
# use fixed orientation here just to save memory later
mne.convert_forward_solution(fwd, force_fixed=True, copy=False)

with mne.use_coil_def(coil_def_fname):
    fig = mne.viz.plot_alignment(evoked.info, trans=trans, subject=subject,
                                 subjects_dir=subjects_dir,
                                 surfaces=('head', 'pial'), bem=bem)

mne.viz.set_3d_view(figure=fig, azimuth=45, elevation=60, distance=0.4,
                    focalpoint=(0.02, 0, 0.04))

###############################################################################
# Perform dipole fitting
# ----------------------

# Fit dipoles on a subset of time points
with mne.use_coil_def(coil_def_fname):
    dip_opm, _ = mne.fit_dipole(evoked.copy().crop(0.015, 0.080),
                                cov, bem, trans, verbose=True)
idx = np.argmax(dip_opm.gof)
print('Best dipole at t=%0.1f ms with %0.1f%% GOF'
      % (1000 * dip_opm.times[idx], dip_opm.gof[idx]))

# Plot N20m dipole as an example
dip_opm.plot_locations(trans, subject, subjects_dir,
                       mode='orthoview', idx=idx)

###############################################################################
# Perform minimum-norm localization
# ---------------------------------
# Due to the small number of sensors, there will be some leakage of activity
# to areas with low/no sensitivity. Constraining the source space to
# areas we are sensitive to might be a good idea.

inverse_operator = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd, cov, loose=0., depth=None)
del fwd, cov

method = "MNE"
snr = 3.
lambda2 = 1. / snr ** 2
stc = mne.minimum_norm.apply_inverse(
    evoked, inverse_operator, lambda2, method=method,
    pick_ori=None, verbose=True)

# Plot source estimate at time of best dipole fit
brain = stc.plot(hemi='rh', views='lat', subjects_dir=subjects_dir,
                 initial_time=dip_opm.times[idx],
                 clim=dict(kind='percent', lims=[99, 99.9, 99.99]))
