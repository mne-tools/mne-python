"""
.. _ex-spm-faces:

==========================================
From raw data to dSPM on SPM Faces dataset
==========================================

Runs a full pipeline using MNE-Python. This example does quite a bit of processing, so
even on a fast machine it can take several minutes to complete.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import mne
from mne import combine_evoked, io
from mne.datasets import spm_face
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.preprocessing import ICA, create_eog_epochs

print(__doc__)

data_path = spm_face.data_path()
subjects_dir = data_path / "subjects"
spm_path = data_path / "MEG" / "spm"

# %%
# Load data, filter it, and fit ICA.

raw_fname = spm_path / "SPM_CTF_MEG_example_faces1_3D.ds"
raw = io.read_raw_ctf(raw_fname, preload=True)  # Take first run
# Here to save memory and time we'll downsample heavily -- this is not
# advised for real data as it can effectively jitter events!
raw.resample(100)
raw.filter(1.0, None)  # high-pass
reject = dict(mag=5e-12)
ica = ICA(n_components=0.95, max_iter="auto", random_state=0)
ica.fit(raw, reject=reject)
# compute correlation scores, get bad indices sorted by score
eog_epochs = create_eog_epochs(raw, ch_name="MRT31-2908", reject=reject)
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name="MRT31-2908")
ica.plot_scores(eog_scores, eog_inds)  # see scores the selection is based on
ica.plot_components(eog_inds)  # view topographic sensitivity of components
ica.exclude += eog_inds[:1]  # we saw the 2nd ECG component looked too dipolar
ica.plot_overlay(eog_epochs.average())  # inspect artifact removal

# %%
# Epoch data and apply ICA.
events = mne.find_events(raw, stim_channel="UPPT001")
event_ids = {"faces": 1, "scrambled": 2}
tmin, tmax = -0.2, 0.6
baseline = None  # no baseline as high-pass is applied
epochs = mne.Epochs(
    raw,
    events,
    event_ids,
    tmin,
    tmax,
    picks="meg",
    baseline=None,
    preload=True,
    reject=reject,
)
del raw
ica.apply(epochs)  # clean data, default in place
evoked = [epochs[k].average() for k in event_ids]
contrast = combine_evoked(evoked, weights=[-1, 1])  # Faces - scrambled
evoked.append(contrast)
for e in evoked:
    e.plot(ylim=dict(mag=[-400, 400]))

# %%
# Estimate noise covariance and look at the whitened evoked data

noise_cov = mne.compute_covariance(epochs, tmax=0, method="shrunk", rank=None)
evoked[0].plot_white(noise_cov)

# %%
# Compute forward model

trans_fname = spm_path / "SPM_CTF_MEG_example_faces1_3D_raw-trans.fif"
src = subjects_dir / "spm" / "bem" / "spm-oct-6-src.fif"
bem = subjects_dir / "spm" / "bem" / "spm-5120-5120-5120-bem-sol.fif"
forward = mne.make_forward_solution(contrast.info, trans_fname, src, bem)

# %%
# Compute inverse solution and plot

# sphinx_gallery_thumbnail_number = 8

snr = 3.0
lambda2 = 1.0 / snr**2
inverse_operator = make_inverse_operator(contrast.info, forward, noise_cov)
stc = apply_inverse(contrast, inverse_operator, lambda2, method="dSPM", pick_ori=None)
brain = stc.plot(
    hemi="both",
    subjects_dir=subjects_dir,
    initial_time=0.170,
    views=["ven"],
    clim={"kind": "value", "lims": [3.0, 6.0, 9.0]},
)
