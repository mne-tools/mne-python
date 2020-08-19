"""
=============================
Re-referencing the EEG signal
=============================

This example shows how to load raw data and apply some EEG referencing schemes.
"""
# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from matplotlib import pyplot as plt

print(__doc__)

# Setup for reading the raw data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Read the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(event_fname)

# The EEG channels will be plotted to visualize the difference in referencing
# schemes.
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, exclude='bads')

###############################################################################
# We will now apply different EEG referencing schemes and plot the resulting
# evoked potentials. Note that when we construct epochs with ``mne.Epochs``, we
# supply the ``proj=True`` argument. This means that any available projectors
# are applied automatically. Specifically, if there is an average reference
# projector set by ``raw.set_eeg_reference('average', projection=True)``, MNE
# applies this projector when creating epochs.

reject = dict(eog=150e-6)
epochs_params = dict(events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                     picks=picks, reject=reject, proj=True)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    nrows=4, ncols=1, sharex=True, figsize=(6, 10))

# We first want to plot the data without any added reference (i.e., using only
# the reference that was applied during recording of the data).
# However, this particular data already has an average reference projection
# applied that we now need to remove again using :func:`mne.set_eeg_reference`
raw, _ = mne.set_eeg_reference(raw, [])  # use [] to remove average projection
evoked_no_ref = mne.Epochs(raw, **epochs_params).average()

evoked_no_ref.plot(axes=ax1, titles=dict(eeg='Original reference'), show=False,
                   time_unit='s')

# Now we want to plot the data with an average reference, so let's add the
# projection we removed earlier back to the data. Note that we can use
# "set_eeg_reference" as a method on the ``raw`` object as well.
raw.set_eeg_reference('average', projection=True)
evoked_car = mne.Epochs(raw, **epochs_params).average()

evoked_car.plot(axes=ax2, titles=dict(eeg='Average reference'), show=False,
                time_unit='s')

# Re-reference from an average reference to the mean of channels EEG 001 and
# EEG 002.
raw.set_eeg_reference(['EEG 001', 'EEG 002'])
evoked_custom = mne.Epochs(raw, **epochs_params).average()

evoked_custom.plot(axes=ax3, titles=dict(eeg='Custom reference'),
                   time_unit='s', show=False)

# Re-reference using REST :footcite:`Yao2001`. To do this, we need a forward
# solution, which we can quickly create:
sphere = mne.make_sphere_model('auto', 'auto', raw.info)
src = mne.setup_volume_source_space(sphere=sphere, exclude=30.,
                                    pos=15.)  # large "pos" just for speed!
forward = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
raw.set_eeg_reference('REST', forward=forward)
evoked_rest = mne.Epochs(raw, **epochs_params).average()

evoked_rest.plot(axes=ax4, titles=dict(eeg='REST (∞) reference'),
                 time_unit='s', show=True)

###############################################################################
# References
# ----------
# .. footbibliography::
