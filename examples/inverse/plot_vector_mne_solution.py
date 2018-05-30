"""
==============================
Plotting the full MNE solution
==============================

The source space that is used for the inverse computation defines a set of
dipoles, distributed across the cortex. When visualizing a source estimate, it
is sometimes useful to show the dipole directions in addiion to their estimated
magnitude.
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)
import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

# Read evoked data
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))

# Read inverse solution
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
inv = read_inverse_operator(fname_inv)

# Apply inverse solution, set pick_ori='vector' to obtain a
# :class:`mne.VectorSourceEstimate` object
snr = 3.0
lambda2 = 1.0 / snr ** 2
stc = apply_inverse(evoked, inv, lambda2, 'dSPM', pick_ori='vector')

# Use peak getter to move visualization to the time point of the peak magnitude
_, peak_time = stc.magnitude().get_peak(hemi='lh')

# Plot the source estimate
brain = stc.plot(initial_time=peak_time, hemi='lh', subjects_dir=subjects_dir)
