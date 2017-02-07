"""
==============================
Plotting the full MNE solution
==============================

The inverse operator's source space is shown in 3D, including the
directionality of the dipoles.
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

# Read the source space
fname_src = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
src = mne.read_source_spaces(fname_src)

# Apply inverse solution, set pick_ori='full' to obtain a FullSourceEstimate
# object
snr = 3.0
lambda2 = 1.0 / snr ** 2
s = apply_inverse(evoked, inv, lambda2, 'dSPM', pick_ori='full')

# Use peak getter to move vizualization to the time point of the peak magnitude
_, peak_time = s.get_peak(hemi='lh')

# Plot the source estimate
fig = s.plot(src, time=peak_time, hemi='lh', high_resolution=True)
