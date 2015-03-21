"""
================================================
Compute MNE-dSPM inverse solution on evoked data
================================================

Compute dSPM inverse solution on MNE evoked dataset
and stores the solution in stc files for visualisation.

"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
from mne.datasets import sample
from mne import read_evokeds
from mne.minimum_norm import apply_inverse, read_inverse_operator

print(__doc__)

data_path = sample.data_path()
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
subjects_dir = data_path + '/subjects'

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)

# Compute inverse solution
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_ori=None)

# Save result in stc files
stc.save('mne_%s_inverse' % method)

###############################################################################
# View activation time-series
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.show()

# Plot brain in 3D with PySurfer if available. Note that the subject name
# is already known by the SourceEstimate stc object.
brain = stc.plot(surface='inflated', hemi='rh', subjects_dir=subjects_dir,
                 transparent=True, clim='auto')
brain.show_view('lateral')

# use peak getter to move vizualization to the time point of the peak
vertno_max, time_idx = stc.get_peak(hemi='rh', time_as_index=True)

brain.set_data_time_index(time_idx)

# draw marker at maximum peaking vertex
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6)
brain.save_image('dSPM_map.png')
