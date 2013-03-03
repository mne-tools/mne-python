"""
================================================
Compute MNE-dSPM inverse solution on evoked data
================================================

Compute dSPM inverse solution on MNE evoked dataset
and stores the solution in stc files for visualisation.

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl
from mne.datasets import sample
from mne.fiff import Evoked
from mne.minimum_norm import apply_inverse, read_inverse_operator


data_path = sample.data_path()
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
subjects_dir = data_path + '/subjects'

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
evoked = Evoked(fname_evoked, setno=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)

# Compute inverse solution
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_normal=False)

# Save result in stc files
stc.save('mne_%s_inverse' % method)

###############################################################################
# View activation time-series
pl.plot(1e3 * stc.times, stc.data[::100, :].T)
pl.xlabel('time (ms)')
pl.ylabel('%s value' % method)
pl.show()

# Plot brain in 3D with PySurfer if available. Note that the subject name
# is already known by the SourceEstimate stc object.
brain = stc.plot(surface='inflated', hemi='rh', subjects_dir=subjects_dir)
brain.set_data_time_index(180)
brain.scale_data_colormap(fmin=8, fmid=12, fmax=15, transparent=True)
brain.show_view('lateral')
brain.save_image('dSPM_map.png')
