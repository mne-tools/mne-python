# -*- coding: utf-8 -*-
"""
===============
Do a dipole fit
===============

This shows how to fit a dipole using mne-python.

For a comparison of fits between MNE-C and mne-python, see:

    https://gist.github.com/Eric89GXL/ca55f791200fe1dc3dd2

Note that for 3D graphics you may need to choose a specific IPython backend,
such as:
`%matplotlib qt` or `%matplotlib wx`
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from os import path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.forward import make_forward_dipole
from mne.simulation import simulate_evoked

print(__doc__)

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_ave = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_bem = op.join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
fname_surf_lh = op.join(subjects_dir, 'sample', 'surf', 'lh.white')

# Let's localize the N100m (using MEG only)
evoked = mne.read_evokeds(fname_ave, condition='Right Auditory',
                          baseline=(None, 0))
evoked.pick_types(meg=True, eeg=False)
evoked.crop(0.07, 0.08)

# Fit a dipole
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]

# Plot the result in 3D brain
dip.plot_locations(fname_trans, 'sample', subjects_dir)

# Calculate and visualise magnetic field predicted by dipole with maximum GOF
# and compare to the measured data, highlighting the ipsilateral source
stc, fwd = make_forward_dipole(dip, fname_bem, evoked.info, fname_trans)
pred_evoked = simulate_evoked(fwd, stc, evoked.info, None, snr=np.inf)

# FIXME the last dipole happens to have max gof, but plot_topomap fails!
# removing the last half sample time adjustment exposes the bug
# PR in the works...
bestfit_t = dip.times[np.argmax(dip.gof)] - 0.5/evoked.info['sfreq']
# rememeber to create a subplot for the colorbar
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=[10., 3.4])
vmin, vmax = -400, 400  # make sure each plot has same colour range
evoked.plot_topomap(times=bestfit_t, ch_type='mag', outlines='skirt',
                    time_format='Measured field', colorbar=False,
                    vmin=vmin, vmax=vmax, axes=axes[0])

pred_evoked.plot_topomap(times=bestfit_t, ch_type='mag', outlines='skirt',
                         time_format='Predicted field', colorbar=False,
                         vmin=vmin, vmax=vmax, axes=axes[1])

# FIXME why doesn't 'diff = evoked - pred_evoked' work?!? (just returns evoked)
# diff = evoked - pred_evoked
diff = mne.evoked.combine_evoked([evoked, pred_evoked], [1, -1])
diff.plot_topomap(times=bestfit_t, ch_type='mag',
                  outlines='skirt', time_format='Difference', colorbar=True,
                  vmin=vmin, vmax=vmax, axes=axes[2])
plt.suptitle('Comparison of measured and predicted fields '
             'at {:.0f} ms'.format(bestfit_t*1000.), fontsize=16)
