"""
=======================================================================
Compute MNE-dSPM inverse solution on evoked data in volume source space
=======================================================================

Compute dSPM inverse solution on MNE evoked dataset in a volume source
space and stores the solution in a nifti file for visualisation.

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import pylab as pl
import mne
from mne.datasets import sample
from mne.fiff import Evoked
from mne.minimum_norm import apply_inverse, read_inverse_operator

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-vol-7-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
evoked = Evoked(fname_evoked, setno=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)
src = inverse_operator['src']

# Compute inverse solution
stc = apply_inverse(evoked, inverse_operator, lambda2, method)

max_idx = np.unravel_index(np.argmax(stc.data.ravel()), stc.data.shape)

# Export result as a 4D nifti object
img = stc.as_volume(src, mri_resolution=True, time_index=max_idx[1])

data = img.get_data()

# Awesome example activation map : take whatever is > .6 max
data_act = np.ma.MaskedArray(data, mask=(data < .1 * np.max(data)))

import pylab as pl
display_options = {}
display_options['interpolation'] = 'nearest'
display_options['cmap'] = pl.cm.gray

act_display_options = {}
act_display_options['interpolation'] = 'nearest'
act_display_options['cmap'] = pl.cm.hot


from pynax.view import ImshowView, PlotView
from pynax.core import Mark


mri_fname = subjects_dir + '/sample/mri/T1.mgz'
import nibabel as nib
t1_img = nib.load(mri_fname)

t1_data = t1_img.get_data()

fig = pl.figure(figsize=(6, 4), facecolor='k')

# Marks
mx = Mark(128, {'color': 'r'})
my = Mark(128, {'color': 'g'})
mz = Mark(128, {'color': 'b'})

ax_y = fig.add_axes([0.0, 0.2, 0.333, 0.8])
vy = ImshowView(ax_y, t1_data, [mx, 'v', 'h'], display_options)
vy.add_hmark(my)
vy.add_vmark(mz)
vy.add_layer(data_act, [mx, 'v', 'h'], display_options=act_display_options)
vy.draw()

ax_x = fig.add_axes([0.333, 0.2, 0.333, 0.8])
vx = ImshowView(ax_x, t1_data, ['h', 'v', my], display_options)
vx.add_hmark(mx)
vx.add_vmark(mz)
vx.add_layer(data_act, ['h', 'v', my], display_options=act_display_options)
vx.draw()

ax_z = fig.add_axes([0.666, 0.2, 0.333, 0.8])
vz = ImshowView(ax_z, t1_data, ['h', mz, 'v'], display_options)
vz.add_hmark(mx)
vz.add_vmark(my)
vz.add_layer(data_act, ['h', mz, 'v'], display_options=act_display_options)
vz.draw()


pl.show()


"""
# Save it as a nifti file
import nibabel as nib
nib.save(img, 'mne_%s_inverse.nii.gz' % method)

data = img.get_data()

# Plot result (one slice)
coronal_slice = data[:, 10, :, 60]
pl.close('all')
pl.imshow(np.ma.masked_less(coronal_slice, 8), cmap=pl.cm.Reds,
          interpolation='nearest')
pl.colorbar()
pl.contour(coronal_slice != 0, 1, colors=['black'])
pl.xticks([])
pl.yticks([])
pl.show()
"""
