"""
============================================
Reading BEM surfaces from a forward solution
============================================

Plot BEM surfaces used for forward solution generation.
"""
# Author: Jaakko Leppakangas <jaeilepp@gmail.com>
#
# License: BSD (3-clause)
import os.path as op
from mayavi import mlab

import mne
from mne.datasets.sample import data_path

print(__doc__)

data_path = data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(fname)
subjects_dir = op.join(data_path, 'subjects')

###############################################################################
# Here we use :func:`mne.viz.plot_trans` with ``trans=None`` to plot only the
# surfaces without any transformations. For plotting transformation, see
# :ref:`tut_forward`.

mne.viz.plot_trans(raw.info, trans=None, subject='sample',
                   subjects_dir=subjects_dir, meg_sensors=[], eeg_sensors=[],
                   head='outer_skin', skull=['inner_skull', 'outer_skull'])
mlab.view(40, 60)

###############################################################################
# If you're using a sphere model, ``bem`` parameter can be used to pass it to
# the plotting function.

sphere = mne.make_sphere_model(info=raw.info, r0='auto',
                               head_radius='auto',
                               relative_radii=[0.8353, 0.847, 0.93, 1],
                               sigmas=(0.33, 1.0, 0.0042, 0.33))
mne.viz.plot_trans(raw.info, trans=None, source=None, bem=sphere,
                   head=True, skull=['inner_skull', 'outer_skull'])
