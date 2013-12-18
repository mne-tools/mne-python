"""
======================
Plot M/EEG field lines
======================

In this example, M/EEG data are remapped onto the
MEG helmet (MEG) and subject's head surface (EEG).
This process can be comutationally intensive.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

print(__doc__)

import numpy as np
import mne

data_path = mne.datasets.sample.data_path()
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
setno = 'Left Auditory'

trans = mne.read_trans(trans_fname)
evoked = mne.fiff.read_evoked(evoked_fname, setno=setno,
                              baseline=(-0.2, 0.0))
helmet_surf = mne.get_meg_helmet_surf(evoked.info, trans)

# these need to be in head coordinates
head_surf = mne.get_head_surface('sample')

helmet_map = mne.make_surface_mapping(evoked.info, helmet_surf, trans, 'meg',
                                      n_jobs=0)
head_map = mne.make_surface_mapping(evoked.info, head_surf, trans, 'eeg',
                                    n_jobs=0)

# let's look at the N100
evoked.crop(0.09, 0.10)
evoked_eeg = mne.fiff.pick_types_evoked(evoked, meg=False, eeg=True)
evoked_meg = mne.fiff.pick_types_evoked(evoked, meg=True, eeg=False)
helmet_data = np.dot(helmet_map, evoked_meg.data[:, 0])
head_data = np.dot(head_map, evoked_eeg.data[:, 0])

# Plot them
from mayavi import mlab
alphas = [1.0, 0.5]
colors = [(0.6, 0.6, 0.6), (1.0, 1.0, 1.0)]
colormap = mne.viz.mne_analyze_colormap(format='mayavi')
fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))
for ii, (surf, data) in enumerate(zip([head_surf, helmet_surf],
                                      [head_data, helmet_data])):
    x = surf['rr'][:, 0]
    y = surf['rr'][:, 1]
    z = surf['rr'][:, 2]
    # Make a solid surface
    alpha = alphas[ii]
    mesh0 = mlab.pipeline.triangular_mesh_source(x, y, z, surf['tris'])
    surf0 = mlab.pipeline.surface(mesh0, color=colors[ii], opacity=alpha)

    # Now show our field lines on top
    mesh = mlab.pipeline.triangular_mesh_source(x, y, z, surf['tris'],
                                                scalars=data)
    surf = mlab.pipeline.surface(mesh)
    surf.module_manager.scalar_lut_manager.lut.table = colormap
    cont = mlab.pipeline.contour_surface(mesh, contours=50, line_width=1.0)
    cont.module_manager.scalar_lut_manager.lut.table = colormap

text_str = '%s, t = %0.0f ms' % (setno, 1000 * evoked.times[0])
mlab.text(0.01, 0.01, text_str, width=0.4)
mlab.view(10, 60)
