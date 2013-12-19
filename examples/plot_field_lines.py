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
subjects_dir = data_path + '/subjects'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
setno = 'Left Auditory'

trans = mne.read_trans(trans_fname)
evoked = mne.fiff.read_evoked(evoked_fname, setno=setno,
                              baseline=(-0.2, 0.0))
helmet_surf = mne.get_meg_helmet_surf(evoked.info)
head_surf = mne.get_head_surface('sample', subjects_dir=subjects_dir)
head_surf = mne.transform_surface_to(head_surf, 'head', trans)

helmet_map = mne.make_surface_mapping(evoked.info, helmet_surf, 'meg',
                                      n_jobs=-1)
head_map = mne.make_surface_mapping(evoked.info, head_surf, 'eeg', n_jobs=-1)

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
colormap_lines = np.concatenate([np.tile([0., 0., 255., 255.], (127, 1)),
                                 np.tile([0., 0., 0., 255.], (2, 1)),
                                 np.tile([255., 0., 0., 255.], (127, 1))])
fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))
for ii, (surf, data) in enumerate(zip([head_surf, helmet_surf],
                                      [head_data, helmet_data])):
    x, y, z = surf['rr'].T
    # Make a solid surface
    vlim = np.max(np.abs(data))
    alpha = alphas[ii]
    mesh = mlab.pipeline.triangular_mesh_source(x, y, z, surf['tris'])
    hsurf = mlab.pipeline.surface(mesh, color=colors[ii], opacity=alpha)

    # Now show our field pattern
    mesh = mlab.pipeline.triangular_mesh_source(x, y, z, surf['tris'],
                                                scalars=data)
    fsurf = mlab.pipeline.surface(mesh, vmin=-vlim, vmax=vlim)
    fsurf.module_manager.scalar_lut_manager.lut.table = colormap

    # And the field lines on top
    mesh = mlab.pipeline.triangular_mesh_source(x, y, z, surf['tris'],
                                                scalars=data)
    cont = mlab.pipeline.contour_surface(mesh, contours=21, line_width=1.0,
                                         vmin=-vlim, vmax=vlim, opacity=alpha)
    cont.module_manager.scalar_lut_manager.lut.table = colormap_lines

text_str = '%s, t = %0.0f ms' % (setno, 1000 * evoked.times[0])
mlab.text(0.01, 0.01, text_str, width=0.4)
mlab.view(20, 80, roll=-60)
