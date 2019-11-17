
"""
=====================
Plot Dipole Simulator
=====================

"""
# Author: Alex Rockhill <aprockhill206@gmail.com>
#
# License: BSD (3-clause)


import os
import os.path as op
import numpy as np

from mne.viz.utils import mne_analyze_colormap
from mne.datasets import sample
from mne import read_evokeds, read_forward_solution, pick_types

from traits.api import HasTraits, Range, Enum, Instance, \
        on_trait_change
from traitsui.api import View, Item, Group

from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
                MlabSceneModel

# from mne.viz import plot_evoked_field
from mne.forward import make_field_map

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked data
condition = 'Left Auditory'

fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-eeg-oct-6-fwd.fif')
# eeg_fwd = read_forward_solution(fname_fwd)


class DipoleModel(HasTraits):
    r = Range(0., 20., 10., style='simple')
    phi = Range(0., 360., 1., style='simple')
    theta = Range(0., 180., 1., style='simple')
    x = Range(-100., 100., 0, style='simple')
    y = Range(-100, 100., 0., style='simple')
    z = Range(-100, 100., 0., style='simple')
    modality = Enum('grad', 'mag', 'eeg')

    current_modality = None

    scale = 1e-2

    evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))

    surf_maps = make_field_map(evoked, subject='sample',
                               subjects_dir=op.join(data_path, 'subjects'),
                               trans=fname_trans)

    scene = Instance(MlabSceneModel, ())

    plot = Instance(PipelineBase)

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                Group('_', 'r', 'phi', 'theta', 'x', 'y', 'z', 'modality'),
                resizable=True,
                )

    @on_trait_change('r,phi,theta,x,y,z,modality,scene.activated')
    def update_plot(self):
        if self.plot is None:
            self.current_modality = self.modality
            self.set_modality()
            self.plot = self.scene.mlab.quiver3d(self.x, self.y, self.z,
                                                 *self.get_arrow(),
                                                 line_width=10,
                                                 scale_factor=self.scale)
        else:
            u, v, w = self.get_arrow()
            self.plot.mlab_source.trait_set(x=self.x * self.scale,
                                            y=self.y * self.scale,
                                            z=self.z * self.scale,
                                            u=u, v=v, w=w)
            if self.current_modality != self.modality:
                print('change modality')

    def get_arrow(self):
        u = self.r * np.cos(self.theta) * np.sin(self.phi)
        v = self.r * np.sin(self.theta) * np.sin(self.phi)
        w = self.r * np.cos(self.phi)
        return u, v, w

    def set_modality(self):
        for i, this_map in enumerate(self.surf_maps):
            if self.modality == this_map['kind']:
                break
        map_ch_names = this_map['ch_names']
        map_data = this_map['data']
        surf = this_map['surf']
        if self.modality == 'eeg':
            pick = pick_types(self.evoked.info, meg=False, eeg=True)
        else:
            pick = pick_types(self.evoked.info, meg=True, eeg=False,
                              ref_meg=False)

        ch_names = [self.evoked.ch_names[k] for k in pick]
        set_ch_names = set(ch_names)
        set_map_ch_names = set(map_ch_names)
        if set_ch_names != set_map_ch_names:
            message = ['Channels in map and data do not match.']
            diff = set_map_ch_names - set_ch_names
            if len(diff):
                message += ['%s not in data file. ' % list(diff)]
            diff = set_ch_names - set_map_ch_names
            if len(diff):
                message += ['%s not in map file.' % list(diff)]
            raise RuntimeError(' '.join(message))
        data = np.dot(map_data, self.evoked.data[pick, 0])
        vlim = np.max(np.abs(data))
        colormap = mne_analyze_colormap(format='mayavi')
        x, y, z = surf['rr'].T
        mesh = self.scene.mlab.pipeline.triangular_mesh_source(
            x, y, z, surf['tris'], scalars=data)
        mesh = self.scene.mlab.pipeline.poly_data_normals(mesh)
        mesh.filter.compute_cell_normals = False
        mesh.filter.consistency = False
        mesh.filter.non_manifold_traversal = False
        mesh.filter.splitting = False
        surface = self.scene.mlab.pipeline.surface(
            mesh, color=(0.6, 0.6, 0.6), opacity=0.1,
            vmin=-vlim, vmax=vlim)
        surface.module_manager.scalar_lut_manager.lut.table = colormap
        surface.actor.property.backface_culling = False
        # self.scene.view.set_camera(azimuth=10, elevation=60)


dipole_model = DipoleModel()
dipole_model.configure_traits()
