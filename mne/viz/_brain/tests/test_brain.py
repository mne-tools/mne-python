# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#
# License: Simplified BSD

import os.path as path

import pytest
import numpy as np
from numpy.testing import assert_allclose

from mne import SourceEstimate, read_source_estimate
from mne.source_space import read_source_spaces, vertex_to_mni
from mne.datasets import testing
from mne.viz._brain import _Brain, _TimeViewer, _LinkViewer
from mne.viz._brain.colormap import calculate_lut

from matplotlib import cm

data_path = testing.data_path(download=False)
subject_id = 'sample'
subjects_dir = path.join(data_path, 'subjects')
fname_stc = path.join(data_path, 'MEG/sample/sample_audvis_trunc-meg')
fname_label = path.join(data_path, 'MEG/sample/labels/Vis-lh.label')
src_fname = path.join(data_path, 'subjects', 'sample', 'bem',
                      'sample-oct-6-src.fif')
surf = 'inflated'


class TstVTKPicker(object):
    """Class to test cell picking."""

    def __init__(self, mesh, cell_id):
        self.mesh = mesh
        self.cell_id = cell_id
        self.point_id = None

    def GetCellId(self):
        """Return the picked cell."""
        return self.cell_id

    def GetDataSet(self):
        """Return the picked mesh."""
        return self.mesh

    def GetPickPosition(self):
        """Return the picked position."""
        vtk_cell = self.mesh.GetCell(self.cell_id)
        cell = [vtk_cell.GetPointId(point_id) for point_id
                in range(vtk_cell.GetNumberOfPoints())]
        self.point_id = cell[0]
        return self.mesh.points[self.point_id]


@testing.requires_testing_data
def test_brain_init(renderer):
    """Test initialization of the _Brain instance."""
    hemi = 'both'

    with pytest.raises(ValueError, match='"size" parameter must be'):
        _Brain(subject_id=subject_id, hemi=hemi, surf=surf, size=[1, 2, 3])
    with pytest.raises(TypeError, match='figure'):
        _Brain(subject_id=subject_id, hemi=hemi, surf=surf, figure='foo')
    with pytest.raises(ValueError, match='interaction'):
        _Brain(subject_id=subject_id, hemi=hemi, surf=surf, interaction=0)
    with pytest.raises(KeyError):
        _Brain(subject_id=subject_id, hemi='foo', surf=surf)

    brain = _Brain(subject_id, hemi, surf, size=(300, 300),
                   subjects_dir=subjects_dir, title='test')
    brain.show_view(view=dict(azimuth=180., elevation=90.))
    brain.close()


@testing.requires_testing_data
def test_brain_screenshot(renderer):
    """Test screenshot of a _Brain instance."""
    brain = _Brain(subject_id, hemi='both', size=600,
                   surf=surf, subjects_dir=subjects_dir)
    img = brain.screenshot(mode='rgb')
    assert(img.shape == (600, 600, 3))
    brain.close()


@testing.requires_testing_data
def test_brain_add_data(renderer):
    """Test adding data in _Brain instance."""
    stc = read_source_estimate(fname_stc)

    hemi = 'lh'
    hemi_data = stc.data[:len(stc.vertices[0]), 10]
    hemi_vertices = stc.vertices[0]
    fmin = stc.data.min()
    fmax = stc.data.max()

    brain_data = _Brain(subject_id, hemi, surf, size=300,
                        subjects_dir=subjects_dir)

    with pytest.raises(TypeError, match='scale_factor'):
        brain_data.add_data(hemi_data, scale_factor='foo')
    with pytest.raises(TypeError, match='vector_alpha'):
        brain_data.add_data(hemi_data, vector_alpha='foo')
    with pytest.raises(ValueError, match='thresh'):
        brain_data.add_data(hemi_data, thresh=-1)
    with pytest.raises(ValueError, match='remove_existing'):
        brain_data.add_data(hemi_data, remove_existing=-1)
    with pytest.raises(ValueError, match='time_label_size'):
        brain_data.add_data(hemi_data, time_label_size=-1)
    with pytest.raises(ValueError, match='is positive'):
        brain_data.add_data(hemi_data, smoothing_steps=-1)
    with pytest.raises(TypeError, match='int or NoneType'):
        brain_data.add_data(hemi_data, smoothing_steps='foo')
    with pytest.raises(ValueError):
        brain_data.add_data(array=np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi,
                            fmax=fmax, vertices=None)

    brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi, fmax=fmax,
                        colormap='hot', vertices=hemi_vertices,
                        smoothing_steps='nearest', colorbar=False, time=None)
    brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi, fmax=fmax,
                        colormap='hot', vertices=hemi_vertices,
                        smoothing_steps=1, initial_time=0., colorbar=False,
                        time=None)
    brain_data.close()


@testing.requires_testing_data
def test_brain_add_label(renderer):
    """Test adding data in _Brain instance."""
    from mne.label import read_label
    brain = _Brain(subject_id, hemi='lh', size=500,
                   surf=surf, subjects_dir=subjects_dir)
    label = read_label(fname_label)
    brain.add_label(label, scalar_thresh=0.)
    brain.close()

    brain = _Brain(subject_id, hemi='split', size=500,
                   surf=surf, subjects_dir=subjects_dir)
    brain.add_label(fname_label)
    brain.close()


@testing.requires_testing_data
def test_brain_add_foci(renderer):
    """Test adding foci in _Brain instance."""
    brain = _Brain(subject_id, hemi='lh', size=500,
                   surf=surf, subjects_dir=subjects_dir)
    brain.add_foci([0], coords_as_verts=True,
                   hemi='lh', color='blue')
    brain.close()


@testing.requires_testing_data
def test_brain_add_text(renderer):
    """Test adding text in _Brain instance."""
    brain = _Brain(subject_id, hemi='lh', size=250,
                   surf=surf, subjects_dir=subjects_dir)
    brain.add_text(x=0, y=0, text='foo')
    brain.close()


@testing.requires_testing_data
def test_brain_timeviewer(renderer_interactive):
    """Test _TimeViewer primitives."""
    if renderer_interactive.get_3d_backend() != 'pyvista':
        pytest.skip()
    brain_data = _create_testing_brain(hemi='both')

    time_viewer = _TimeViewer(brain_data)
    time_viewer.time_call(value=0)
    time_viewer.orientation_call(value='lat', update_widget=True)
    time_viewer.orientation_call(value='medial', update_widget=True)
    time_viewer.smoothing_call(value=1)
    time_viewer.fmin_call(value=12.0)
    time_viewer.fmax_call(value=4.0)
    time_viewer.fmid_call(value=6.0)
    time_viewer.fmid_call(value=4.0)
    time_viewer.fscale_call(value=1.1)
    time_viewer.toggle_interface()
    time_viewer.playback_speed_call(value=0.1)
    time_viewer.toggle_playback()
    time_viewer.apply_auto_scaling()
    time_viewer.restore_user_scaling()


@testing.requires_testing_data
@pytest.mark.parametrize('hemi', ['lh', 'rh', 'split', 'both'])
def test_brain_timeviewer_traces(renderer_interactive, hemi):
    """Test _TimeViewer traces."""
    if renderer_interactive.get_3d_backend() != 'pyvista':
        pytest.skip()
    brain_data = _create_testing_brain(hemi=hemi)
    time_viewer = _TimeViewer(brain_data, show_traces=True)
    assert hasattr(time_viewer, "picked_points")
    assert hasattr(time_viewer, "_spheres")

    # test points picked by default
    picked_points = time_viewer.picked_points
    spheres = time_viewer._spheres
    hemi_str = [hemi] if hemi in ('lh', 'rh') else ['lh', 'rh']
    for current_hemi in hemi_str:
        assert len(picked_points[current_hemi]) == 1
    assert len(spheres) == len(hemi_str)

    # test removing points
    time_viewer.clear_points()
    assert len(picked_points['lh']) == 0
    assert len(picked_points['rh']) == 0

    # test picking a cell at random
    for idx, current_hemi in enumerate(hemi_str):
        current_mesh = brain_data._hemi_meshes[current_hemi]
        cell_id = np.random.randint(0, current_mesh.n_cells)
        test_picker = TstVTKPicker(current_mesh, cell_id)
        assert cell_id == test_picker.cell_id
        assert test_picker.point_id is None
        time_viewer.on_pick(test_picker, None)
        assert test_picker.point_id is not None
        assert len(picked_points[current_hemi]) == 1
        assert picked_points[current_hemi][0] == test_picker.point_id
        sphere = spheres[idx]
        vertex_id = sphere._vertex_id
        assert vertex_id == test_picker.point_id
        line = sphere._line

        hemi_prefix = 'L' if current_hemi == 'lh' else 'R'
        hemi_int = 0 if current_hemi == 'lh' else 1
        mni = vertex_to_mni(
            vertices=vertex_id,
            hemis=hemi_int,
            subject=brain_data._subject_id,
            subjects_dir=brain_data._subjects_dir
        )
        label = "{}:{} MNI: {}".format(
            hemi_prefix, str(vertex_id).ljust(6),
            ', '.join('%5.1f' % m for m in mni))

        assert line.get_label() == label
    assert len(spheres) == len(hemi_str)


@testing.requires_testing_data
def test_brain_linkviewer(renderer_interactive):
    """Test _LinkViewer primitives."""
    if renderer_interactive.get_3d_backend() != 'pyvista':
        pytest.skip()
    brain_data = _create_testing_brain(hemi='split')
    _TimeViewer(brain_data)

    link_viewer = _LinkViewer([brain_data])
    link_viewer.set_time_point(value=0)
    link_viewer.set_playback_speed(value=0.1)
    link_viewer.toggle_playback()


def test_brain_colormap():
    """Test brain's colormap functions."""
    colormap = "coolwarm"
    alpha = 1.0
    fmin = 0.0
    fmid = 0.5
    fmax = 1.0
    center = None
    calculate_lut(colormap, alpha=alpha, fmin=fmin,
                  fmid=fmid, fmax=fmax, center=center)
    center = 0.0
    colormap = cm.get_cmap(colormap)
    calculate_lut(colormap, alpha=alpha, fmin=fmin,
                  fmid=fmid, fmax=fmax, center=center)

    cmap = cm.get_cmap(colormap)
    zero_alpha = np.array([1., 1., 1., 0])
    half_alpha = np.array([1., 1., 1., 0.5])
    atol = 1.5 / 256.

    # fmin < fmid < fmax
    lut = calculate_lut(colormap, alpha, 1, 2, 3)
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0) * zero_alpha, atol=atol)
    assert_allclose(lut[127], cmap(0.5), atol=atol)
    assert_allclose(lut[-1], cmap(1.), atol=atol)
    # divergent
    lut = calculate_lut(colormap, alpha, 0, 1, 2, 0)
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0), atol=atol)
    assert_allclose(lut[63], cmap(0.25), atol=atol)
    assert_allclose(lut[127], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[192], cmap(0.75), atol=atol)
    assert_allclose(lut[-1], cmap(1.), atol=atol)

    # fmin == fmid == fmax
    lut = calculate_lut(colormap, alpha, 1, 1, 1)
    zero_alpha = np.array([1., 1., 1., 0])
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0) * zero_alpha, atol=atol)
    assert_allclose(lut[1], cmap(0.5), atol=atol)
    assert_allclose(lut[-1], cmap(1.), atol=atol)
    # divergent
    lut = calculate_lut(colormap, alpha, 0, 0, 0, 0)
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0), atol=atol)
    assert_allclose(lut[127], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[-1], cmap(1.), atol=atol)

    # fmin == fmid < fmax
    lut = calculate_lut(colormap, alpha, 1, 1, 2)
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0.) * zero_alpha, atol=atol)
    assert_allclose(lut[1], cmap(0.5), atol=atol)
    assert_allclose(lut[-1], cmap(1.), atol=atol)
    # divergent
    lut = calculate_lut(colormap, alpha, 1, 1, 2, 0)
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0), atol=atol)
    assert_allclose(lut[62], cmap(0.245), atol=atol)
    assert_allclose(lut[64], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[127], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[191], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[193], cmap(0.755), atol=atol)
    assert_allclose(lut[-1], cmap(1.), atol=atol)
    lut = calculate_lut(colormap, alpha, 0, 0, 1, 0)
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0), atol=atol)
    assert_allclose(lut[126], cmap(0.25), atol=atol)
    assert_allclose(lut[127], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[129], cmap(0.75), atol=atol)
    assert_allclose(lut[-1], cmap(1.), atol=atol)

    # fmin < fmid == fmax
    lut = calculate_lut(colormap, alpha, 1, 2, 2)
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0) * zero_alpha, atol=atol)
    assert_allclose(lut[-2], cmap(0.5), atol=atol)
    assert_allclose(lut[-1], cmap(1.), atol=atol)
    # divergent
    lut = calculate_lut(colormap, alpha, 1, 2, 2, 0)
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0), atol=atol)
    assert_allclose(lut[1], cmap(0.25), atol=2 * atol)
    assert_allclose(lut[32], cmap(0.375) * half_alpha, atol=atol)
    assert_allclose(lut[64], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[127], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[191], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[223], cmap(0.625) * half_alpha, atol=atol)
    assert_allclose(lut[-2], cmap(0.7475), atol=2 * atol)
    assert_allclose(lut[-1], cmap(1.), atol=2 * atol)
    lut = calculate_lut(colormap, alpha, 0, 1, 1, 0)
    assert lut.shape == (256, 4)
    assert_allclose(lut[0], cmap(0), atol=atol)
    assert_allclose(lut[1], cmap(0.25), atol=2 * atol)
    assert_allclose(lut[64], cmap(0.375) * half_alpha, atol=atol)
    assert_allclose(lut[127], cmap(0.5) * zero_alpha, atol=atol)
    assert_allclose(lut[191], cmap(0.625) * half_alpha, atol=atol)
    assert_allclose(lut[-2], cmap(0.75), atol=2 * atol)
    assert_allclose(lut[-1], cmap(1.), atol=atol)

    with pytest.raises(ValueError, match=r'.*fmin \(1\) <= fmid \(0\) <= fma'):
        calculate_lut(colormap, alpha, 1, 0, 2)


def _create_testing_brain(hemi):
    sample_src = read_source_spaces(src_fname)

    # dense version
    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.zeros((n_verts * n_time))
    stc_size = stc_data.size
    stc_data[(np.random.rand(stc_size // 20) * stc_size).astype(int)] = \
        np.random.RandomState(0).rand(stc_data.size // 20)
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1)

    fmin = stc.data.min()
    fmax = stc.data.max()
    brain_data = _Brain(subject_id, hemi, surf, size=300,
                        subjects_dir=subjects_dir)
    hemi_list = ['lh', 'rh'] if hemi in ['both', 'split'] else [hemi]
    for hemi_str in hemi_list:
        hemi_idx = 0 if hemi_str == 'lh' else 1
        data = getattr(stc, hemi_str + '_data')
        vertices = stc.vertices[hemi_idx]
        brain_data.add_data(data, fmin=fmin, hemi=hemi_str, fmax=fmax,
                            colormap='hot', vertices=vertices,
                            colorbar=True)
    return brain_data
