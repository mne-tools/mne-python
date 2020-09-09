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

from mne import (read_source_estimate, SourceEstimate, MixedSourceEstimate,
                 VolSourceEstimate)
from mne.source_space import (read_source_spaces, vertex_to_mni,
                              setup_volume_source_space)
from mne.datasets import testing
from mne.utils import check_version
from mne.viz._brain import _Brain, _TimeViewer, _LinkViewer, _BrainScraper
from mne.viz._brain.colormap import calculate_lut

from matplotlib import cm, image
import matplotlib.pyplot as plt

data_path = testing.data_path(download=False)
subject_id = 'sample'
subjects_dir = path.join(data_path, 'subjects')
fname_stc = path.join(data_path, 'MEG/sample/sample_audvis_trunc-meg')
fname_label = path.join(data_path, 'MEG/sample/labels/Vis-lh.label')
src_fname = path.join(data_path, 'subjects', 'sample', 'bem',
                      'sample-oct-6-src.fif')


class _Collection(object):
    def __init__(self, actors):
        self._actors = actors

    def GetNumberOfItems(self):
        return len(self._actors)

    def GetItemAsObject(self, ii):
        return self._actors[ii]


class TstVTKPicker(object):
    """Class to test cell picking."""

    def __init__(self, mesh, cell_id, hemi, brain):
        self.mesh = mesh
        self.cell_id = cell_id
        self.point_id = None
        self.hemi = hemi
        self.brain = brain
        self._actors = ()

    def GetCellId(self):
        """Return the picked cell."""
        return self.cell_id

    def GetDataSet(self):
        """Return the picked mesh."""
        return self.mesh

    def GetPickPosition(self):
        """Return the picked position."""
        if self.hemi == 'vol':
            self.point_id = self.cell_id
            return self.brain._data['vol']['grid_coords'][self.cell_id]
        else:
            vtk_cell = self.mesh.GetCell(self.cell_id)
            cell = [vtk_cell.GetPointId(point_id) for point_id
                    in range(vtk_cell.GetNumberOfPoints())]
            self.point_id = cell[0]
            return self.mesh.points[self.point_id]

    def GetProp3Ds(self):
        """Return all picked Prop3Ds."""
        return _Collection(self._actors)

    def GetRenderer(self):
        """Return the "renderer"."""
        return self  # set this to also be the renderer and active camera

    GetActiveCamera = GetRenderer

    def GetPosition(self):
        """Return the position."""
        return np.array(self.GetPickPosition()) - (0, 0, 100)


@testing.requires_testing_data
def test_brain_init(renderer, tmpdir, pixel_ratio):
    """Test initialization of the _Brain instance."""
    from mne.label import read_label
    hemi = 'lh'
    surf = 'inflated'
    cortex = 'low_contrast'
    title = 'test'
    size = (300, 300)

    kwargs = dict(subject_id=subject_id, subjects_dir=subjects_dir)
    with pytest.raises(ValueError, match='"size" parameter must be'):
        _Brain(hemi=hemi, surf=surf, size=[1, 2, 3], **kwargs)
    with pytest.raises(TypeError, match='figure'):
        _Brain(hemi=hemi, surf=surf, figure='foo', **kwargs)
    with pytest.raises(TypeError, match='interaction'):
        _Brain(hemi=hemi, surf=surf, interaction=0, **kwargs)
    with pytest.raises(ValueError, match='interaction'):
        _Brain(hemi=hemi, surf=surf, interaction='foo', **kwargs)
    with pytest.raises(KeyError):
        _Brain(hemi='foo', surf=surf, **kwargs)

    brain = _Brain(hemi=hemi, surf=surf, size=size, title=title,
                   cortex=cortex, units='m', **kwargs)
    assert brain.interaction == 'trackball'
    # add_data
    stc = read_source_estimate(fname_stc)
    fmin = stc.data.min()
    fmax = stc.data.max()
    for h in brain._hemis:
        if h == 'lh':
            hi = 0
        else:
            hi = 1
        hemi_data = stc.data[:len(stc.vertices[hi]), 10]
        hemi_vertices = stc.vertices[hi]

        with pytest.raises(TypeError, match='scale_factor'):
            brain.add_data(hemi_data, hemi=h, scale_factor='foo')
        with pytest.raises(TypeError, match='vector_alpha'):
            brain.add_data(hemi_data, hemi=h, vector_alpha='foo')
        with pytest.raises(ValueError, match='thresh'):
            brain.add_data(hemi_data, hemi=h, thresh=-1)
        with pytest.raises(ValueError, match='remove_existing'):
            brain.add_data(hemi_data, hemi=h, remove_existing=-1)
        with pytest.raises(ValueError, match='time_label_size'):
            brain.add_data(hemi_data, hemi=h, time_label_size=-1,
                           vertices=hemi_vertices)
        with pytest.raises(ValueError, match='is positive'):
            brain.add_data(hemi_data, hemi=h, smoothing_steps=-1,
                           vertices=hemi_vertices)
        with pytest.raises(TypeError, match='int or NoneType'):
            brain.add_data(hemi_data, hemi=h, smoothing_steps='foo')
        with pytest.raises(ValueError, match='dimension mismatch'):
            brain.add_data(array=np.array([0, 1, 2]), hemi=h,
                           vertices=hemi_vertices)
        with pytest.raises(ValueError, match='vertices parameter must not be'):
            brain.add_data(hemi_data, fmin=fmin, hemi=hemi,
                           fmax=fmax, vertices=None)
        with pytest.raises(ValueError, match='has shape'):
            brain.add_data(hemi_data[:, np.newaxis], fmin=fmin, hemi=hemi,
                           fmax=fmax, vertices=None, time=[0, 1])

        brain.add_data(hemi_data, fmin=fmin, hemi=h, fmax=fmax,
                       colormap='hot', vertices=hemi_vertices,
                       smoothing_steps='nearest', colorbar=(0, 0), time=None)
        assert brain.data['lh']['array'] is hemi_data
        assert brain.views == ['lateral']
        assert brain.hemis == ('lh',)
        brain.add_data(hemi_data[:, np.newaxis], fmin=fmin, hemi=h, fmax=fmax,
                       colormap='hot', vertices=hemi_vertices,
                       smoothing_steps=1, initial_time=0., colorbar=False,
                       time=[0])
        brain.set_time_point(0)  # should hit _safe_interp1d

        with pytest.raises(ValueError, match='consistent with'):
            brain.add_data(hemi_data[:, np.newaxis], fmin=fmin, hemi=h,
                           fmax=fmax, colormap='hot', vertices=hemi_vertices,
                           smoothing_steps='nearest', colorbar=False,
                           time=[1])
        with pytest.raises(ValueError, match='different from'):
            brain.add_data(hemi_data[:, np.newaxis][:, [0, 0]],
                           fmin=fmin, hemi=h, fmax=fmax, colormap='hot',
                           vertices=hemi_vertices)
        with pytest.raises(ValueError, match='need shape'):
            brain.add_data(hemi_data[:, np.newaxis], time=[0, 1],
                           fmin=fmin, hemi=h, fmax=fmax, colormap='hot',
                           vertices=hemi_vertices)
        with pytest.raises(ValueError, match='If array has 3'):
            brain.add_data(hemi_data[:, np.newaxis, np.newaxis],
                           fmin=fmin, hemi=h, fmax=fmax, colormap='hot',
                           vertices=hemi_vertices)
    # add label
    label = read_label(fname_label)
    brain.add_label(label, scalar_thresh=0.)
    brain.remove_labels()
    brain.add_label(fname_label)
    brain.add_label('V1', borders=True)
    brain.remove_labels()
    brain.remove_labels()

    # add foci
    brain.add_foci([0], coords_as_verts=True,
                   hemi=hemi, color='blue')

    # add text
    brain.add_text(x=0, y=0, text='foo')

    # add annotation
    annots = ['aparc', path.join(subjects_dir, 'fsaverage', 'label',
                                 'lh.PALS_B12_Lobes.annot')]
    borders = [True, 2]
    alphas = [1, 0.5]
    colors = [None, 'r']
    brain = _Brain(subject_id='fsaverage', hemi=hemi, size=size,
                   surf='inflated', subjects_dir=subjects_dir)
    for a, b, p, color in zip(annots, borders, alphas, colors):
        brain.add_annotation(a, b, p, color=color)

    brain.show_view(dict(focalpoint=(1e-5, 1e-5, 1e-5)), roll=1, distance=500)

    # image and screenshot
    fname = path.join(str(tmpdir), 'test.png')
    assert not path.isfile(fname)
    brain.save_image(fname)
    assert path.isfile(fname)
    brain.show_view(view=dict(azimuth=180., elevation=90.))
    img = brain.screenshot(mode='rgb')
    if renderer._get_3d_backend() == 'mayavi':
        pixel_ratio = 1.  # no HiDPI when using the testing backend
    want_size = np.array([size[0] * pixel_ratio, size[1] * pixel_ratio, 3])
    assert_allclose(img.shape, want_size)
    brain.close()


@testing.requires_testing_data
@pytest.mark.slowtest
def test_brain_save_movie(tmpdir, renderer):
    """Test saving a movie of a _Brain instance."""
    if renderer._get_3d_backend() == "mayavi":
        pytest.skip('Save movie only supported on PyVista')
    brain_data = _create_testing_brain(hemi='lh', time_viewer=False)
    filename = str(path.join(tmpdir, "brain_test.mov"))
    brain_data.save_movie(filename, time_dilation=1,
                          interpolation='nearest')
    assert path.isfile(filename)
    brain_data.close()


@testing.requires_testing_data
@pytest.mark.slowtest
def test_brain_timeviewer(renderer_interactive, pixel_ratio):
    """Test _TimeViewer primitives."""
    if renderer_interactive._get_3d_backend() != 'pyvista':
        pytest.skip('TimeViewer tests only supported on PyVista')
    brain_data = _create_testing_brain(hemi='both', show_traces=False)

    with pytest.raises(RuntimeError, match='already'):
        _TimeViewer(brain_data)
    time_viewer = brain_data.time_viewer
    time_viewer.callbacks["time"](value=0)
    time_viewer.callbacks["orientation_lh_0_0"](
        value='lat',
        update_widget=True
    )
    time_viewer.callbacks["orientation_lh_0_0"](
        value='medial',
        update_widget=True
    )
    time_viewer.callbacks["time"](
        value=0.0,
        time_as_index=False,
    )
    time_viewer.callbacks["smoothing"](value=1)
    time_viewer.callbacks["fmin"](value=12.0)
    time_viewer.callbacks["fmax"](value=4.0)
    time_viewer.callbacks["fmid"](value=6.0)
    time_viewer.callbacks["fmid"](value=4.0)
    time_viewer.callbacks["fscale"](value=1.1)
    time_viewer.callbacks["fmin"](value=12.0)
    time_viewer.callbacks["fmid"](value=4.0)
    time_viewer.toggle_interface()
    time_viewer.callbacks["playback_speed"](value=0.1)
    time_viewer.toggle_playback()
    time_viewer.apply_auto_scaling()
    time_viewer.restore_user_scaling()
    time_viewer.reset()
    plt.close('all')
    time_viewer.help()
    assert len(plt.get_fignums()) == 1
    plt.close('all')

    # screenshot
    brain_data.show_view(view=dict(azimuth=180., elevation=90.))
    img = brain_data.screenshot(mode='rgb')
    want_shape = np.array([300 * pixel_ratio, 300 * pixel_ratio, 3])
    assert_allclose(img.shape, want_shape)


@testing.requires_testing_data
@pytest.mark.parametrize('hemi', [
    'lh',
    pytest.param('rh', marks=pytest.mark.slowtest),
    pytest.param('split', marks=pytest.mark.slowtest),
    pytest.param('both', marks=pytest.mark.slowtest),
])
@pytest.mark.parametrize('src', [
    'surface',
    pytest.param('volume', marks=pytest.mark.slowtest),
    pytest.param('mixed', marks=pytest.mark.slowtest),
])
@pytest.mark.slowtest
def test_brain_timeviewer_traces(renderer_interactive, hemi, src, tmpdir):
    """Test _TimeViewer traces."""
    if renderer_interactive._get_3d_backend() != 'pyvista':
        pytest.skip('Only PyVista supports traces')
    brain_data = _create_testing_brain(
        hemi=hemi, surf='white', src=src, show_traces=0.5, initial_time=0,
        volume_options=None,  # for speed, don't upsample
    )
    with pytest.raises(RuntimeError, match='already'):
        _TimeViewer(brain_data)
    time_viewer = brain_data.time_viewer
    assert time_viewer.show_traces
    assert hasattr(time_viewer, "picked_points")
    assert hasattr(time_viewer, "_spheres")

    # test points picked by default
    picked_points = brain_data.get_picked_points()
    spheres = time_viewer._spheres
    hemi_str = list()
    if src in ('surface', 'mixed'):
        hemi_str.extend([hemi] if hemi in ('lh', 'rh') else ['lh', 'rh'])
    if src in ('mixed', 'volume'):
        hemi_str.extend(['vol'])
    for current_hemi in hemi_str:
        assert len(picked_points[current_hemi]) == 1
    n_spheres = len(hemi_str)
    if hemi == 'split' and src in ('mixed', 'volume'):
        n_spheres += 1
    assert len(spheres) == n_spheres

    # test removing points
    time_viewer.clear_points()
    assert len(spheres) == 0
    for key in ('lh', 'rh', 'vol'):
        assert len(picked_points[key]) == 0

    # test picking a cell at random
    rng = np.random.RandomState(0)
    for idx, current_hemi in enumerate(hemi_str):
        assert len(spheres) == 0
        if current_hemi == 'vol':
            current_mesh = brain_data._data['vol']['grid']
            vertices = brain_data._data['vol']['vertices']
            values = current_mesh.cell_arrays['values'][vertices]
            cell_id = vertices[np.argmax(np.abs(values))]
        else:
            current_mesh = brain_data._hemi_meshes[current_hemi]
            cell_id = rng.randint(0, current_mesh.n_cells)
        test_picker = TstVTKPicker(None, None, current_hemi, brain_data)
        assert time_viewer.on_pick(test_picker, None) is None
        test_picker = TstVTKPicker(
            current_mesh, cell_id, current_hemi, brain_data)
        assert cell_id == test_picker.cell_id
        assert test_picker.point_id is None
        time_viewer.on_pick(test_picker, None)
        assert test_picker.point_id is not None
        assert len(picked_points[current_hemi]) == 1
        assert picked_points[current_hemi][0] == test_picker.point_id
        assert len(spheres) > 0
        sphere = spheres[-1]
        vertex_id = sphere._vertex_id
        assert vertex_id == test_picker.point_id
        line = sphere._line

        hemi_prefix = current_hemi[0].upper()
        if current_hemi == 'vol':
            assert hemi_prefix + ':' in line.get_label()
            assert 'MNI' in line.get_label()
            continue  # the MNI conversion is more complex
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

        # remove the sphere by clicking in its vicinity
        old_len = len(spheres)
        test_picker._actors = sum((s._actors for s in spheres), [])
        time_viewer.on_pick(test_picker, None)
        assert len(spheres) < old_len

    # and the scraper for it (will close the instance)
    if not check_version('sphinx_gallery'):
        return
    screenshot = brain_data.screenshot()
    fnames = [str(tmpdir.join('temp.png'))]
    block_vars = dict(image_path_iterator=iter(fnames),
                      example_globals=dict(brain=brain_data))
    gallery_conf = dict(src_dir=str(tmpdir))
    scraper = _BrainScraper()
    rst = scraper(None, block_vars, gallery_conf)
    assert 'temp.png' in rst
    assert path.isfile(fnames[0])
    img = image.imread(fnames[0])
    assert img.shape[1] == screenshot.shape[1]  # same width
    assert img.shape[0] > screenshot.shape[0]  # larger height


@testing.requires_testing_data
@pytest.mark.slowtest
def test_brain_linkviewer(renderer_interactive):
    """Test _LinkViewer primitives."""
    if renderer_interactive._get_3d_backend() != 'pyvista':
        pytest.skip('Linkviewer only supported on PyVista')
    brain1 = _create_testing_brain(hemi='lh', show_traces=False)
    brain2 = _create_testing_brain(hemi='lh', show_traces=True)
    brain1._times = brain1._times * 2
    with pytest.warns(RuntimeWarning, match='linking time'):
        link_viewer = _LinkViewer(
            [brain1, brain2],
            time=True,
            camera=False,
            colorbar=False,
            picking=False,
        )

    brain_data = _create_testing_brain(hemi='split', show_traces=True)
    link_viewer = _LinkViewer(
        [brain2, brain_data],
        time=True,
        camera=True,
        colorbar=True,
        picking=True,
    )
    link_viewer.set_time_point(value=0)
    link_viewer.time_viewers[0].mpl_canvas.time_func(0)
    link_viewer.set_fmin(0)
    link_viewer.set_fmid(0.5)
    link_viewer.set_fmax(1)
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


def _create_testing_brain(hemi, surf='inflated', src='surface', size=300,
                          **kwargs):
    assert src in ('surface', 'mixed', 'volume')
    meth = 'plot'
    if src in ('surface', 'mixed'):
        sample_src = read_source_spaces(src_fname)
        klass = MixedSourceEstimate if src == 'mixed' else SourceEstimate
    if src in ('volume', 'mixed'):
        vol_src = setup_volume_source_space(
            subject_id, 7., mri='aseg.mgz',
            volume_label='Left-Cerebellum-Cortex',
            subjects_dir=subjects_dir, add_interpolator=False)
        assert len(vol_src) == 1
        assert vol_src[0]['nuse'] == 150
        if src == 'mixed':
            sample_src = sample_src + vol_src
        else:
            sample_src = vol_src
            klass = VolSourceEstimate
            meth = 'plot_3d'
    assert sample_src.kind == src

    # dense version
    rng = np.random.RandomState(0)
    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.zeros((n_verts * n_time))
    stc_size = stc_data.size
    stc_data[(rng.rand(stc_size // 20) * stc_size).astype(int)] = \
        rng.rand(stc_data.size // 20)
    stc_data.shape = (n_verts, n_time)
    stc = klass(stc_data, vertices, 1, 1)

    fmin = stc.data.min()
    fmax = stc.data.max()
    fmid = (fmin + fmax) / 2.
    brain_data = getattr(stc, meth)(
        subject=subject_id, hemi=hemi, surface=surf, size=size,
        subjects_dir=subjects_dir, colormap='hot',
        clim=dict(kind='value', lims=(fmin, fmid, fmax)), src=sample_src,
        **kwargs)
    return brain_data
