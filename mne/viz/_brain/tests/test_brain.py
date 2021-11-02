# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#
# License: Simplified BSD

import os
import os.path as op
import sys

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mne import (read_source_estimate, read_evokeds, read_cov,
                 read_forward_solution, pick_types_forward,
                 SourceEstimate, MixedSourceEstimate, write_surface,
                 VolSourceEstimate, vertex_to_mni)
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.source_space import (read_source_spaces,
                              setup_volume_source_space)
from mne.datasets import testing
from mne.fixes import _cell_data
from mne.io import read_info
from mne.utils import check_version, requires_pysurfer
from mne.label import read_label
from mne.viz._brain import Brain, _LinkViewer, _BrainScraper, _LayeredMesh
from mne.viz._brain.colormap import calculate_lut

from matplotlib import cm, image
from matplotlib.lines import Line2D

data_path = testing.data_path(download=False)
subject_id = 'sample'
subjects_dir = op.join(data_path, 'subjects')
sample_dir = op.join(data_path, 'MEG', 'sample')
fname_raw_testing = op.join(sample_dir, 'sample_audvis_trunc_raw.fif')
fname_trans = op.join(sample_dir, 'sample_audvis_trunc-trans.fif')
fname_stc = op.join(sample_dir, 'sample_audvis_trunc-meg')
fname_label = op.join(sample_dir, 'labels', 'Vis-lh.label')
fname_cov = op.join(sample_dir, 'sample_audvis_trunc-cov.fif')
fname_evoked = op.join(sample_dir, 'sample_audvis_trunc-ave.fif')
fname_fwd = op.join(sample_dir, 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
src_fname = op.join(subjects_dir, subject_id, 'bem', 'sample-oct-6-src.fif')


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


def test_layered_mesh(renderer_interactive_pyvistaqt):
    """Test management of scalars/colormap overlay."""
    mesh = _LayeredMesh(
        renderer=renderer_interactive_pyvistaqt._get_renderer(size=(300, 300)),
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        triangles=np.array([[0, 1, 2], [1, 2, 3]]),
        normals=np.array([[0, 0, 1]] * 4),
    )
    assert not mesh._is_mapped
    mesh.map()
    assert mesh._is_mapped
    assert mesh._current_colors is None
    assert mesh._cached_colors is None
    mesh.update()
    assert len(mesh._overlays) == 0
    mesh.add_overlay(
        scalars=np.array([0, 1, 1, 0]),
        colormap=np.array([(1, 1, 1, 1), (0, 0, 0, 0)]),
        rng=[0, 1],
        opacity=None,
        name='test1',
    )
    assert mesh._current_colors is not None
    assert mesh._cached_colors is None
    assert len(mesh._overlays) == 1
    assert 'test1' in mesh._overlays
    mesh.add_overlay(
        scalars=np.array([1, 0, 0, 1]),
        colormap=np.array([(1, 1, 1, 1), (0, 0, 0, 0)]),
        rng=[0, 1],
        opacity=None,
        name='test2',
    )
    assert mesh._current_colors is not None
    assert mesh._cached_colors is not None
    assert len(mesh._overlays) == 2
    assert 'test2' in mesh._overlays
    mesh.remove_overlay('test2')
    assert 'test2' not in mesh._overlays
    mesh.update()
    assert len(mesh._overlays) == 1
    mesh._clean()


@testing.requires_testing_data
def test_brain_gc(renderer_pyvistaqt, brain_gc):
    """Test that a minimal version of Brain gets GC'ed."""
    brain = Brain('fsaverage', 'both', 'inflated', subjects_dir=subjects_dir)
    brain.close()


@requires_pysurfer
@testing.requires_testing_data
def test_brain_routines(renderer, brain_gc):
    """Test backend agnostic Brain routines."""
    brain_klass = renderer.get_brain_class()
    if renderer.get_3d_backend() == "mayavi":
        from surfer import Brain
    else:  # PyVista
        from mne.viz._brain import Brain
    assert brain_klass == Brain


@testing.requires_testing_data
def test_brain_init(renderer_pyvistaqt, tmp_path, pixel_ratio, brain_gc):
    """Test initialization of the Brain instance."""
    from mne.source_estimate import _BaseSourceEstimate

    class FakeSTC(_BaseSourceEstimate):
        def __init__(self):
            pass
    hemi = 'lh'
    surf = 'inflated'
    cortex = 'low_contrast'
    title = 'test'
    size = (300, 300)

    kwargs = dict(subject_id=subject_id, subjects_dir=subjects_dir)
    with pytest.raises(ValueError, match='"size" parameter must be'):
        Brain(hemi=hemi, surf=surf, size=[1, 2, 3], **kwargs)
    with pytest.raises(ValueError, match='.*hemi.*Allowed values.*'):
        Brain(hemi='foo', surf=surf, **kwargs)
    with pytest.raises(ValueError, match='.*view.*Allowed values.*'):
        Brain(hemi='lh', surf=surf, views='foo', **kwargs)
    with pytest.raises(TypeError, match='figure'):
        Brain(hemi=hemi, surf=surf, figure='foo', **kwargs)
    with pytest.raises(TypeError, match='interaction'):
        Brain(hemi=hemi, surf=surf, interaction=0, **kwargs)
    with pytest.raises(ValueError, match='interaction'):
        Brain(hemi=hemi, surf=surf, interaction='foo', **kwargs)
    with pytest.raises(FileNotFoundError, match=r'lh\.whatever'):
        Brain(subject_id, 'lh', 'whatever')
    with pytest.raises(ValueError, match='`surf` cannot be seghead'):
        Brain(hemi='lh', surf='seghead', **kwargs)
    with pytest.raises(ValueError, match='RGB argument'):
        Brain('sample', cortex='badcolor')
    Brain(subject_id, hemi=None, surf=None)  # test no surfaces
    renderer_pyvistaqt.backend._close_all()

    brain = Brain(hemi=hemi, surf=surf, size=size, title=title,
                  cortex=cortex, units='m',
                  silhouette=dict(decimate=0.95), **kwargs)
    assert 'data' not in brain._actors
    with pytest.raises(TypeError, match='not supported'):
        brain._check_stc(hemi='lh', array=FakeSTC(), vertices=None)
    with pytest.raises(ValueError, match='add_data'):
        brain.setup_time_viewer(time_viewer=True)
    brain._hemi = 'foo'  # for testing: hemis
    with pytest.raises(ValueError, match='not be None'):
        brain._check_hemi(hemi=None)
    with pytest.raises(ValueError, match='Invalid.*hemi.*Allowed'):
        brain._check_hemi(hemi='foo')
    brain._hemi = hemi  # end testing: hemis
    with pytest.raises(ValueError, match='bool or positive'):
        brain._to_borders(None, None, 'foo')
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
        with pytest.raises(ValueError, match='brain has no defined times'):
            brain.set_time(0.)
        assert brain.data['lh']['array'] is hemi_data
        assert brain.views == ['lateral']
        assert brain.hemis == ('lh',)
        brain.add_data(hemi_data[:, np.newaxis], fmin=fmin, hemi=h, fmax=fmax,
                       colormap='hot', vertices=hemi_vertices,
                       smoothing_steps=1, initial_time=0., colorbar=False,
                       time=[0])
        with pytest.raises(ValueError, match='the range of available times'):
            brain.set_time(7.)
        brain.set_time(0.)
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
    assert len(brain._actors['data']) == 4
    brain.remove_data()
    assert 'data' not in brain._actors

    # add label
    label = read_label(fname_label)
    with pytest.raises(ValueError, match="not a filename"):
        brain.add_label(0)
    with pytest.raises(ValueError, match="does not exist"):
        brain.add_label('foo', subdir='bar')
    label.name = None  # test unnamed label
    brain.add_label(label, scalar_thresh=0., color="green")
    assert isinstance(brain.labels[label.hemi], list)
    overlays = brain._layered_meshes[label.hemi]._overlays
    assert 'unnamed0' in overlays
    assert np.allclose(overlays['unnamed0']._colormap[0],
                       [0, 0, 0, 0])  # first component is transparent
    assert np.allclose(overlays['unnamed0']._colormap[1],
                       [0, 128, 0, 255])  # second is green
    brain.remove_labels()
    assert 'unnamed0' not in overlays
    brain.add_label(fname_label)
    brain.add_label('V1', borders=True)
    brain.remove_labels()
    brain.remove_labels()

    # add foci
    brain.add_foci([0], coords_as_verts=True,
                   hemi=hemi, color='blue')

    # add head and skull
    brain.add_head(color='red', alpha=0.1)
    brain.remove_head()
    brain.add_skull(outer=True, color='green', alpha=0.1)
    brain.remove_skull()

    # add volume labels
    brain.add_volume_labels(
        aseg='aseg', labels=('Brain-Stem', 'Left-Hippocampus',
                             'Left-Amygdala'))
    brain.remove_volume_labels()

    # add sensors
    info = read_info(fname_raw_testing)
    brain.add_sensors(info, trans=fname_trans)
    for kind in ('meg', 'eeg', 'fnirs', 'ecog', 'seeg', 'dbs', 'helmet'):
        brain.remove_sensors(kind)
    brain.add_sensors(info, trans=fname_trans)
    brain.remove_sensors()

    info['chs'][0]['coord_frame'] = 99
    with pytest.raises(RuntimeError, match='must be "meg", "head" or "mri"'):
        brain.add_sensors(info, trans=fname_trans)

    # add text
    brain.add_text(x=0, y=0, text='foo')
    with pytest.raises(ValueError, match='already exists'):
        brain.add_text(x=0, y=0, text='foo')
    brain.remove_text('foo')
    brain.add_text(x=0, y=0, text='foo')
    brain.remove_text()

    brain.close()

    # add annotation
    annots = ['aparc', op.join(subjects_dir, 'fsaverage', 'label',
                               'lh.PALS_B12_Lobes.annot')]
    borders = [True, 2]
    alphas = [1, 0.5]
    colors = [None, 'r']
    brain = Brain(subject_id='fsaverage', hemi='both', size=size,
                  surf='inflated', subjects_dir=subjects_dir)
    with pytest.raises(RuntimeError, match="both hemispheres"):
        brain.add_annotation(annots[-1])
    with pytest.raises(ValueError, match="does not exist"):
        brain.add_annotation('foo')
    brain.close()
    brain = Brain(subject_id='fsaverage', hemi=hemi, size=size,
                  surf='inflated', subjects_dir=subjects_dir)
    for a, b, p, color in zip(annots, borders, alphas, colors):
        brain.add_annotation(a, b, p, color=color)

    view_args = dict(roll=1, distance=500, focalpoint=(1e-5, 1e-5, 1e-5))
    cam = brain._renderer.figure.plotter.camera
    previous_roll = cam.GetRoll()
    brain.show_view(**view_args)
    assert_allclose(cam.GetFocalPoint(), view_args["focalpoint"])
    assert_allclose(cam.GetDistance(), view_args["distance"])
    assert_allclose(cam.GetRoll(), previous_roll + view_args["roll"])
    del view_args

    with pytest.warns(DeprecationWarning, match='`view` is a dict'):
        brain.show_view(view=dict(azimuth=180., elevation=90.))

    # image and screenshot
    fname = op.join(str(tmp_path), 'test.png')
    assert not op.isfile(fname)
    brain.save_image(fname)
    assert op.isfile(fname)
    fp = np.array(
        brain._renderer.figure.plotter.renderer.ComputeVisiblePropBounds())
    fp = (fp[1::2] + fp[::2]) * 0.5
    azimuth, elevation = 180., 90.
    for view_args in (dict(azimuth=azimuth, elevation=elevation,
                           focalpoint='auto'),
                      dict(view='lateral', hemi='lh')):
        brain.show_view(**view_args)
        assert_allclose(brain._renderer.figure._azimuth, azimuth)
        assert_allclose(brain._renderer.figure._elevation, elevation)
        assert_allclose(cam.GetFocalPoint(), fp)
    del view_args
    img = brain.screenshot(mode='rgba')
    want_size = np.array([size[0] * pixel_ratio, size[1] * pixel_ratio, 4])
    assert_allclose(img.shape, want_size)
    brain.close()


@testing.requires_testing_data
@pytest.mark.skipif(os.getenv('CI_OS_NAME', '') == 'osx',
                    reason='Unreliable/segfault on macOS CI')
@pytest.mark.parametrize('hemi', ('lh', 'rh'))
def test_single_hemi(hemi, renderer_interactive_pyvistaqt, brain_gc):
    """Test single hemi support."""
    stc = read_source_estimate(fname_stc)
    idx, order = (0, 1) if hemi == 'lh' else (1, -1)
    stc = SourceEstimate(
        getattr(stc, f'{hemi}_data'), [stc.vertices[idx], []][::order],
        0, 1, 'sample')
    brain = stc.plot(
        subjects_dir=subjects_dir, hemi='both', size=300,
        cortex='0.5')  # single cortex string arg
    brain.close()

    # test skipping when len(vertices) == 0
    stc.vertices[1 - idx] = np.array([])
    brain = stc.plot(
        subjects_dir=subjects_dir, hemi=hemi, size=300)
    brain.close()


@testing.requires_testing_data
@pytest.mark.slowtest
def test_brain_save_movie(tmp_path, renderer, brain_gc):
    """Test saving a movie of a Brain instance."""
    if renderer._get_3d_backend() == "mayavi":
        pytest.skip('Save movie only supported on PyVista')
    from imageio_ffmpeg import count_frames_and_secs
    brain = _create_testing_brain(hemi='lh', time_viewer=False,
                                  cortex=['r', 'b'])  # custom binarized
    filename = str(op.join(tmp_path, "brain_test.mov"))
    for interactive_state in (False, True):
        # for coverage, we set interactivity
        if interactive_state:
            brain._renderer.plotter.enable()
        else:
            brain._renderer.plotter.disable()
        with pytest.raises(TypeError, match='unexpected keyword argument'):
            brain.save_movie(filename, time_dilation=1, tmin=1, tmax=1.1,
                             bad_name='blah')
        assert not op.isfile(filename)
        tmin = 1
        tmax = 5
        duration = np.floor(tmax - tmin)
        brain.save_movie(filename, time_dilation=1., tmin=tmin,
                         tmax=tmax, interpolation='nearest')
        assert op.isfile(filename)
        _, nsecs = count_frames_and_secs(filename)
        assert_allclose(duration, nsecs, atol=0.2)

        os.remove(filename)
    brain.close()


_TINY_SIZE = (350, 300)


def tiny(tmp_path):
    """Create a tiny fake brain."""
    # This is a minimal version of what we need for our viz-with-timeviewer
    # support currently
    subject = 'test'
    (tmp_path / subject).mkdir()
    subject_dir = tmp_path / subject
    (subject_dir / 'surf').mkdir()
    surf_dir = subject_dir / 'surf'
    rng = np.random.RandomState(0)
    rr = rng.randn(4, 3)
    tris = np.array([[0, 1, 2], [2, 1, 3]])
    curv = rng.randn(len(rr))
    with open(surf_dir / 'lh.curv', 'wb') as fid:
        fid.write(np.array([255, 255, 255], dtype=np.uint8))
        fid.write(np.array([len(rr), 0, 1], dtype='>i4'))
        fid.write(curv.astype('>f4'))
    write_surface(surf_dir / 'lh.white', rr, tris)
    write_surface(surf_dir / 'rh.white', rr, tris)  # needed for vertex tc
    vertices = [np.arange(len(rr)), []]
    data = rng.randn(len(rr), 10)
    stc = SourceEstimate(data, vertices, 0, 1, subject)
    brain = stc.plot(subjects_dir=tmp_path, hemi='lh', surface='white',
                     size=_TINY_SIZE)
    # in principle this should be sufficient:
    #
    # ratio = brain.mpl_canvas.canvas.window().devicePixelRatio()
    #
    # but in practice VTK can mess up sizes, so let's just calculate it.
    sz = brain.plotter.size()
    sz = (sz.width(), sz.height())
    sz_ren = brain.plotter.renderer.GetSize()
    ratio = np.median(np.array(sz_ren) / np.array(sz))
    return brain, ratio


@pytest.mark.filterwarnings('ignore:.*constrained_layout not applied.*:')
def test_brain_screenshot(renderer_interactive_pyvistaqt, tmp_path, brain_gc):
    """Test time viewer screenshot."""
    # XXX disable for sprint because it's too unreliable
    if sys.platform == 'darwin' and os.getenv('GITHUB_ACTIONS', '') == 'true':
        pytest.skip('Test is unreliable on GitHub Actions macOS')
    tiny_brain, ratio = tiny(tmp_path)
    img_nv = tiny_brain.screenshot(time_viewer=False)
    want = (_TINY_SIZE[1] * ratio, _TINY_SIZE[0] * ratio, 3)
    assert img_nv.shape == want
    img_v = tiny_brain.screenshot(time_viewer=True)
    assert img_v.shape[1:] == want[1:]
    assert_allclose(img_v.shape[0], want[0] * 4 / 3, atol=3)  # some slop
    tiny_brain.close()


def _assert_brain_range(brain, rng):
    __tracebackhide__ = True
    assert brain._cmap_range == rng, 'brain._cmap_range == rng'
    for hemi, layerer in brain._layered_meshes.items():
        for key, mesh in layerer._overlays.items():
            if key == 'curv':
                continue
            assert mesh._rng == rng, \
                f'_layered_meshes[{repr(hemi)}][{repr(key)}]._rng != {rng}'


@testing.requires_testing_data
@pytest.mark.slowtest
def test_brain_time_viewer(renderer_interactive_pyvistaqt, pixel_ratio,
                           brain_gc):
    """Test time viewer primitives."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        _create_testing_brain(hemi='lh', show_traces=-1.0)
    with pytest.raises(ValueError, match="got unknown keys"):
        _create_testing_brain(hemi='lh', surf='white', src='volume',
                              volume_options={'foo': 'bar'})
    brain = _create_testing_brain(
        hemi='both', show_traces=False,
        brain_kwargs=dict(silhouette=dict(decimate=0.95))
    )
    # test sub routines when show_traces=False
    brain._on_pick(None, None)
    brain._configure_vertex_time_course()
    brain._configure_label_time_course()
    brain.setup_time_viewer()  # for coverage
    brain.callbacks["time"](value=0)
    assert "renderer" not in brain.callbacks
    brain.callbacks["orientation"](
        value='lat',
        update_widget=True
    )
    brain.callbacks["orientation"](
        value='medial',
        update_widget=True
    )
    brain.callbacks["time"](
        value=0.0,
        time_as_index=False,
    )
    # Need to process events for old Qt
    brain.callbacks["smoothing"](value=1)
    _assert_brain_range(brain, [0.1, 0.3])
    from mne.utils import use_log_level
    print('\nCallback fmin\n')
    with use_log_level('debug'):
        brain.callbacks["fmin"](value=12.0)
    assert brain._data["fmin"] == 12.0
    brain.callbacks["fmax"](value=4.0)
    _assert_brain_range(brain, [4.0, 4.0])
    brain.callbacks["fmid"](value=6.0)
    _assert_brain_range(brain, [4.0, 6.0])
    brain.callbacks["fmid"](value=4.0)
    brain.callbacks["fplus"]()
    brain.callbacks["fminus"]()
    brain.callbacks["fmin"](value=12.0)
    brain.callbacks["fmid"](value=4.0)
    _assert_brain_range(brain, [4.0, 12.0])
    brain._shift_time(op=lambda x, y: x + y)
    brain._shift_time(op=lambda x, y: x - y)
    brain._rotate_azimuth(15)
    brain._rotate_elevation(15)
    brain.toggle_interface()
    brain.toggle_interface(value=False)
    brain.callbacks["playback_speed"](value=0.1)
    brain.toggle_playback()
    brain.toggle_playback(value=False)
    brain.apply_auto_scaling()
    brain.restore_user_scaling()
    brain.reset()

    assert brain.help_canvas is not None
    assert not brain.help_canvas.canvas.isVisible()
    brain.help()
    assert brain.help_canvas.canvas.isVisible()

    # screenshot
    # Need to turn the interface back on otherwise the window is too wide
    # (it keeps the window size and expands the 3D area when the interface
    # is toggled off)
    brain.toggle_interface(value=True)
    brain.show_view(azimuth=180., elevation=90.)
    img = brain.screenshot(mode='rgb')
    want_shape = np.array([300 * pixel_ratio, 300 * pixel_ratio, 3])
    assert_allclose(img.shape, want_shape)
    brain.close()


@testing.requires_testing_data
@pytest.mark.parametrize('hemi', [
    'lh',
    pytest.param('rh', marks=pytest.mark.slowtest),
    pytest.param('split', marks=pytest.mark.slowtest),
    pytest.param('both', marks=pytest.mark.slowtest),
])
@pytest.mark.parametrize('src', [
    'surface',
    pytest.param('vector', marks=pytest.mark.slowtest),
    pytest.param('volume', marks=pytest.mark.slowtest),
    pytest.param('mixed', marks=pytest.mark.slowtest),
])
@pytest.mark.slowtest
def test_brain_traces(renderer_interactive_pyvistaqt, hemi, src, tmp_path,
                      brain_gc):
    """Test brain traces."""
    hemi_str = list()
    if src in ('surface', 'vector', 'mixed'):
        hemi_str.extend([hemi] if hemi in ('lh', 'rh') else ['lh', 'rh'])
    if src in ('mixed', 'volume'):
        hemi_str.extend(['vol'])

    # label traces
    brain = _create_testing_brain(
        hemi=hemi, surf='white', src=src, show_traces='label',
        volume_options=None,  # for speed, don't upsample
        n_time=5, initial_time=0,
    )
    if src == 'surface':
        brain._data['src'] = None  # test src=None
    if src in ('surface', 'vector', 'mixed'):
        assert brain.show_traces
        assert brain.traces_mode == 'label'
        brain.widgets["extract_mode"].set_value('max')

        # test picking a cell at random
        rng = np.random.RandomState(0)
        for idx, current_hemi in enumerate(hemi_str):
            if current_hemi == 'vol':
                continue
            current_mesh = brain._layered_meshes[current_hemi]._polydata
            cell_id = rng.randint(0, current_mesh.n_cells)
            test_picker = TstVTKPicker(
                current_mesh, cell_id, current_hemi, brain)
            assert len(brain.picked_patches[current_hemi]) == 0
            brain._on_pick(test_picker, None)
            assert len(brain.picked_patches[current_hemi]) == 1
            for label_id in list(brain.picked_patches[current_hemi]):
                label = brain._annotation_labels[current_hemi][label_id]
                assert isinstance(label._line, Line2D)
            brain.widgets["extract_mode"].set_value('mean')
            brain.clear_glyphs()
            assert len(brain.picked_patches[current_hemi]) == 0
            brain._on_pick(test_picker, None)  # picked and added
            assert len(brain.picked_patches[current_hemi]) == 1
            brain._on_pick(test_picker, None)  # picked again so removed
            assert len(brain.picked_patches[current_hemi]) == 0
        # test switching from 'label' to 'vertex'
        brain.widgets["annotation"].set_value('None')
        brain.widgets["extract_mode"].set_value('max')
    else:  # volume
        assert "annotation" not in brain.widgets
        assert "extract_mode" not in brain.widgets
    brain.close()

    # test colormap
    if src != 'vector':
        brain = _create_testing_brain(
            hemi=hemi, surf='white', src=src, show_traces=0.5,
            initial_time=0,
            volume_options=None,  # for speed, don't upsample
            n_time=1 if src == 'mixed' else 5, diverging=True,
            add_data_kwargs=dict(colorbar_kwargs=dict(n_labels=3)),
        )
        # mne_analyze should be chosen
        ctab = brain._data['ctable']
        assert_array_equal(ctab[0], [0, 255, 255, 255])  # opaque cyan
        assert_array_equal(ctab[-1], [255, 255, 0, 255])  # opaque yellow
        assert_allclose(ctab[len(ctab) // 2], [128, 128, 128, 0], atol=3)
        brain.close()

    # vertex traces
    brain = _create_testing_brain(
        hemi=hemi, surf='white', src=src, show_traces=0.5,
        initial_time=0,
        volume_options=None,  # for speed, don't upsample
        n_time=1 if src == 'mixed' else 5,
        add_data_kwargs=dict(colorbar_kwargs=dict(n_labels=3)),
    )
    assert brain.show_traces
    assert brain.traces_mode == 'vertex'
    assert hasattr(brain, "picked_points")
    assert hasattr(brain, "_spheres")
    assert brain._scalar_bar.GetNumberOfLabels() == 3

    # add foci should work for volumes
    brain.add_foci([[0, 0, 0]], hemi='lh' if src == 'surface' else 'vol')

    # test points picked by default
    picked_points = brain.get_picked_points()
    spheres = brain._spheres
    for current_hemi in hemi_str:
        assert len(picked_points[current_hemi]) == 1
    n_spheres = len(hemi_str)
    n_actors = n_spheres
    if hemi == 'split' and src in ('mixed', 'volume'):
        n_spheres += 1
    assert len(spheres) == n_spheres

    # test that there are actually enough actors
    assert len(brain._actors['data']) == n_actors

    # test switching from 'vertex' to 'label'
    if src == 'surface':
        brain.widgets["annotation"].set_value('aparc')
        brain.widgets["annotation"].set_value('None')
    # test removing points
    brain.clear_glyphs()
    assert len(spheres) == 0
    for key in ('lh', 'rh', 'vol'):
        assert len(picked_points[key]) == 0

    # test picking a cell at random
    rng = np.random.RandomState(0)
    for idx, current_hemi in enumerate(hemi_str):
        assert len(spheres) == 0
        if current_hemi == 'vol':
            current_mesh = brain._data['vol']['grid']
            vertices = brain._data['vol']['vertices']
            values = _cell_data(current_mesh)['values'][vertices]
            cell_id = vertices[np.argmax(np.abs(values))]
        else:
            current_mesh = brain._layered_meshes[current_hemi]._polydata
            cell_id = rng.randint(0, current_mesh.n_cells)
        test_picker = TstVTKPicker(None, None, current_hemi, brain)
        assert brain._on_pick(test_picker, None) is None
        test_picker = TstVTKPicker(
            current_mesh, cell_id, current_hemi, brain)
        assert cell_id == test_picker.cell_id
        assert test_picker.point_id is None
        brain._on_pick(test_picker, None)
        brain._on_pick(test_picker, None)
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
            subject=brain._subject_id,
            subjects_dir=brain._subjects_dir
        )
        label = "{}:{} MNI: {}".format(
            hemi_prefix, str(vertex_id).ljust(6),
            ', '.join('%5.1f' % m for m in mni))

        assert line.get_label() == label

        # remove the sphere by clicking in its vicinity
        old_len = len(spheres)
        test_picker._actors = sum((s._actors for s in spheres), [])
        brain._on_pick(test_picker, None)
        assert len(spheres) < old_len

    screenshot = brain.screenshot()
    screenshot_all = brain.screenshot(time_viewer=True)
    assert screenshot.shape[0] < screenshot_all.shape[0]
    # and the scraper for it (will close the instance)
    # only test one condition to save time
    if not (hemi == 'rh' and src == 'surface' and
            check_version('sphinx_gallery')):
        brain.close()
        return
    fnames = [str(tmp_path / f'temp_{ii}.png') for ii in range(2)]
    block_vars = dict(image_path_iterator=iter(fnames),
                      example_globals=dict(brain=brain))
    block = ('code', """
something
# brain.save_movie(time_dilation=1, framerate=1,
#                  interpolation='linear', time_viewer=True)
#
""", 1)
    gallery_conf = dict(src_dir=str(tmp_path), compress_images=[])
    scraper = _BrainScraper()
    rst = scraper(block, block_vars, gallery_conf)
    assert brain.plotter is None  # closed
    gif_0 = fnames[0][:-3] + 'gif'
    for fname in (gif_0, fnames[1]):
        assert op.basename(fname) in rst
        assert op.isfile(fname)
        img = image.imread(fname)
        assert img.shape[1] == screenshot.shape[1]  # same width
        assert img.shape[0] > screenshot.shape[0]  # larger height
        assert img.shape[:2] == screenshot_all.shape[:2]


@testing.requires_testing_data
@pytest.mark.slowtest
def test_brain_linkviewer(renderer_interactive_pyvistaqt, brain_gc):
    """Test _LinkViewer primitives."""
    brain1 = _create_testing_brain(hemi='lh', show_traces=False)
    brain2 = _create_testing_brain(hemi='lh', show_traces='separate')
    brain1._times = brain1._times * 2
    with pytest.warns(RuntimeWarning, match='linking time'):
        link_viewer = _LinkViewer(
            [brain1, brain2],
            time=True,
            camera=False,
            colorbar=False,
            picking=False,
        )
    brain1.close()

    brain_data = _create_testing_brain(hemi='split', show_traces='vertex')
    link_viewer = _LinkViewer(
        [brain2, brain_data],
        time=True,
        camera=True,
        colorbar=True,
        picking=True,
    )
    link_viewer.leader.set_time_point(0)
    link_viewer.leader.mpl_canvas.time_func(0)
    link_viewer.leader.callbacks["fmin"](0)
    link_viewer.leader.callbacks["fmid"](0.5)
    link_viewer.leader.callbacks["fmax"](1)
    link_viewer.leader.set_playback_speed(0.1)
    link_viewer.leader.toggle_playback()
    brain2.close()
    brain_data.close()


def test_calculate_lut():
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


def _create_testing_brain(hemi, surf='inflated', src='surface',
                          size=300, n_time=5, diverging=False, **kwargs):
    assert src in ('surface', 'vector', 'mixed', 'volume')
    meth = 'plot'
    if src in ('surface', 'mixed'):
        sample_src = read_source_spaces(src_fname)
        klass = MixedSourceEstimate if src == 'mixed' else SourceEstimate
    if src == 'vector':
        fwd = read_forward_solution(fname_fwd)
        fwd = pick_types_forward(fwd, meg=True, eeg=False)
        evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0]
        noise_cov = read_cov(fname_cov)
        free = make_inverse_operator(
            evoked.info, fwd, noise_cov, loose=1.)
        stc = apply_inverse(evoked, free, pick_ori='vector')
        return stc.plot(
            subject=subject_id, hemi=hemi, size=size,
            subjects_dir=subjects_dir, colormap='auto',
            **kwargs)
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
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.zeros((n_verts * n_time))
    stc_size = stc_data.size
    stc_data[(rng.rand(stc_size // 20) * stc_size).astype(int)] = \
        rng.rand(stc_data.size // 20)
    stc_data.shape = (n_verts, n_time)
    if diverging:
        stc_data -= 0.5
    stc = klass(stc_data, vertices, 1, 1)

    clim = dict(kind='value', lims=[0.1, 0.2, 0.3])
    if diverging:
        clim['pos_lims'] = clim.pop('lims')

    brain_data = getattr(stc, meth)(
        subject=subject_id, hemi=hemi, surface=surf, size=size,
        subjects_dir=subjects_dir, colormap='auto',
        clim=clim, src=sample_src,
        **kwargs)
    return brain_data
