# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: Simplified BSD

import os.path as op
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from mne import (make_field_map, pick_channels_evoked, read_evokeds,
                 read_trans, read_dipole, SourceEstimate, VectorSourceEstimate,
                 VolSourceEstimate, make_sphere_model, use_coil_def,
                 setup_volume_source_space, read_forward_solution,
                 VolVectorSourceEstimate, convert_forward_solution,
                 compute_source_morph, MixedSourceEstimate)
from mne.io import (read_raw_ctf, read_raw_bti, read_raw_kit, read_info,
                    read_raw_nirx)
from mne.io._digitization import write_dig
from mne.io.pick import pick_info
from mne.io.constants import FIFF
from mne.viz import (plot_sparse_source_estimates, plot_source_estimates,
                     snapshot_brain_montage, plot_head_positions,
                     plot_alignment, plot_volume_source_estimates,
                     plot_sensors_connectivity, plot_brain_colorbar,
                     link_brains, mne_analyze_colormap)
from mne.viz._3d import _process_clim, _linearize_map, _get_map_ticks
from mne.viz.utils import _fake_click
from mne.utils import (requires_pysurfer, run_tests_if_main,
                       requires_nibabel, requires_dipy,
                       traits_test, requires_version, catch_logging,
                       run_subprocess, modified_env)
from mne.datasets import testing
from mne.source_space import read_source_spaces
from mne.bem import read_bem_solution, read_bem_surfaces


data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
trans_fname = op.join(data_dir, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
src_fname = op.join(data_dir, 'subjects', 'sample', 'bem',
                    'sample-oct-6-src.fif')
dip_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_trunc_set1.dip')
ctf_fname = op.join(data_dir, 'CTF', 'testdata_ctf.ds')
nirx_fname = op.join(data_dir, 'NIRx', 'nirx_15_2_recording_w_short')

io_dir = op.join(op.abspath(op.dirname(__file__)), '..', '..', 'io')
base_dir = op.join(io_dir, 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')

fwd_fname = op.join(data_dir, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-vol-7-fwd.fif')
fwd_fname2 = op.join(data_dir, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')

base_dir = op.join(io_dir, 'bti', 'tests', 'data')
pdf_fname = op.join(base_dir, 'test_pdf_linux')
config_fname = op.join(base_dir, 'test_config_linux')
hs_fname = op.join(base_dir, 'test_hs_linux')
sqd_fname = op.join(io_dir, 'kit', 'tests', 'data', 'test.sqd')

coil_3d = """# custom cube coil def
1   9999    1   8  3e-03  0.000e+00     "QuSpin ZFOPM 3mm cube"
  0.1250 -0.750e-03 -0.750e-03 -0.750e-03  0.000  0.000  1.000
  0.1250 -0.750e-03  0.750e-03 -0.750e-03  0.000  0.000  1.000
  0.1250  0.750e-03 -0.750e-03 -0.750e-03  0.000  0.000  1.000
  0.1250  0.750e-03  0.750e-03 -0.750e-03  0.000  0.000  1.000
  0.1250 -0.750e-03 -0.750e-03  0.750e-03  0.000  0.000  1.000
  0.1250 -0.750e-03  0.750e-03  0.750e-03  0.000  0.000  1.000
  0.1250  0.750e-03 -0.750e-03  0.750e-03  0.000  0.000  1.000
  0.1250  0.750e-03  0.750e-03  0.750e-03  0.000  0.000  1.000
"""


def test_plot_head_positions():
    """Test plotting of head positions."""
    info = read_info(evoked_fname)
    pos = np.random.RandomState(0).randn(4, 10)
    pos[:, 0] = np.arange(len(pos))
    destination = (0., 0., 0.04)
    with pytest.warns(None):  # old MPL will cause a warning
        plot_head_positions(pos)
        plot_head_positions(pos, mode='field', info=info,
                            destination=destination)
        plot_head_positions([pos, pos])  # list support
        pytest.raises(ValueError, plot_head_positions, ['pos'])
        pytest.raises(ValueError, plot_head_positions, pos[:, :9])
    pytest.raises(ValueError, plot_head_positions, pos, 'foo')
    with pytest.raises(ValueError, match='shape'):
        plot_head_positions(pos, axes=1.)
    plt.close('all')


@testing.requires_testing_data
@requires_pysurfer
@traits_test
def test_plot_sparse_source_estimates(renderer_interactive):
    """Test plotting of (sparse) source estimates."""
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

    colormap = 'mne_analyze'
    plot_source_estimates(stc, 'sample', colormap=colormap,
                          background=(1, 1, 0),
                          subjects_dir=subjects_dir, colorbar=True,
                          clim='auto')
    pytest.raises(TypeError, plot_source_estimates, stc, 'sample',
                  figure='foo', hemi='both', clim='auto',
                  subjects_dir=subjects_dir)

    # now do sparse version
    vertices = sample_src[0]['vertno']
    inds = [111, 333]
    stc_data = np.zeros((len(inds), n_time))
    stc_data[0, 1] = 1.
    stc_data[1, 4] = 2.
    vertices = [vertices[inds], np.empty(0, dtype=np.int)]
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    surf = plot_sparse_source_estimates(sample_src, stc, bgcolor=(1, 1, 1),
                                        opacity=0.5, high_resolution=False)
    if renderer_interactive.get_3d_backend() == 'mayavi':
        import mayavi  # noqa: F401 analysis:ignore
        assert isinstance(surf, mayavi.modules.surface.Surface)


@testing.requires_testing_data
@traits_test
def test_plot_evoked_field(renderer):
    """Test plotting evoked field."""
    evoked = read_evokeds(evoked_fname, condition='Left Auditory',
                          baseline=(-0.2, 0.0))
    evoked = pick_channels_evoked(evoked, evoked.ch_names[::10])  # speed
    for t in ['meg', None]:
        with pytest.warns(RuntimeWarning, match='projection'):
            maps = make_field_map(evoked, trans_fname, subject='sample',
                                  subjects_dir=subjects_dir, n_jobs=1,
                                  ch_type=t)
        fig = evoked.plot_field(maps, time=0.1)
        if renderer.get_3d_backend() == 'mayavi':
            import mayavi  # noqa: F401 analysis:ignore
            assert isinstance(fig, mayavi.core.scene.Scene)


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
@traits_test
def test_plot_alignment(tmpdir, renderer):
    """Test plotting of -trans.fif files and MEG sensor layouts."""
    # generate fiducials file for testing
    tempdir = str(tmpdir)
    fiducials_path = op.join(tempdir, 'fiducials.fif')
    fid = [{'coord_frame': 5, 'ident': 1, 'kind': 1,
            'r': [-0.08061612, -0.02908875, -0.04131077]},
           {'coord_frame': 5, 'ident': 2, 'kind': 1,
            'r': [0.00146763, 0.08506715, -0.03483611]},
           {'coord_frame': 5, 'ident': 3, 'kind': 1,
            'r': [0.08436285, -0.02850276, -0.04127743]}]
    write_dig(fiducials_path, fid, 5)

    renderer.backend._close_all()
    evoked = read_evokeds(evoked_fname)[0]
    sample_src = read_source_spaces(src_fname)
    bti = read_raw_bti(pdf_fname, config_fname, hs_fname, convert=True,
                       preload=False).info
    infos = dict(
        Neuromag=evoked.info,
        CTF=read_raw_ctf(ctf_fname).info,
        BTi=bti,
        KIT=read_raw_kit(sqd_fname).info,
    )
    for system, info in infos.items():
        meg = ['helmet', 'sensors']
        if system == 'KIT':
            meg.append('ref')
        fig = plot_alignment(info, read_trans(trans_fname), subject='sample',
                             subjects_dir=subjects_dir, meg=meg)
        rend = renderer.backend._Renderer(fig=fig)
        rend.close()
    # KIT ref sensor coil def is defined
    renderer.backend._close_all()
    info = infos['Neuromag']
    pytest.raises(TypeError, plot_alignment, 'foo', trans_fname,
                  subject='sample', subjects_dir=subjects_dir)
    pytest.raises(OSError, plot_alignment, info, trans_fname,
                  subject='sample', subjects_dir=subjects_dir, src='foo')
    pytest.raises(ValueError, plot_alignment, info, trans_fname,
                  subject='fsaverage', subjects_dir=subjects_dir,
                  src=sample_src)
    sample_src.plot(subjects_dir=subjects_dir, head=True, skull=True,
                    brain='white')
    renderer.backend._close_all()
    # no-head version
    renderer.backend._close_all()
    # all coord frames
    pytest.raises(ValueError, plot_alignment, info)
    plot_alignment(info, surfaces=[])
    for coord_frame in ('meg', 'head', 'mri'):
        fig = plot_alignment(info, meg=['helmet', 'sensors'], dig=True,
                             coord_frame=coord_frame, trans=Path(trans_fname),
                             subject='sample', mri_fiducials=fiducials_path,
                             subjects_dir=subjects_dir, src=src_fname)
    renderer.backend._close_all()
    # EEG only with strange options
    evoked_eeg_ecog_seeg = evoked.copy().pick_types(meg=False, eeg=True)
    evoked_eeg_ecog_seeg.info['projs'] = []  # "remove" avg proj
    evoked_eeg_ecog_seeg.set_channel_types({'EEG 001': 'ecog',
                                            'EEG 002': 'seeg'})
    with pytest.warns(RuntimeWarning, match='Cannot plot MEG'):
        plot_alignment(evoked_eeg_ecog_seeg.info, subject='sample',
                       trans=trans_fname, subjects_dir=subjects_dir,
                       surfaces=['white', 'outer_skin', 'outer_skull'],
                       meg=['helmet', 'sensors'],
                       eeg=['original', 'projected'], ecog=True, seeg=True)
    renderer.backend._close_all()

    sphere = make_sphere_model(info=evoked.info, r0='auto', head_radius='auto')
    bem_sol = read_bem_solution(op.join(subjects_dir, 'sample', 'bem',
                                        'sample-1280-1280-1280-bem-sol.fif'))
    bem_surfs = read_bem_surfaces(op.join(subjects_dir, 'sample', 'bem',
                                          'sample-1280-1280-1280-bem.fif'))
    sample_src[0]['coord_frame'] = 4  # hack for coverage
    plot_alignment(info, subject='sample', eeg='projected',
                   meg='helmet', bem=sphere, dig=True,
                   surfaces=['brain', 'inner_skull', 'outer_skull',
                             'outer_skin'])
    plot_alignment(info, trans_fname, subject='sample', meg='helmet',
                   subjects_dir=subjects_dir, eeg='projected', bem=sphere,
                   surfaces=['head', 'brain'], src=sample_src)
    assert all(surf['coord_frame'] == FIFF.FIFFV_COORD_MRI
               for surf in bem_sol['surfs'])
    plot_alignment(info, trans_fname, subject='sample', meg=[],
                   subjects_dir=subjects_dir, bem=bem_sol, eeg=True,
                   surfaces=['head', 'inflated', 'outer_skull', 'inner_skull'])
    assert all(surf['coord_frame'] == FIFF.FIFFV_COORD_MRI
               for surf in bem_sol['surfs'])
    plot_alignment(info, trans_fname, subject='sample',
                   meg=True, subjects_dir=subjects_dir,
                   surfaces=['head', 'inner_skull'], bem=bem_surfs)
    # single-layer BEM can still plot head surface
    assert bem_surfs[-1]['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN
    bem_sol_homog = read_bem_solution(op.join(subjects_dir, 'sample', 'bem',
                                              'sample-1280-bem-sol.fif'))
    for use_bem in (bem_surfs[-1:], bem_sol_homog):
        with catch_logging() as log:
            plot_alignment(info, trans_fname, subject='sample',
                           meg=True, subjects_dir=subjects_dir,
                           surfaces=['head', 'inner_skull'], bem=use_bem,
                           verbose=True)
        log = log.getvalue()
        assert 'not find the surface for head in the provided BEM model' in log
    # sphere model
    sphere = make_sphere_model('auto', 'auto', evoked.info)
    src = setup_volume_source_space(sphere=sphere)
    plot_alignment(info, eeg='projected', meg='helmet', bem=sphere,
                   src=src, dig=True, surfaces=['brain', 'inner_skull',
                                                'outer_skull', 'outer_skin'])
    sphere = make_sphere_model('auto', None, evoked.info)  # one layer
    # no info is permitted
    fig = plot_alignment(trans=trans_fname, subject='sample', meg=False,
                         coord_frame='mri', subjects_dir=subjects_dir,
                         surfaces=['brain'], bem=sphere, show_axes=True)
    renderer.backend._close_all()
    if renderer.get_3d_backend() == 'mayavi':
        import mayavi  # noqa: F401 analysis:ignore
        assert isinstance(fig, mayavi.core.scene.Scene)

    # 3D coil with no defined draw (ConvexHull)
    info_cube = pick_info(info, [0])
    info['dig'] = None
    info_cube['chs'][0]['coil_type'] = 9999
    with pytest.raises(RuntimeError, match='coil definition not found'):
        plot_alignment(info_cube, meg='sensors', surfaces=())
    coil_def_fname = op.join(tempdir, 'temp')
    with open(coil_def_fname, 'w') as fid:
        fid.write(coil_3d)
    with use_coil_def(coil_def_fname):
        plot_alignment(info_cube, meg='sensors', surfaces=(), dig=True)

    # one layer bem with skull surfaces:
    with pytest.raises(ValueError, match='sphere conductor model must have'):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir,
                       surfaces=['brain', 'head', 'inner_skull'], bem=sphere)
    # wrong eeg value:
    with pytest.raises(ValueError, match='Invalid value for the .eeg'):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir, eeg='foo')
    # wrong meg value:
    with pytest.raises(ValueError, match='Invalid value for the .meg'):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir, meg='bar')
    # multiple brain surfaces:
    with pytest.raises(ValueError, match='Only one brain surface can be plot'):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir,
                       surfaces=['white', 'pial'])
    with pytest.raises(TypeError, match='all entries in surfaces must be'):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir,
                       surfaces=[1])
    with pytest.raises(ValueError, match='Unknown surface type'):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir,
                       surfaces=['foo'])
    fwd_fname = op.join(data_dir, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
    fwd = read_forward_solution(fwd_fname)
    plot_alignment(subject='sample', subjects_dir=subjects_dir,
                   trans=trans_fname, fwd=fwd,
                   surfaces='white', coord_frame='head')
    fwd = convert_forward_solution(fwd, force_fixed=True)
    plot_alignment(subject='sample', subjects_dir=subjects_dir,
                   trans=trans_fname, fwd=fwd,
                   surfaces='white', coord_frame='head')

    # fNIRS
    info = read_raw_nirx(nirx_fname).info
    with catch_logging() as log:
        plot_alignment(info, subject='fsaverage', surfaces=(), verbose=True)
    log = log.getvalue()
    assert '26 fnirs pairs' in log

    with catch_logging() as log:
        plot_alignment(info, subject='fsaverage', surfaces=(), verbose=True,
                       fnirs='channels')
    log = log.getvalue()
    assert '26 fnirs locations' in log

    with catch_logging() as log:
        plot_alignment(info, subject='fsaverage', surfaces=(), verbose=True,
                       fnirs='pairs')
    log = log.getvalue()
    assert '26 fnirs pairs' in log

    with catch_logging() as log:
        plot_alignment(info, subject='fsaverage', surfaces=(), verbose=True,
                       fnirs=['channels', 'pairs'])
    log = log.getvalue()
    assert '26 fnirs pairs' in log
    assert '26 fnirs locations' in log

    renderer.backend._close_all()


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
@requires_pysurfer
@traits_test
def test_process_clim_plot(renderer_interactive):
    """Test functionality for determining control points with stc.plot."""
    sample_src = read_source_spaces(src_fname)
    kwargs = dict(subjects_dir=subjects_dir, smoothing_steps=1)

    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.random.RandomState(0).rand((n_verts * n_time))
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1, 'sample')

    # Test for simple use cases
    stc.plot(**kwargs)
    stc.plot(clim=dict(pos_lims=(10, 50, 90)), **kwargs)
    stc.plot(colormap='hot', clim='auto', **kwargs)
    stc.plot(colormap='mne', clim='auto', **kwargs)
    stc.plot(clim=dict(kind='value', lims=(10, 50, 90)), figure=99, **kwargs)
    pytest.raises(TypeError, stc.plot, clim='auto', figure=[0], **kwargs)

    # Test for correct clim values
    with pytest.raises(ValueError, match='monotonically'):
        stc.plot(clim=dict(kind='value', pos_lims=[0, 1, 0]), **kwargs)
    with pytest.raises(ValueError, match=r'.*must be \(3,\)'):
        stc.plot(colormap='mne', clim=dict(pos_lims=(5, 10, 15, 20)), **kwargs)
    with pytest.raises(ValueError, match="'value', 'values' and 'percent'"):
        stc.plot(clim=dict(pos_lims=(5, 10, 15), kind='foo'), **kwargs)
    with pytest.raises(ValueError, match='must be "auto" or dict'):
        stc.plot(colormap='mne', clim='foo', **kwargs)
    with pytest.raises(TypeError, match='must be an instance of'):
        plot_source_estimates('foo', clim='auto', **kwargs)
    with pytest.raises(ValueError, match='hemi'):
        stc.plot(hemi='foo', clim='auto', **kwargs)
    with pytest.raises(ValueError, match='Exactly one'):
        stc.plot(clim=dict(lims=[0, 1, 2], pos_lims=[0, 1, 2], kind='value'),
                 **kwargs)

    # Test handling of degenerate data: thresholded maps
    stc._data.fill(0.)
    with pytest.warns(RuntimeWarning, match='All data were zero'):
        plot_source_estimates(stc, **kwargs)


def _assert_mapdata_equal(a, b):
    __tracebackhide__ = True
    assert set(a.keys()) == {'clim', 'colormap', 'transparent'}
    assert a.keys() == b.keys()
    assert a['transparent'] == b['transparent'], 'transparent'
    aa, bb = a['clim'], b['clim']
    assert aa.keys() == bb.keys(), 'clim keys'
    assert aa['kind'] == bb['kind'] == 'value'
    key = 'pos_lims' if 'pos_lims' in aa else 'lims'
    assert_array_equal(aa[key], bb[key], err_msg=key)
    assert isinstance(a['colormap'], Colormap), 'Colormap'
    assert isinstance(b['colormap'], Colormap), 'Colormap'
    assert a['colormap'].name == b['colormap'].name


def test_process_clim_round_trip():
    """Test basic input-output support."""
    # With some negative data
    out = _process_clim('auto', 'auto', True, -1.)
    want = dict(
        colormap=mne_analyze_colormap([0, 0.5, 1], 'matplotlib'),
        clim=dict(kind='value', pos_lims=[1, 1, 1]),
        transparent=True,)
    _assert_mapdata_equal(out, want)
    out2 = _process_clim(**out)
    _assert_mapdata_equal(out, out2)
    _linearize_map(out)  # smoke test
    ticks = _get_map_ticks(out)
    assert_allclose(ticks, [-1, 0, 1])

    # With some positive data
    out = _process_clim('auto', 'auto', True, 1.)
    want = dict(
        colormap=plt.get_cmap('hot'),
        clim=dict(kind='value', lims=[1, 1, 1]),
        transparent=True,)
    _assert_mapdata_equal(out, want)
    out2 = _process_clim(**out)
    _assert_mapdata_equal(out, out2)
    _linearize_map(out)
    ticks = _get_map_ticks(out)
    assert_allclose(ticks, [1])

    # With some actual inputs
    clim = dict(kind='value', pos_lims=[0, 0.5, 1])
    out = _process_clim(clim, 'auto', True)
    want = dict(
        colormap=mne_analyze_colormap([0, 0.5, 1], 'matplotlib'),
        clim=clim, transparent=True)
    _assert_mapdata_equal(out, want)
    _linearize_map(out)
    ticks = _get_map_ticks(out)
    assert_allclose(ticks, [-1, -0.5, 0, 0.5, 1])

    clim = dict(kind='value', pos_lims=[0.25, 0.5, 1])
    out = _process_clim(clim, 'auto', True)
    want = dict(
        colormap=mne_analyze_colormap([0, 0.5, 1], 'matplotlib'),
        clim=clim, transparent=True)
    _assert_mapdata_equal(out, want)
    _linearize_map(out)
    ticks = _get_map_ticks(out)
    assert_allclose(ticks, [-1, -0.5, -0.25, 0, 0.25, 0.5, 1])


@testing.requires_testing_data
@requires_nibabel()
def test_stc_mpl():
    """Test plotting source estimates with matplotlib."""
    sample_src = read_source_spaces(src_fname)

    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.ones((n_verts * n_time))
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1, 'sample')
    with pytest.warns(RuntimeWarning, match='not included'):
        stc.plot(subjects_dir=subjects_dir, time_unit='s', views='ven',
                 hemi='rh', smoothing_steps=2, subject='sample',
                 backend='matplotlib', spacing='oct1', initial_time=0.001,
                 colormap='Reds')
        fig = stc.plot(subjects_dir=subjects_dir, time_unit='ms', views='dor',
                       hemi='lh', smoothing_steps=2, subject='sample',
                       backend='matplotlib', spacing='ico2', time_viewer=True,
                       colormap='mne')
        time_viewer = fig.time_viewer
        _fake_click(time_viewer, time_viewer.axes[0], (0.5, 0.5))  # change t
        time_viewer.canvas.key_press_event('ctrl+right')
        time_viewer.canvas.key_press_event('left')
    pytest.raises(ValueError, stc.plot, subjects_dir=subjects_dir,
                  hemi='both', subject='sample', backend='matplotlib')
    pytest.raises(ValueError, stc.plot, subjects_dir=subjects_dir,
                  time_unit='ss', subject='sample', backend='matplotlib')
    plt.close('all')


@pytest.mark.timeout(60)  # can sometimes take > 60 sec
@testing.requires_testing_data
@requires_nibabel()
@pytest.mark.parametrize('coord_frame, idx, show_all, title',
                         [('head', 'gof', True, 'Test'),
                          ('mri', 'amplitude', False, None)])
def test_plot_dipole_mri_orthoview(coord_frame, idx, show_all, title):
    """Test mpl dipole plotting."""
    dipoles = read_dipole(dip_fname)
    trans = read_trans(trans_fname)
    fig = dipoles.plot_locations(trans=trans, subject='sample',
                                 subjects_dir=subjects_dir,
                                 coord_frame=coord_frame, idx=idx,
                                 show_all=show_all, title=title,
                                 mode='orthoview')
    fig.canvas.scroll_event(0.5, 0.5, 1)  # scroll up
    fig.canvas.scroll_event(0.5, 0.5, -1)  # scroll down
    fig.canvas.key_press_event('up')
    fig.canvas.key_press_event('down')
    fig.canvas.key_press_event('a')  # some other key
    ax = plt.subplot(111)
    pytest.raises(TypeError, dipoles.plot_locations, trans, 'sample',
                  subjects_dir, ax=ax)
    plt.close('all')


@testing.requires_testing_data
def test_plot_dipole_orientations(renderer):
    """Test dipole plotting in 3d."""
    dipoles = read_dipole(dip_fname)
    trans = read_trans(trans_fname)
    for coord_frame, mode in zip(['head', 'mri'],
                                 ['arrow', 'sphere']):
        dipoles.plot_locations(trans=trans, subject='sample',
                               subjects_dir=subjects_dir,
                               mode=mode, coord_frame=coord_frame)
    renderer.backend._close_all()


@testing.requires_testing_data
@traits_test
def test_snapshot_brain_montage(renderer):
    """Test snapshot brain montage."""
    info = read_info(evoked_fname)
    fig = plot_alignment(
        info, trans=None, subject='sample', subjects_dir=subjects_dir)

    xyz = np.vstack([ich['loc'][:3] for ich in info['chs']])
    ch_names = [ich['ch_name'] for ich in info['chs']]
    xyz_dict = dict(zip(ch_names, xyz))
    xyz_dict[info['chs'][0]['ch_name']] = [1, 2]  # Set one ch to only 2 vals

    # Make sure wrong types are checked
    pytest.raises(TypeError, snapshot_brain_montage, fig, xyz)

    # All chs must have 3 position values
    pytest.raises(ValueError, snapshot_brain_montage, fig, xyz_dict)

    # Make sure we raise error if the figure has no scene
    pytest.raises(ValueError, snapshot_brain_montage, None, info)


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
@requires_dipy()
@requires_nibabel()
@requires_version('nilearn', '0.4')
@pytest.mark.parametrize('mode, stype, init_t, want_t, init_p, want_p', [
    ('glass_brain', 's', None, 2, None, (-30.9, 18.4, 56.7)),
    ('stat_map', 'vec', 1, 1, None, (15.7, 16.0, -6.3)),
    ('glass_brain', 'vec', None, 1, (10, -10, 20), (6.6, -9.0, 19.9)),
    ('stat_map', 's', 1, 1, (-10, 5, 10), (-12.3, 2.0, 7.7))])
def test_plot_volume_source_estimates(mode, stype, init_t, want_t,
                                      init_p, want_p):
    """Test interactive plotting of volume source estimates."""
    forward = read_forward_solution(fwd_fname)
    sample_src = forward['src']
    if init_p is not None:
        init_p = np.array(init_p) / 1000.

    vertices = [s['vertno'] for s in sample_src]
    n_verts = sum(len(v) for v in vertices)
    n_time = 2
    data = np.random.RandomState(0).rand(n_verts, n_time)

    if stype == 'vec':
        stc = VolVectorSourceEstimate(
            np.tile(data[:, np.newaxis], (1, 3, 1)), vertices, 1, 1)
    else:
        assert stype == 's'
        stc = VolSourceEstimate(data, vertices, 1, 1)
    with pytest.warns(None):  # sometimes get scalars/index warning
        with catch_logging() as log:
            fig = stc.plot(
                sample_src, subject='sample', subjects_dir=subjects_dir,
                mode=mode, initial_time=init_t, initial_pos=init_p,
                verbose=True)
    log = log.getvalue()
    want_str = 't = %0.3f s' % want_t
    assert want_str in log, (want_str, init_t)
    want_str = '(%0.1f, %0.1f, %0.1f) mm' % want_p
    assert want_str in log, (want_str, init_p)
    for ax_idx in [0, 2, 3, 4]:
        _fake_click(fig, fig.axes[ax_idx], (0.3, 0.5))
    fig.canvas.key_press_event('left')
    fig.canvas.key_press_event('shift+right')


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
@requires_dipy()
@requires_nibabel()
@requires_version('nilearn', '0.4')
def test_plot_volume_source_estimates_morph():
    """Test interactive plotting of volume source estimates with morph."""
    forward = read_forward_solution(fwd_fname)
    sample_src = forward['src']
    vertices = [s['vertno'] for s in sample_src]
    n_verts = sum(len(v) for v in vertices)
    n_time = 2
    data = np.random.RandomState(0).rand(n_verts, n_time)
    stc = VolSourceEstimate(data, vertices, 1, 1)
    sample_src[0]['subject_his_id'] = 'sample'  # old src
    morph = compute_source_morph(sample_src, 'sample', 'fsaverage', zooms=5,
                                 subjects_dir=subjects_dir)
    initial_pos = (-0.05, -0.01, -0.006)
    with pytest.warns(None):  # sometimes get scalars/index warning
        with catch_logging() as log:
            stc.plot(morph, subjects_dir=subjects_dir, mode='glass_brain',
                     initial_pos=initial_pos, verbose=True)
    log = log.getvalue()
    assert 't = 1.000 s' in log
    assert '(-52.0, -8.0, -7.0) mm' in log

    with pytest.raises(ValueError, match='Allowed values are'):
        stc.plot(sample_src, 'sample', subjects_dir, mode='abcd')
    vertices.append([])
    surface_stc = SourceEstimate(data, vertices, 1, 1)
    with pytest.raises(TypeError, match='an instance of VolSourceEstimate'):
        plot_volume_source_estimates(surface_stc, sample_src, 'sample',
                                     subjects_dir)
    with pytest.raises(ValueError, match='Negative colormap limits'):
        stc.plot(sample_src, 'sample', subjects_dir,
                 clim=dict(lims=[-1, 2, 3], kind='value'))


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
@requires_pysurfer
@traits_test
def test_plot_vector_source_estimates(renderer_interactive):
    """Test plotting of vector source estimates."""
    sample_src = read_source_spaces(src_fname)

    vertices = [s['vertno'] for s in sample_src]
    n_verts = sum(len(v) for v in vertices)
    n_time = 5
    data = np.random.RandomState(0).rand(n_verts, 3, n_time)
    stc = VectorSourceEstimate(data, vertices, 1, 1)

    brain = stc.plot('sample', subjects_dir=subjects_dir, hemi='both',
                     smoothing_steps=1, verbose='error')
    brain.close()
    del brain

    with pytest.raises(ValueError, match='use "pos_lims"'):
        stc.plot('sample', subjects_dir=subjects_dir,
                 clim=dict(pos_lims=[1, 2, 3]))


@testing.requires_testing_data
def test_plot_sensors_connectivity(renderer):
    """Test plotting of sensors connectivity."""
    from mne import io, pick_types

    data_path = data_dir
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    raw = io.read_raw_fif(raw_fname)
    picks = pick_types(raw.info, meg='grad', eeg=False, stim=False,
                       eog=True, exclude='bads')
    n_channels = len(picks)
    con = np.random.RandomState(42).randn(n_channels, n_channels)
    info = raw.info
    with pytest.raises(TypeError):
        plot_sensors_connectivity(info='foo', con=con,
                                  picks=picks)
    with pytest.raises(ValueError):
        plot_sensors_connectivity(info=info, con=con[::2, ::2],
                                  picks=picks)

    plot_sensors_connectivity(info=info, con=con, picks=picks)


@pytest.mark.parametrize('orientation', ('horizontal', 'vertical'))
@pytest.mark.parametrize('diverging', (True, False))
@pytest.mark.parametrize('lims', ([0.5, 1, 10], [0, 1, 10]))
def test_brain_colorbar(orientation, diverging, lims):
    """Test brain colorbar plotting."""
    _, ax = plt.subplots()
    clim = dict(kind='value')
    if diverging:
        clim['pos_lims'] = lims
    else:
        clim['lims'] = lims
    plot_brain_colorbar(ax, clim, orientation=orientation)
    if orientation == 'vertical':
        have, empty = ax.get_yticklabels, ax.get_xticklabels
    else:
        have, empty = ax.get_xticklabels, ax.get_yticklabels
    if diverging:
        if lims[0] == 0:
            ticks = list(-np.array(lims[1:][::-1])) + lims
        else:
            ticks = list(-np.array(lims[::-1])) + [0] + lims
    else:
        ticks = lims
    plt.draw()
    # old mpl always spans 0->1 for the actual ticks, so we need to
    # look at the labels
    assert_array_equal(
        [float(h.get_text().replace('−', '-')) for h in have()], ticks)
    assert_array_equal(empty(), [])
    plt.close('all')


@pytest.mark.slowtest  # slow-ish on Travis OSX
@requires_pysurfer
@testing.requires_testing_data
@traits_test
def test_mixed_sources_plot_surface(renderer_interactive):
    """Test plot_surface() for  mixed source space."""
    src = read_source_spaces(fwd_fname2)
    N = np.sum([s['nuse'] for s in src])  # number of sources

    T = 2  # number of time points
    S = 3  # number of source spaces

    rng = np.random.RandomState(0)
    data = rng.randn(N, T)
    vertno = S * [np.arange(N // S)]

    stc = MixedSourceEstimate(data, vertno, 0, 1)

    stc.surface().plot(views='lat', hemi='split',
                       subject='fsaverage', subjects_dir=subjects_dir,
                       colorbar=False)


@testing.requires_testing_data
@traits_test
def test_link_brains(renderer_interactive):
    """Test plotting linked brains."""
    sample_src = read_source_spaces(src_fname)
    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.zeros((n_verts * n_time))
    stc_size = stc_data.size
    stc_data[(np.random.rand(stc_size // 20) * stc_size).astype(int)] = \
        np.random.RandomState(0).rand(stc_data.size // 20)
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1)

    colormap = 'mne_analyze'
    brain = plot_source_estimates(
        stc, 'sample', colormap=colormap,
        background=(1, 1, 0),
        subjects_dir=subjects_dir, colorbar=True,
        clim='auto'
    )
    if renderer_interactive.get_3d_backend() != 'pyvista':
        with pytest.raises(NotImplementedError, match='backend is pyvista'):
            link_brains(brain)
    else:
        with pytest.raises(ValueError, match='is empty'):
            link_brains([])
        with pytest.raises(TypeError, match='type is Brain'):
            link_brains('foo')
        link_brains(brain)


def test_renderer(renderer):
    """Test that renderers are available on demand."""
    backend = renderer.get_3d_backend()
    cmd = [sys.executable, '-uc',
           'import mne; mne.viz.create_3d_figure((800, 600)); '
           'backend = mne.viz.get_3d_backend(); '
           'assert backend == %r, backend' % (backend,)]
    with modified_env(MNE_3D_BACKEND=backend):
        run_subprocess(cmd)


run_tests_if_main()
