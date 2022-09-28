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

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from mne import (make_field_map, pick_channels_evoked, read_evokeds,
                 read_trans, read_dipole, SourceEstimate,
                 make_sphere_model, use_coil_def, pick_types,
                 setup_volume_source_space, read_forward_solution,
                 convert_forward_solution, MixedSourceEstimate)
from mne.source_estimate import _BaseVolSourceEstimate
from mne.io import (read_raw_ctf, read_raw_bti, read_raw_kit, read_info,
                    read_raw_nirx)
from mne.io._digitization import write_dig
from mne.io.pick import pick_info
from mne.io.constants import FIFF
from mne.minimum_norm import apply_inverse
from mne.viz import (plot_sparse_source_estimates, plot_source_estimates,
                     snapshot_brain_montage, plot_head_positions,
                     plot_alignment, Figure3D,
                     plot_brain_colorbar, link_brains, mne_analyze_colormap)
from mne.viz._3d import _process_clim, _linearize_map, _get_map_ticks
from mne.viz.utils import _fake_click, _fake_keypress, _fake_scroll, _get_cmap
from mne.utils import requires_nibabel, catch_logging, _record_warnings
from mne.datasets import testing
from mne.source_space import read_source_spaces
from mne.transforms import Transform
from mne.bem import read_bem_solution, read_bem_surfaces


data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
trans_fname = op.join(data_dir, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
src_fname = op.join(data_dir, 'subjects', 'sample', 'bem',
                    'sample-oct-6-src.fif')
dip_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_trunc_set1.dip')
ctf_fname = op.join(data_dir, 'CTF', 'testdata_ctf.ds')
nirx_fname = op.join(data_dir, 'NIRx', 'nirscout',
                     'nirx_15_2_recording_w_short')

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
1   9998    1   4  3e-03  0.000e+00     "3mm square"
  0.1250 -0.750e-03 -0.750e-03 0.000  0.000  0.000  1.000
  0.1250 -0.750e-03  0.750e-03 0.000  0.000  0.000  1.000
  0.1250  0.750e-03 -0.750e-03 0.000  0.000  0.000  1.000
  0.1250  0.750e-03  0.750e-03 0.000  0.000  0.000  1.000
"""


def test_plot_head_positions():
    """Test plotting of head positions."""
    info = read_info(evoked_fname)
    pos = np.random.RandomState(0).randn(4, 10)
    pos[:, 0] = np.arange(len(pos))
    destination = (0., 0., 0.04)
    with _record_warnings():  # old MPL will cause a warning
        plot_head_positions(pos)
        plot_head_positions(pos, mode='field', info=info,
                            destination=destination)
        plot_head_positions([pos, pos])  # list support
        pytest.raises(ValueError, plot_head_positions, ['pos'])
        pytest.raises(ValueError, plot_head_positions, pos[:, :9])
    pytest.raises(ValueError, plot_head_positions, pos, 'foo')
    with pytest.raises(ValueError, match='shape'):
        plot_head_positions(pos, axes=1.)


@testing.requires_testing_data
@pytest.mark.slowtest
def test_plot_sparse_source_estimates(renderer_interactive, brain_gc):
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
    brain = plot_source_estimates(
        stc, 'sample', colormap=colormap, background=(1, 1, 0),
        subjects_dir=subjects_dir, colorbar=True, clim='auto')
    brain.close()
    del brain
    with pytest.raises(TypeError, match='figure must be'):
        plot_source_estimates(
            stc, 'sample', figure='foo', hemi='both', clim='auto',
            subjects_dir=subjects_dir)

    # now do sparse version
    vertices = sample_src[0]['vertno']
    inds = [111, 333]
    stc_data = np.zeros((len(inds), n_time))
    stc_data[0, 1] = 1.
    stc_data[1, 4] = 2.
    vertices = [vertices[inds], np.empty(0, dtype=np.int64)]
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    out = plot_sparse_source_estimates(
        sample_src, stc, bgcolor=(1, 1, 1), opacity=0.5,
        high_resolution=False)
    assert isinstance(out, Figure3D)


@testing.requires_testing_data
@pytest.mark.slowtest
def test_plot_evoked_field(renderer):
    """Test plotting evoked field."""
    evoked = read_evokeds(evoked_fname, condition='Left Auditory',
                          baseline=(-0.2, 0.0))
    evoked = pick_channels_evoked(evoked, evoked.ch_names[::10])  # speed
    for t, n_contours in zip(['meg', None], [21, 0]):
        with pytest.warns(RuntimeWarning, match='projection'):
            maps = make_field_map(evoked, trans_fname, subject='sample',
                                  subjects_dir=subjects_dir, n_jobs=None,
                                  ch_type=t)
        evoked.plot_field(maps, time=0.1, n_contours=n_contours)


def _assert_n_actors(fig, renderer, n_actors):
    __tracebackhide__ = True
    assert isinstance(fig, Figure3D)
    assert len(fig.plotter.renderer.actors) == n_actors


@pytest.mark.slowtest  # Slow on Azure
@testing.requires_testing_data  # all use trans + head surf
@pytest.mark.parametrize('system', [
    'Neuromag',
    'CTF',
    'BTi',
    'KIT',
])
def test_plot_alignment_meg(renderer, system):
    """Test plotting of MEG sensors + helmet."""
    if system == 'Neuromag':
        this_info = read_info(evoked_fname)
    elif system == 'CTF':
        this_info = read_raw_ctf(ctf_fname).info
    elif system == 'BTi':
        this_info = read_raw_bti(
            pdf_fname, config_fname, hs_fname, convert=True,
            preload=False).info
    else:
        assert system == 'KIT'
        this_info = read_raw_kit(sqd_fname).info

    meg = ['helmet', 'sensors']
    if system == 'KIT':
        meg.append('ref')
    fig = plot_alignment(
        this_info, read_trans(trans_fname), subject='sample',
        subjects_dir=subjects_dir, meg=meg, eeg=False)
    assert isinstance(fig, Figure3D)
    # count the number of objects: should be n_meg_ch + 1 (helmet) + 1 (head)
    use_info = pick_info(this_info, pick_types(
        this_info, meg=True, eeg=False, ref_meg='ref' in meg, exclude=()))
    n_actors = use_info['nchan'] + 2
    _assert_n_actors(fig, renderer, n_actors)


@testing.requires_testing_data
def test_plot_alignment_surf(renderer):
    """Test plotting of a surface."""
    info = read_info(evoked_fname)
    fig = plot_alignment(
        info, read_trans(trans_fname), subject='sample',
        subjects_dir=subjects_dir, meg=False, eeg=False, dig=False,
        surfaces=['white', 'head'])
    _assert_n_actors(fig, renderer, 3)  # left and right hemis plus head


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
def test_plot_alignment_basic(tmp_path, renderer, mixed_fwd_cov_evoked):
    """Test plotting of -trans.fif files and MEG sensor layouts."""
    # generate fiducials file for testing
    tempdir = str(tmp_path)
    fiducials_path = op.join(tempdir, 'fiducials.fif')
    fid = [{'coord_frame': 5, 'ident': 1, 'kind': 1,
            'r': [-0.08061612, -0.02908875, -0.04131077]},
           {'coord_frame': 5, 'ident': 2, 'kind': 1,
            'r': [0.00146763, 0.08506715, -0.03483611]},
           {'coord_frame': 5, 'ident': 3, 'kind': 1,
            'r': [0.08436285, -0.02850276, -0.04127743]}]
    write_dig(fiducials_path, fid, 5)
    evoked = read_evokeds(evoked_fname)[0]
    info = evoked.info

    sample_src = read_source_spaces(src_fname)
    pytest.raises(TypeError, plot_alignment, 'foo', trans_fname,
                  subject='sample', subjects_dir=subjects_dir)
    pytest.raises(OSError, plot_alignment, info, trans_fname,
                  subject='sample', subjects_dir=subjects_dir, src='foo')
    pytest.raises(ValueError, plot_alignment, info, trans_fname,
                  subject='fsaverage', subjects_dir=subjects_dir,
                  src=sample_src)
    sample_src.plot(subjects_dir=subjects_dir, head=True, skull=True,
                    brain='white')
    # mixed source space
    mixed_src = mixed_fwd_cov_evoked[0]['src']
    assert mixed_src.kind == 'mixed'
    fig = plot_alignment(
        info, meg=['helmet', 'sensors'], dig=True,
        coord_frame='head', trans=Path(trans_fname),
        subject='sample', mri_fiducials=fiducials_path,
        subjects_dir=subjects_dir, src=mixed_src)
    assert isinstance(fig, Figure3D)
    renderer.backend._close_all()
    # no-head version
    renderer.backend._close_all()
    # trans required
    with pytest.raises(ValueError, match='transformation matrix.*in head'):
        plot_alignment(info, trans=None, src=src_fname)
    with pytest.raises(ValueError, match='transformation matrix.*in head'):
        plot_alignment(info, trans=None, mri_fiducials=True)
    with pytest.raises(ValueError, match='transformation matrix.*in head'):
        plot_alignment(info, trans=None, surfaces=['brain'])
    assert mixed_src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD
    with pytest.raises(ValueError, match='head-coordinate source space in mr'):
        plot_alignment(trans=None, src=mixed_src, coord_frame='mri')
    # all coord frames
    plot_alignment(info)  # works: surfaces='auto' default
    for coord_frame in ('meg', 'head', 'mri'):
        plot_alignment(
            info, meg=['helmet', 'sensors'], dig=True, coord_frame=coord_frame,
            trans=Path(trans_fname), subject='sample', src=src_fname,
            mri_fiducials=fiducials_path, subjects_dir=subjects_dir)
    renderer.backend._close_all()
    # EEG only with strange options
    evoked_eeg_ecog_seeg = evoked.copy().pick_types(meg=False, eeg=True)
    with evoked_eeg_ecog_seeg.info._unlock():
        evoked_eeg_ecog_seeg.info['projs'] = []  # "remove" avg proj
    evoked_eeg_ecog_seeg.set_channel_types({'EEG 001': 'ecog',
                                            'EEG 002': 'seeg'})
    with catch_logging() as log:
        plot_alignment(evoked_eeg_ecog_seeg.info, subject='sample',
                       trans=trans_fname, subjects_dir=subjects_dir,
                       surfaces=['white', 'outer_skin', 'outer_skull'],
                       meg=['helmet', 'sensors'],
                       eeg=['original', 'projected'], ecog=True, seeg=True,
                       verbose=True)
    log = log.getvalue()
    assert 'ecog: 1' in log
    assert 'seeg: 1' in log
    renderer.backend._close_all()

    sphere = make_sphere_model(info=info, r0='auto', head_radius='auto')
    bem_sol = read_bem_solution(op.join(subjects_dir, 'sample', 'bem',
                                        'sample-1280-1280-1280-bem-sol.fif'))
    bem_surfs = read_bem_surfaces(op.join(subjects_dir, 'sample', 'bem',
                                          'sample-1280-1280-1280-bem.fif'))
    sample_src[0]['coord_frame'] = 4  # hack for coverage
    plot_alignment(info, trans_fname, subject='sample',
                   eeg='projected', meg='helmet', bem=sphere, dig=True,
                   surfaces=['brain', 'inner_skull', 'outer_skull',
                             'outer_skin'])
    plot_alignment(info, trans_fname, subject='sample', meg='helmet',
                   subjects_dir=subjects_dir, eeg='projected', bem=sphere,
                   surfaces=['head', 'brain'], src=sample_src)
    # no trans okay, no mri surfaces
    plot_alignment(info, bem=sphere, surfaces=['brain'])
    with pytest.raises(ValueError, match='A head surface is required'):
        plot_alignment(info, trans=trans_fname, subject='sample',
                       subjects_dir=subjects_dir, eeg='projected',
                       surfaces=[])
    with pytest.raises(RuntimeError, match='No brain surface found'):
        plot_alignment(info, trans=trans_fname, subject='foo',
                       subjects_dir=subjects_dir, surfaces=['brain'])
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
    sphere = make_sphere_model('auto', 'auto', info)
    src = setup_volume_source_space(sphere=sphere)
    plot_alignment(
        info, trans=Transform('head', 'mri'), eeg='projected',
        meg='helmet', bem=sphere, src=src, dig=True,
        surfaces=['brain', 'inner_skull', 'outer_skull', 'outer_skin'])
    sphere = make_sphere_model('auto', None, info)  # one layer
    # if you ask for a brain surface with a 1-layer sphere model it's an error
    with pytest.raises(RuntimeError, match='Sphere model does not have'):
        plot_alignment(
            trans=trans_fname, subject='sample', subjects_dir=subjects_dir,
            surfaces=['brain'], bem=sphere)
    # but you can ask for a specific brain surface, and
    # no info is permitted
    plot_alignment(
        trans=trans_fname, subject='sample', meg=False, coord_frame='mri',
        subjects_dir=subjects_dir, surfaces=['white'], bem=sphere,
        show_axes=True)
    renderer.backend._close_all()
    # TODO: We need to make this class public and document it properly
    # assert isinstance(fig, some_public_class)
    # 3D coil with no defined draw (ConvexHull)
    info_cube = pick_info(info, np.arange(6))
    with info._unlock():
        info['dig'] = None
    info_cube['chs'][0]['coil_type'] = 9999
    info_cube['chs'][1]['coil_type'] = 9998
    with pytest.raises(RuntimeError, match='coil definition not found'):
        plot_alignment(info_cube, meg='sensors', surfaces=())
    coil_def_fname = op.join(tempdir, 'temp')
    with open(coil_def_fname, 'w') as fid:
        fid.write(coil_3d)
    # make sure our other OPMs can be plotted, too
    for ii, kind in enumerate(('QUSPIN_ZFOPM_MAG', 'QUSPIN_ZFOPM_MAG2',
                               'FIELDLINE_OPM_MAG_GEN1',
                               'KERNEL_OPM_MAG_GEN1'), 2):
        info_cube['chs'][ii]['coil_type'] = getattr(
            FIFF, f'FIFFV_COIL_{kind}')
    with use_coil_def(coil_def_fname):
        with catch_logging() as log:
            plot_alignment(info_cube, meg='sensors', surfaces=(), dig=True,
                           verbose='debug')
    log = log.getvalue()
    assert 'planar geometry' in log

    # one layer bem with skull surfaces:
    with pytest.raises(RuntimeError, match='Sphere model does not.*boundary'):
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
    with pytest.raises(TypeError, match='surfaces.*must be'):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir,
                       surfaces=[1])
    with pytest.raises(ValueError, match='Unknown surface type'):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir,
                       surfaces=['foo'])
    with pytest.raises(TypeError, match="must be an instance of "):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir,
                       surfaces=dict(brain='super clear'))
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        plot_alignment(info=info, trans=trans_fname,
                       subject='sample', subjects_dir=subjects_dir,
                       surfaces=dict(brain=42))
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
    fwd['coord_frame'] = FIFF.FIFFV_COORD_MRI  # check required to get to MRI
    with pytest.raises(ValueError, match='transformation matrix.*in head coo'):
        plot_alignment(info, trans=None, fwd=fwd)
    # surfaces as dict
    plot_alignment(subject='sample', coord_frame='head',
                   trans=trans_fname, subjects_dir=subjects_dir,
                   surfaces={'white': 0.4, 'outer_skull': 0.6, 'head': None})


@testing.requires_testing_data
def test_plot_alignment_fnirs(renderer, tmp_path):
    """Test fNIRS plotting."""
    # Here we use subjects_dir=tmp_path, since no surfaces should actually
    # be loaded!

    # fNIRS (default is pairs)
    info = read_raw_nirx(nirx_fname).info
    assert info['nchan'] == 26
    kwargs = dict(trans='fsaverage', subject='fsaverage', surfaces=(),
                  verbose=True, subjects_dir=tmp_path)
    with catch_logging() as log:
        fig = plot_alignment(info, **kwargs)
    log = log.getvalue()
    assert f'fnirs_cw_amplitude: {info["nchan"]}' in log
    _assert_n_actors(fig, renderer, info['nchan'])

    fig = plot_alignment(
        info, fnirs=['channels', 'sources', 'detectors'], **kwargs)
    _assert_n_actors(fig, renderer, 3)


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
def test_process_clim_plot(renderer_interactive, brain_gc):
    """Test functionality for determining control points with stc.plot."""
    sample_src = read_source_spaces(src_fname)
    kwargs = dict(subjects_dir=subjects_dir, smoothing_steps=1,
                  time_viewer=False, show_traces=False)

    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.random.RandomState(0).rand((n_verts * n_time))
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1, 'sample')

    # Test for simple use cases
    brain = stc.plot(**kwargs)
    assert brain.data['center'] is None
    brain.close()
    brain = stc.plot(clim=dict(pos_lims=(10, 50, 90)), **kwargs)
    assert brain.data['center'] == 0.
    brain.close()
    brain = stc.plot(colormap='hot', clim='auto', **kwargs)
    brain.close()
    brain = stc.plot(colormap='mne', clim='auto', **kwargs)
    brain.close()
    brain = stc.plot(clim=dict(kind='value', lims=(10, 50, 90)), figure=99,
                     **kwargs)
    brain.close()
    with pytest.raises(TypeError, match='must be a'):
        stc.plot(clim='auto', figure=[0], **kwargs)

    # Test for correct clim values
    with pytest.raises(ValueError, match='monotonically'):
        stc.plot(clim=dict(kind='value', pos_lims=[0, 1, 0]), **kwargs)
    with pytest.raises(ValueError, match=r'.*must be \(3,\)'):
        stc.plot(colormap='mne', clim=dict(pos_lims=(5, 10, 15, 20)), **kwargs)
    with pytest.raises(ValueError, match="'value', 'values', and 'percent'"):
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
        brain = plot_source_estimates(stc, **kwargs)
    brain.close()


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
        colormap=_get_cmap('hot'),
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
    stc.plot(subjects_dir=subjects_dir, time_unit='s', views='ven',
             hemi='rh', smoothing_steps=7, subject='sample',
             backend='matplotlib', spacing='oct1', initial_time=0.001,
             colormap='Reds')
    fig = stc.plot(subjects_dir=subjects_dir, time_unit='ms', views='dor',
                   hemi='lh', smoothing_steps=7, subject='sample',
                   backend='matplotlib', spacing='ico2', time_viewer=True,
                   colormap='mne')
    time_viewer = fig.time_viewer
    _fake_click(time_viewer, time_viewer.axes[0], (0.5, 0.5))  # change t
    _fake_keypress(time_viewer, 'ctrl+right')
    _fake_keypress(time_viewer, 'left')
    pytest.raises(ValueError, stc.plot, subjects_dir=subjects_dir,
                  hemi='both', subject='sample', backend='matplotlib')
    pytest.raises(ValueError, stc.plot, subjects_dir=subjects_dir,
                  time_unit='ss', subject='sample', backend='matplotlib')


@pytest.mark.slowtest
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
    _fake_scroll(fig, 0.5, 0.5, 1)  # scroll up
    _fake_scroll(fig, 0.5, 0.5, -1)  # scroll down
    _fake_keypress(fig, 'up')
    _fake_keypress(fig, 'down')
    _fake_keypress(fig, 'a')  # some other key
    ax = fig.add_subplot(211)
    with pytest.raises(TypeError, match='instance of Axes3D'):
        dipoles.plot_locations(trans, 'sample', subjects_dir, ax=ax)


@testing.requires_testing_data
@pytest.mark.parametrize('surf, coord_frame, ax, title', [
    pytest.param('white', 'mri', None, None, marks=pytest.mark.slowtest),
    pytest.param(None, 'head', None, None, marks=pytest.mark.slowtest),
    (None, 'mri_rotated', 'mpl', 'check'),
])
def test_plot_dipole_mri_outlines(surf, coord_frame, ax, title):
    """Test mpl dipole plotting."""
    dipoles = read_dipole(dip_fname)
    trans = read_trans(trans_fname)
    if ax is not None:
        assert isinstance(ax, str) and ax == 'mpl', ax
        _, ax = plt.subplots(3, 1)
        ax = list(ax)
        with pytest.raises(ValueError, match='but the length is 2'):
            dipoles.plot_locations(
                trans, 'sample', subjects_dir, ax=ax[:2], mode='outlines')
    fig = dipoles.plot_locations(
        trans=trans, subject='sample', subjects_dir=subjects_dir,
        mode='outlines', coord_frame=coord_frame, surf=surf, ax=ax,
        title=title)
    assert isinstance(fig, Figure)


@testing.requires_testing_data
def test_plot_dipole_orientations(renderer):
    """Test dipole plotting in 3d."""
    dipoles = read_dipole(dip_fname)
    trans = read_trans(trans_fname)
    for coord_frame, mode in zip(['head', 'mri'],
                                 ['arrow', 'sphere']):
        fig = dipoles.plot_locations(
            trans=trans, subject='sample', subjects_dir=subjects_dir,
            mode=mode, coord_frame=coord_frame)
        assert isinstance(fig, Figure3D)
    renderer.backend._close_all()


@pytest.mark.slowtest  # slow on Azure
@testing.requires_testing_data
def test_snapshot_brain_montage(renderer):
    """Test snapshot brain montage."""
    info = read_info(evoked_fname)
    fig = plot_alignment(
        info, trans=Transform('head', 'mri'), subject='sample',
        subjects_dir=subjects_dir)

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
@pytest.mark.parametrize('pick_ori', ('vector', None))
@pytest.mark.parametrize('kind', ('surface', 'volume', 'mixed'))
def test_plot_source_estimates(renderer_interactive, all_src_types_inv_evoked,
                               pick_ori, kind, brain_gc):
    """Test plotting of scalar and vector source estimates."""
    backend = renderer_interactive._get_3d_backend()
    invs, evoked = all_src_types_inv_evoked
    inv = invs[kind]
    with _record_warnings():  # PCA mag
        stc = apply_inverse(evoked, inv, pick_ori=pick_ori)
    stc.data[1] *= -1  # make it signed
    meth_key = 'plot_3d' if isinstance(stc, _BaseVolSourceEstimate) else 'plot'
    stc.subject = 'sample'
    meth = getattr(stc, meth_key)
    kwargs = dict(subjects_dir=subjects_dir,
                  time_viewer=False, show_traces=False,  # for speed
                  smoothing_steps=1, verbose='error', src=inv['src'],
                  volume_options=dict(resolution=None),  # for speed
                  )
    if pick_ori != 'vector':
        kwargs['surface'] = 'white'
        kwargs['backend'] = backend
    brain = meth(**kwargs)
    brain.close()
    del brain

    these_kwargs = kwargs.copy()
    these_kwargs['show_traces'] = 'foo'
    with pytest.raises(ValueError, match='show_traces'):
        meth(**these_kwargs)
    del these_kwargs
    if pick_ori == 'vector':
        with pytest.raises(ValueError, match='use "pos_lims"'):
            meth(**kwargs, clim=dict(pos_lims=[1, 2, 3]))
    if kind in ('volume', 'mixed'):
        with pytest.raises(TypeError, match='when stc is a mixed or vol'):
            these_kwargs = kwargs.copy()
            these_kwargs.pop('src')
            meth(**these_kwargs)

    with pytest.raises(ValueError, match='cannot be used'):
        these_kwargs = kwargs.copy()
        these_kwargs.update(show_traces=True, time_viewer=False)
        meth(**these_kwargs)

    # flatmaps (mostly a lot of error checking)
    these_kwargs = kwargs.copy()
    these_kwargs.update(surface='flat', views='auto', hemi='both',
                        verbose='debug')
    if kind == 'surface' and pick_ori != 'vector':
        with catch_logging() as log:
            with pytest.raises(FileNotFoundError, match='flatmap'):
                meth(**these_kwargs)  # sample does not have them
        log = log.getvalue()
        assert 'offset: 0' in log
    fs_stc = stc.copy()
    fs_stc.subject = 'fsaverage'  # this is wrong, but don't have to care
    flat_meth = getattr(fs_stc, meth_key)
    these_kwargs.pop('src')
    if pick_ori == 'vector':
        pass  # can't even pass "surface" variable
    elif kind != 'surface':
        with pytest.raises(TypeError, match='SourceEstimate when a flatmap'):
            flat_meth(**these_kwargs)
    else:
        brain = flat_meth(**these_kwargs)
        brain.close()
        del brain
        these_kwargs.update(surface='inflated', views='flat')
        with pytest.raises(ValueError, match='surface="flat".*views="flat"'):
            flat_meth(**these_kwargs)

    # just test one for speed
    if kind != 'mixed':
        return
    brain = meth(
        views=['lat', 'med', 'ven'], hemi='lh',
        view_layout='horizontal', **kwargs)
    brain.close()
    assert brain._subplot_shape == (1, 3)
    del brain
    these_kwargs = kwargs.copy()
    these_kwargs['volume_options'] = dict(blending='foo')
    with pytest.raises(ValueError, match='mip'):
        meth(**these_kwargs)
    these_kwargs['volume_options'] = dict(badkey='foo')
    with pytest.raises(ValueError, match='unknown'):
        meth(**these_kwargs)
    # with resampling (actually downsampling but it's okay)
    these_kwargs['volume_options'] = dict(resolution=20., surface_alpha=0.)
    brain = meth(**these_kwargs)
    brain.close()
    del brain


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
    cbar = plot_brain_colorbar(ax, clim, orientation=orientation)
    ax = cbar.ax  # in newer mpl this can be inset axes relative to the orig
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
    ax.figure.canvas.draw_idle()
    assert_array_equal(
        [float(h.get_text().replace('−', '-')) for h in have()], ticks)
    assert_array_equal(empty(), [])


@pytest.mark.slowtest  # slow-ish on Travis OSX
@testing.requires_testing_data
def test_mixed_sources_plot_surface(renderer_interactive):
    """Test plot_surface() for mixed source space."""
    src = read_source_spaces(fwd_fname2)
    N = np.sum([s['nuse'] for s in src])  # number of sources

    T = 2  # number of time points
    S = 3  # number of source spaces

    rng = np.random.RandomState(0)
    data = rng.randn(N, T)
    vertno = S * [np.arange(N // S)]

    stc = MixedSourceEstimate(data, vertno, 0, 1)

    brain = stc.surface().plot(views='lat', hemi='split',
                               subject='fsaverage', subjects_dir=subjects_dir,
                               colorbar=False)
    brain.close()
    del brain


@testing.requires_testing_data
@pytest.mark.slowtest
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
    with pytest.raises(ValueError, match='is empty'):
        link_brains([])
    with pytest.raises(TypeError, match='type is Brain'):
        link_brains('foo')
    link_brains(brain, time=True, camera=True)
