# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD-3-Clause

from copy import deepcopy
from os import makedirs
import os.path as op
import re
from shutil import copy

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose

import mne
from mne import (make_bem_model, read_bem_surfaces, write_bem_surfaces,
                 make_bem_solution, read_bem_solution, write_bem_solution,
                 make_sphere_model, Transform, Info, write_surface,
                 write_head_bem)
from mne.preprocessing.maxfilter import fit_sphere_to_headshape
from mne.io.constants import FIFF
from mne.transforms import translation
from mne.datasets import testing
from mne.utils import catch_logging, check_version
from mne.bem import (_ico_downsample, _get_ico_map, _order_surfaces,
                     _assert_complete_surface, _assert_inside,
                     _check_surface_size, _bem_find_surface,
                     make_scalp_surfaces, distance_to_bem)
from mne.surface import read_surface, _get_ico_surface
from mne.io import read_info

fname_raw = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data',
                    'test_raw.fif')
subjects_dir = op.join(testing.data_path(download=False), 'subjects')
fname_bem_3 = op.join(subjects_dir, 'sample', 'bem',
                      'sample-320-320-320-bem.fif')
fname_bem_1 = op.join(subjects_dir, 'sample', 'bem',
                      'sample-320-bem.fif')
fname_bem_sol_3 = op.join(subjects_dir, 'sample', 'bem',
                          'sample-320-320-320-bem-sol.fif')
fname_bem_sol_1 = op.join(subjects_dir, 'sample', 'bem',
                          'sample-320-bem-sol.fif')
fname_dense_head = op.join(subjects_dir, 'sample', 'bem',
                           'sample-head-dense.fif')


def _compare_bem_surfaces(surfs_1, surfs_2):
    """Compare BEM surfaces."""
    names = ['id', 'nn', 'rr', 'coord_frame', 'tris', 'sigma', 'ntri', 'np']
    ignores = ['tri_cent', 'tri_nn', 'tri_area', 'neighbor_tri']
    for s0, s1 in zip(surfs_1, surfs_2):
        assert_equal(set(names), set(s0.keys()) - set(ignores))
        assert_equal(set(names), set(s1.keys()) - set(ignores))
        for name in names:
            assert_allclose(s0[name], s1[name], rtol=1e-3, atol=1e-6,
                            err_msg='Mismatch: "%s"' % name)


def _compare_bem_solutions(sol_a, sol_b):
    """Compare BEM solutions."""
    # compare the surfaces we used
    _compare_bem_surfaces(sol_a['surfs'], sol_b['surfs'])
    # compare the actual solutions
    names = ['bem_method', 'field_mult', 'gamma', 'is_sphere',
             'nsol', 'sigma', 'source_mult', 'solution']
    assert set(sol_a.keys()) == set(sol_b.keys())
    assert set(names + ['solver', 'surfs']) == set(sol_b.keys())
    assert sol_a['solver'] == sol_b['solver']
    for key in names[:-1]:
        assert_allclose(sol_a[key], sol_b[key], rtol=1e-3, atol=1e-5,
                        err_msg='Mismatch: %s' % key)


h5py_mark = pytest.mark.skipif(not check_version('h5py'), reason='Needs h5py')


@testing.requires_testing_data
@pytest.mark.parametrize('ext', [
    'fif',
    pytest.param('h5', marks=h5py_mark),
])
def test_io_bem(tmp_path, ext):
    """Test reading and writing of bem surfaces and solutions."""
    temp_bem = op.join(str(tmp_path), f'temp-bem.{ext}')
    # model
    with pytest.raises(ValueError, match='BEM data not found'):
        read_bem_surfaces(fname_raw)
    with pytest.raises(ValueError, match='surface with id 10'):
        read_bem_surfaces(fname_bem_3, s_id=10)
    surf = read_bem_surfaces(fname_bem_3, patch_stats=True)
    surf = read_bem_surfaces(fname_bem_3, patch_stats=False)
    write_bem_surfaces(temp_bem, surf[0])
    with pytest.raises(IOError, match='exists'):
        write_bem_surfaces(temp_bem, surf[0])
    write_bem_surfaces(temp_bem, surf[0], overwrite=True)
    if ext == 'h5':
        import h5py
        with h5py.File(temp_bem, 'r'):  # make sure it's valid
            pass
    surf_read = read_bem_surfaces(temp_bem, patch_stats=False)
    _compare_bem_surfaces(surf, surf_read)

    # solution
    with pytest.raises(RuntimeError, match='No BEM solution found'):
        read_bem_solution(fname_bem_3)
    temp_sol = op.join(str(tmp_path), f'temp-sol.{ext}')
    sol = read_bem_solution(fname_bem_sol_3)
    assert 'BEM' in repr(sol)
    write_bem_solution(temp_sol, sol)
    sol_read = read_bem_solution(temp_sol)
    _compare_bem_solutions(sol, sol_read)
    sol = read_bem_solution(fname_bem_sol_1)
    with pytest.raises(RuntimeError, match='BEM does not have.*triangulation'):
        _bem_find_surface(sol, 3)


def test_make_sphere_model():
    """Test making a sphere model."""
    info = read_info(fname_raw)
    pytest.raises(ValueError, make_sphere_model, 'foo', 'auto', info)
    pytest.raises(ValueError, make_sphere_model, 'auto', 'auto', None)
    pytest.raises(ValueError, make_sphere_model, 'auto', 'auto', info,
                  relative_radii=(), sigmas=())
    with pytest.raises(ValueError, match='relative_radii.*must match.*sigmas'):
        make_sphere_model('auto', 'auto', info, relative_radii=(1,))
    # here we just make sure it works -- the functionality is actually
    # tested more extensively e.g. in the forward and dipole code
    with catch_logging() as log:
        bem = make_sphere_model('auto', 'auto', info, verbose=True)
    log = log.getvalue()
    assert ' RV = ' in log
    for line in log.split('\n'):
        if ' RV = ' in line:
            val = float(line.split()[-2])
            assert val < 0.01  # actually decent fitting
            break
    assert '3 layers' in repr(bem)
    assert 'Sphere ' in repr(bem)
    assert ' mm' in repr(bem)
    bem = make_sphere_model('auto', None, info)
    assert 'no layers' in repr(bem)
    assert 'Sphere ' in repr(bem)
    with pytest.raises(ValueError, match='at least 2 sigmas.*head_radius'):
        make_sphere_model(sigmas=(0.33,), relative_radii=(1.0,))


@testing.requires_testing_data
@pytest.mark.parametrize('kwargs, fname', [
    pytest.param(dict(), fname_bem_3, marks=pytest.mark.slowtest),  # Azure
    [dict(conductivity=[0.3]), fname_bem_1],
])
def test_make_bem_model(tmp_path, kwargs, fname):
    """Test BEM model creation from Python with I/O."""
    fname_temp = tmp_path / 'temp-bem.fif'
    with catch_logging() as log:
        model = make_bem_model('sample', ico=2, subjects_dir=subjects_dir,
                               verbose=True, **kwargs)
    log = log.getvalue()
    if len(kwargs.get('conductivity', (0, 0, 0))) == 1:
        assert 'distance' not in log
    else:
        assert re.search(r'urfaces is approximately *3\.4 mm', log) is not None
    assert re.search(r'inner skull CM is *0\.65 *-9\.62 *43\.85 mm',
                     log) is not None
    model_c = read_bem_surfaces(fname)
    _compare_bem_surfaces(model, model_c)
    write_bem_surfaces(fname_temp, model)
    model_read = read_bem_surfaces(fname_temp)
    _compare_bem_surfaces(model, model_c)
    _compare_bem_surfaces(model_read, model_c)
    # bad conductivity
    with pytest.raises(ValueError, match='conductivity must be'):
        make_bem_model('sample', 4, [0.3, 0.006], subjects_dir=subjects_dir)


@testing.requires_testing_data
def test_bem_model_topology(tmp_path):
    """Test BEM model topological checks."""
    # bad topology (not enough neighboring tris)
    makedirs(tmp_path / 'foo' / 'bem')
    for fname in ('inner_skull', 'outer_skull', 'outer_skin'):
        fname += '.surf'
        copy(op.join(subjects_dir, 'sample', 'bem', fname),
             tmp_path / 'foo' / 'bem' / fname)
    outer_fname = tmp_path / 'foo' / 'bem' / 'outer_skull.surf'
    rr, tris = read_surface(outer_fname)
    tris = tris[:-1]
    write_surface(outer_fname, rr, tris[:-1], overwrite=True)
    with pytest.raises(RuntimeError, match='Surface outer skull is not compl'):
        make_bem_model('foo', None, subjects_dir=tmp_path)
    # Now get past this error to reach gh-6127 (not enough neighbor tris)
    rr_bad = np.concatenate([rr, np.mean(rr, axis=0, keepdims=True)], axis=0)
    write_surface(outer_fname, rr_bad, tris, overwrite=True)
    with pytest.raises(ValueError, match='Surface outer skull.*triangles'):
        make_bem_model('foo', None, subjects_dir=tmp_path)


@pytest.mark.slowtest
@testing.requires_testing_data
@pytest.mark.parametrize('cond, fname', [
    [(0.3,), fname_bem_sol_1],
    [(0.3, 0.006, 0.3), fname_bem_sol_3],
])
def test_bem_solution(tmp_path, cond, fname):
    """Test making a BEM solution from Python and OpenMEEG with I/O."""
    # test degenerate conditions
    surf = read_bem_surfaces(fname_bem_1)[0]
    with pytest.raises(RuntimeError, match='2 or less'):
        _ico_downsample(surf, 10)
    s_bad = dict(tris=surf['tris'][1:], ntri=surf['ntri'] - 1, rr=surf['rr'])
    with pytest.raises(RuntimeError, match='Cannot decimate.*isomorphic'):
        _ico_downsample(s_bad, 1)
    s_bad = dict(tris=surf['tris'].copy(), ntri=surf['ntri'],
                 rr=surf['rr'])  # bad triangulation
    s_bad['tris'][0] = [0, 0, 0]
    with pytest.raises(RuntimeError, match='ordering is wrong'):
        _ico_downsample(s_bad, 1)
    s_bad['id'] = 1
    with pytest.raises(RuntimeError, match='is not complete'):
        _assert_complete_surface(s_bad)
    s_bad = dict(tris=surf['tris'], ntri=surf['ntri'], rr=surf['rr'].copy())
    s_bad['rr'][0] = 0.
    with pytest.raises(RuntimeError, match='No matching vertex'):
        _get_ico_map(surf, s_bad)

    surfs = read_bem_surfaces(fname_bem_3)
    with pytest.raises(RuntimeError, match='is not completely inside'):
        _assert_inside(surfs[0], surfs[1])  # outside
    surfs[0]['id'] = 100  # bad surfs
    with pytest.raises(RuntimeError, match='bad surface id'):
        _order_surfaces(surfs)
    surfs[1]['rr'] /= 1000.
    with pytest.raises(RuntimeError, match='seem too small'):
        _check_surface_size(surfs[1])

    # actually test functionality
    fname_temp = op.join(str(tmp_path), 'temp-bem-sol.fif')
    # use a model and solution made in Python
    for model_type in ('python', 'c'):
        if model_type == 'python':
            model = make_bem_model('sample', conductivity=cond, ico=2,
                                   subjects_dir=subjects_dir)
        else:
            model = fname_bem_1 if len(cond) == 1 else fname_bem_3
    solution = make_bem_solution(model, verbose=True)
    assert solution['solver'] == 'mne'
    solution_c = read_bem_solution(fname)
    assert solution_c['solver'] == 'mne'
    _compare_bem_solutions(solution, solution_c)
    write_bem_solution(fname_temp, solution)
    solution_read = read_bem_solution(fname_temp)
    assert solution['solver'] == solution_c['solver'] == 'mne'
    assert solution_read['solver'] == 'mne'
    _compare_bem_solutions(solution, solution_c)
    _compare_bem_solutions(solution_read, solution_c)
    # OpenMEEG
    pytest.importorskip(
        'openmeeg', '2.5', reason='OpenMEEG required to fully test BEM '
        'solution computation')
    with catch_logging() as log:
        solution = make_bem_solution(model, solver='openmeeg', verbose=True)
    log = log.getvalue()
    assert 'OpenMEEG' in log
    write_bem_solution(fname_temp, solution, overwrite=True)
    solution_read = read_bem_solution(fname_temp)
    assert solution['solver'] == solution_read['solver'] == 'openmeeg'
    _compare_bem_solutions(solution_read, solution)


def test_fit_sphere_to_headshape():
    """Test fitting a sphere to digitization points."""
    # Create points of various kinds
    rad = 0.09
    big_rad = 0.12
    center = np.array([0.0005, -0.01, 0.04])
    dev_trans = np.array([0., -0.005, -0.01])
    dev_center = center - dev_trans
    dig = [
        # Left auricular
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'ident': FIFF.FIFFV_POINT_LPA,
         'kind': FIFF.FIFFV_POINT_CARDINAL,
         'r': np.array([-1.0, 0.0, 0.0])},
        # Nasion
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'ident': FIFF.FIFFV_POINT_NASION,
         'kind': FIFF.FIFFV_POINT_CARDINAL,
         'r': np.array([0.0, 1.0, 0.0])},
        # Right auricular
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'ident': FIFF.FIFFV_POINT_RPA,
         'kind': FIFF.FIFFV_POINT_CARDINAL,
         'r': np.array([1.0, 0.0, 0.0])},

        # Top of the head (extra point)
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EXTRA,
         'ident': 0,
         'r': np.array([0.0, 0.0, 1.0])},

        # EEG points
        # Fz
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'ident': 0,
         'r': np.array([0, .72, .69])},
        # F3
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'ident': 1,
         'r': np.array([-.55, .67, .50])},
        # F4
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'ident': 2,
         'r': np.array([.55, .67, .50])},
        # Cz
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'ident': 3,
         'r': np.array([0.0, 0.0, 1.0])},
        # Pz
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'ident': 4,
         'r': np.array([0, -.72, .69])},
    ]
    for d in dig:
        d['r'] *= rad
        d['r'] += center

    # Device to head transformation (rotate .2 rad over X-axis)
    dev_head_t = Transform('meg', 'head', translation(*(dev_trans)))
    info = Info(dig=dig, dev_head_t=dev_head_t)

    # Degenerate conditions
    pytest.raises(ValueError, fit_sphere_to_headshape, info,
                  dig_kinds=(FIFF.FIFFV_POINT_HPI,))
    pytest.raises(ValueError, fit_sphere_to_headshape, info,
                  dig_kinds='foo', units='m')
    for d in info['dig']:
        d['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    with pytest.raises(RuntimeError, match='not in head coordinates'):
        fit_sphere_to_headshape(info)
    for d in info['dig']:
        d['coord_frame'] = FIFF.FIFFV_COORD_HEAD

    #  # Test with 4 points that match a perfect sphere
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA)
    with pytest.warns(RuntimeWarning, match='Only .* head digitization'):
        r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds,
                                            units='m')
    kwargs = dict(rtol=1e-3, atol=1e-5)
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, dev_center, **kwargs)

    # Test with all points
    dig_kinds = ('cardinal', FIFF.FIFFV_POINT_EXTRA, 'eeg')
    kwargs = dict(rtol=1e-3, atol=1e-3)
    with pytest.warns(RuntimeWarning, match='Only .* head digitization'):
        r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds,
                                            units='m')
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, dev_center, **kwargs)

    # Test with some noisy EEG points only.
    dig_kinds = 'eeg'
    with pytest.warns(RuntimeWarning, match='Only .* head digitization'):
        r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds,
                                            units='m')
    kwargs = dict(rtol=1e-3, atol=1e-2)
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, center, **kwargs)

    # Test big size
    dig_kinds = ('cardinal', 'extra')
    info_big = deepcopy(info)
    for d in info_big['dig']:
        d['r'] -= center
        d['r'] *= big_rad / rad
        d['r'] += center
    with pytest.warns(RuntimeWarning, match='Estimated head size'):
        r, oh, od = fit_sphere_to_headshape(info_big, dig_kinds=dig_kinds,
                                            units='mm')
    assert_allclose(oh, center * 1000, atol=1e-3)
    assert_allclose(r, big_rad * 1000, atol=1e-3)
    del info_big

    # Test offcenter
    dig_kinds = ('cardinal', 'extra')
    info_shift = deepcopy(info)
    shift_center = np.array([0., -0.03, 0.])
    for d in info_shift['dig']:
        d['r'] -= center
        d['r'] += shift_center
    with pytest.warns(RuntimeWarning, match='from head frame origin'):
        r, oh, od = fit_sphere_to_headshape(
            info_shift, dig_kinds=dig_kinds, units='m')
    assert_allclose(oh, shift_center, atol=1e-6)
    assert_allclose(r, rad, atol=1e-6)

    # Test "auto" mode (default)
    # Should try "extra", fail, and go on to EEG
    with pytest.warns(RuntimeWarning, match='Only .* head digitization'):
        r, oh, od = fit_sphere_to_headshape(info, units='m')
    kwargs = dict(rtol=1e-3, atol=1e-3)
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, dev_center, **kwargs)
    with pytest.warns(RuntimeWarning, match='Only .* head digitization'):
        r2, oh2, od2 = fit_sphere_to_headshape(info, units='m')
    assert_allclose(r, r2, atol=1e-7)
    assert_allclose(oh, oh2, atol=1e-7)
    assert_allclose(od, od2, atol=1e-7)
    # this one should pass, 1 EXTRA point and 3 EEG (but the fit is terrible)
    info = Info(dig=dig[:7], dev_head_t=dev_head_t)
    with pytest.warns(RuntimeWarning, match='Only .* head digitization'):
        r, oh, od = fit_sphere_to_headshape(info, units='m')
    # this one should fail, 1 EXTRA point and 3 EEG (but the fit is terrible)
    info = Info(dig=dig[:6], dev_head_t=dev_head_t)
    pytest.raises(ValueError, fit_sphere_to_headshape, info, units='m')
    pytest.raises(TypeError, fit_sphere_to_headshape, 1, units='m')


@pytest.mark.slowtest  # ~2 min on Azure Windows
@testing.requires_testing_data
def test_io_head_bem(tmp_path):
    """Test reading and writing of defective head surfaces."""
    head = read_bem_surfaces(fname_dense_head)[0]
    fname_defect = op.join(str(tmp_path), 'temp-head-defect.fif')
    # create defects
    head['rr'][0] = np.array([-0.01487014, -0.04563854, -0.12660208])
    head['tris'][0] = np.array([21919, 21918, 21907])

    with pytest.raises(ValueError, match='topological defects:'):
        write_head_bem(fname_defect, head['rr'], head['tris'])
    with pytest.warns(RuntimeWarning, match='topological defects:'):
        write_head_bem(fname_defect, head['rr'], head['tris'],
                       on_defects='warn')
    # test on_defects in read_bem_surfaces
    with pytest.raises(ValueError, match='topological defects:'):
        read_bem_surfaces(fname_defect)
    with pytest.warns(RuntimeWarning, match='topological defects:'):
        head_defect = read_bem_surfaces(fname_defect, on_defects='warn')[0]

    assert head['id'] == head_defect['id'] == FIFF.FIFFV_BEM_SURF_ID_HEAD
    assert np.allclose(head['rr'], head_defect['rr'])
    assert np.allclose(head['tris'], head_defect['tris'])


@pytest.mark.slowtest  # ~4 sec locally
def test_make_scalp_surfaces_topology(tmp_path, monkeypatch):
    """Test topology checks for make_scalp_surfaces."""
    pytest.importorskip('pyvista')
    subjects_dir = tmp_path
    subject = 'test'
    surf_dir = subjects_dir / subject / 'surf'
    makedirs(surf_dir)
    surf = _get_ico_surface(2)
    surf['rr'] *= 100  # mm
    write_surface(surf_dir / 'lh.seghead', surf['rr'], surf['tris'])

    # make it so that decimation really messes up the mesh just by deleting
    # the last N tris
    def _decimate_surface(points, triangles, n_triangles):
        assert len(triangles) >= n_triangles
        return points, triangles[:n_triangles]

    monkeypatch.setattr(mne.bem, 'decimate_surface', _decimate_surface)
    # TODO: These two errors should probably have the same class...

    # Not enough neighbors
    monkeypatch.setattr(mne.bem, '_tri_levels', dict(sparse=315))
    with pytest.raises(ValueError, match='.*have fewer than three.*'):
        make_scalp_surfaces(subject, subjects_dir, force=False, verbose=True)
    monkeypatch.setattr(mne.bem, '_tri_levels', dict(sparse=319))
    # Incomplete surface (sum of solid angles)
    with pytest.raises(RuntimeError, match='.*is not complete.*'):
        make_scalp_surfaces(
            subject, subjects_dir, force=False, verbose=True, overwrite=True)
    bem_dir = subjects_dir / subject / 'bem'
    sparse_path = (bem_dir / f'{subject}-head-sparse.fif')
    assert not sparse_path.is_file()

    # These are ignorable
    monkeypatch.setattr(mne.bem, '_tri_levels', dict(sparse=315))
    with pytest.warns(RuntimeWarning, match='.*have fewer than three.*'):
        make_scalp_surfaces(
            subject, subjects_dir, force=True, overwrite=True)
    surf, = read_bem_surfaces(sparse_path, on_defects='ignore')
    assert len(surf['tris']) == 315
    monkeypatch.setattr(mne.bem, '_tri_levels', dict(sparse=319))
    with pytest.warns(RuntimeWarning, match='.*is not complete.*'):
        make_scalp_surfaces(
            subject, subjects_dir, force=True, overwrite=True)
    surf, = read_bem_surfaces(sparse_path, on_defects='ignore')
    assert len(surf['tris']) == 319


@pytest.mark.parametrize("bem_type", ["bem", "sphere"])
@pytest.mark.parametrize("n_pos", [1, 10])
@testing.requires_testing_data
def test_distance_to_bem(bem_type, n_pos):
    """Test distance_to_bem."""
    # Test spherical ConductorModels
    if bem_type == "sphere":
        bem = make_sphere_model(r0=np.array([0, 0, 0]), verbose=0)
        r = bem['layers'][0]['rad']
        true_dist = np.array([r, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    else:
        bem = read_bem_solution(fname_bem_sol_1)
        r = 0.05
        true_dist = np.array([
            0.01708097, 0.00256595, 0.01022884, 0.02306622, 0.02927288,
            0.04491787, 0.00990493, 0.02244751, 0.04819345, 0.01928304
        ])

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [r, 0.0, 0.0],
            [-r, 0.0, 0.0],
            [0.0, r, 0.0],
            [0.0, -r, 0.0],
            [0.0, 0.0, r],
            [-r / np.sqrt(2.), r / np.sqrt(2.), 0.0],
            [-r / np.sqrt(2.), -r / np.sqrt(2.), 0.0],
            [0, -r / np.sqrt(2.), r / np.sqrt(2.)],
            [r / np.sqrt(3.), r / np.sqrt(3.), r / np.sqrt(3.)]
        ]
    )

    if n_pos == 1:
        pos = pos[0, :]
        true_dist = true_dist[0]

    dist = distance_to_bem(pos, bem)
    if n_pos == 1:
        assert isinstance(dist, float)
    else:
        assert isinstance(dist, np.ndarray)

    assert_allclose(dist, true_dist, rtol=1e-6, atol=1e-6)
