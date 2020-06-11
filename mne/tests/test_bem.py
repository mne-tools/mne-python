# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD 3 clause

from copy import deepcopy
from os import remove, makedirs
import os.path as op
import re
from shutil import copy

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
import matplotlib.pyplot as plt

from mne import (make_bem_model, read_bem_surfaces, write_bem_surfaces,
                 make_bem_solution, read_bem_solution, write_bem_solution,
                 make_sphere_model, Transform, Info, write_surface)
from mne.preprocessing.maxfilter import fit_sphere_to_headshape
from mne.io.constants import FIFF
from mne.transforms import translation
from mne.datasets import testing
from mne.utils import (run_tests_if_main, catch_logging,
                       requires_freesurfer, requires_nibabel)
from mne.bem import (_ico_downsample, _get_ico_map, _order_surfaces,
                     _assert_complete_surface, _assert_inside,
                     _check_surface_size, _bem_find_surface, make_flash_bem)
from mne.surface import read_surface
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
    assert_equal(set(sol_a.keys()), set(sol_b.keys()))
    assert_equal(set(names + ['surfs']), set(sol_b.keys()))
    for key in names:
        assert_allclose(sol_a[key], sol_b[key], rtol=1e-3, atol=1e-5,
                        err_msg='Mismatch: %s' % key)


@testing.requires_testing_data
def test_io_bem(tmpdir):
    """Test reading and writing of bem surfaces and solutions."""
    temp_bem = op.join(str(tmpdir), 'temp-bem.fif')
    pytest.raises(ValueError, read_bem_surfaces, fname_raw)
    pytest.raises(ValueError, read_bem_surfaces, fname_bem_3, s_id=10)
    surf = read_bem_surfaces(fname_bem_3, patch_stats=True)
    surf = read_bem_surfaces(fname_bem_3, patch_stats=False)
    write_bem_surfaces(temp_bem, surf[0])
    surf_read = read_bem_surfaces(temp_bem, patch_stats=False)
    _compare_bem_surfaces(surf, surf_read)

    pytest.raises(RuntimeError, read_bem_solution, fname_bem_3)
    temp_sol = op.join(str(tmpdir), 'temp-sol.fif')
    sol = read_bem_solution(fname_bem_sol_3)
    assert 'BEM' in repr(sol)
    write_bem_solution(temp_sol, sol)
    sol_read = read_bem_solution(temp_sol)
    _compare_bem_solutions(sol, sol_read)
    sol = read_bem_solution(fname_bem_sol_1)
    pytest.raises(RuntimeError, _bem_find_surface, sol, 3)


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
    [dict(), fname_bem_3],
    [dict(conductivity=[0.3]), fname_bem_1],
])
def test_make_bem_model(tmpdir, kwargs, fname):
    """Test BEM model creation from Python with I/O."""
    fname_temp = tmpdir.join('temp-bem.fif')
    with catch_logging() as log:
        model = make_bem_model('sample', ico=2, subjects_dir=subjects_dir,
                               verbose=True, **kwargs)
    log = log.getvalue()
    if len(kwargs.get('conductivity', (0, 0, 0))) == 1:
        assert 'distance' not in log
    else:
        assert re.search(r'urfaces is approximately *3\.6 mm', log) is not None
    assert re.search(r'inner skull CM is *0\.69 *-10\.00 *44\.26 mm',
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
def test_bem_model_topology(tmpdir):
    """Test BEM model topological checks."""
    # bad topology (not enough neighboring tris)
    makedirs(tmpdir.join('foo', 'bem'))
    for fname in ('inner_skull', 'outer_skull', 'outer_skin'):
        fname += '.surf'
        copy(op.join(subjects_dir, 'sample', 'bem', fname),
             str(tmpdir.join('foo', 'bem', fname)))
    outer_fname = tmpdir.join('foo', 'bem', 'outer_skull.surf')
    rr, tris = read_surface(outer_fname)
    tris = tris[:-1]
    write_surface(outer_fname, rr, tris[:-1], overwrite=True)
    with pytest.raises(RuntimeError, match='Surface outer skull is not compl'):
        make_bem_model('foo', None, subjects_dir=tmpdir)
    # Now get past this error to reach gh-6127 (not enough neighbor tris)
    rr_bad = np.concatenate([rr, np.mean(rr, axis=0, keepdims=True)], axis=0)
    write_surface(outer_fname, rr_bad, tris, overwrite=True)
    with pytest.raises(RuntimeError, match='Surface outer skull.*triangles'):
        make_bem_model('foo', None, subjects_dir=tmpdir)


@pytest.mark.slowtest
@testing.requires_testing_data
@pytest.mark.parametrize('cond, fname', [
    [(0.3,), fname_bem_sol_1],
    [(0.3, 0.006, 0.3), fname_bem_sol_3],
])
def test_bem_solution(tmpdir, cond, fname):
    """Test making a BEM solution from Python with I/O."""
    # test degenerate conditions
    surf = read_bem_surfaces(fname_bem_1)[0]
    pytest.raises(RuntimeError, _ico_downsample, surf, 10)  # bad dec grade
    s_bad = dict(tris=surf['tris'][1:], ntri=surf['ntri'] - 1, rr=surf['rr'])
    pytest.raises(RuntimeError, _ico_downsample, s_bad, 1)  # not isomorphic
    s_bad = dict(tris=surf['tris'].copy(), ntri=surf['ntri'],
                 rr=surf['rr'])  # bad triangulation
    s_bad['tris'][0] = [0, 0, 0]
    pytest.raises(RuntimeError, _ico_downsample, s_bad, 1)
    s_bad['id'] = 1
    pytest.raises(RuntimeError, _assert_complete_surface, s_bad)
    s_bad = dict(tris=surf['tris'], ntri=surf['ntri'], rr=surf['rr'].copy())
    s_bad['rr'][0] = 0.
    pytest.raises(RuntimeError, _get_ico_map, surf, s_bad)

    surfs = read_bem_surfaces(fname_bem_3)
    pytest.raises(RuntimeError, _assert_inside, surfs[0], surfs[1])  # outside
    surfs[0]['id'] = 100  # bad surfs
    pytest.raises(RuntimeError, _order_surfaces, surfs)
    surfs[1]['rr'] /= 1000.
    pytest.raises(RuntimeError, _check_surface_size, surfs[1])

    # actually test functionality
    fname_temp = op.join(str(tmpdir), 'temp-bem-sol.fif')
    # use a model and solution made in Python
    for model_type in ('python', 'c'):
        if model_type == 'python':
            model = make_bem_model('sample', conductivity=cond, ico=2,
                                   subjects_dir=subjects_dir)
        else:
            model = fname_bem_1 if len(cond) == 1 else fname_bem_3
    solution = make_bem_solution(model, verbose=True)
    solution_c = read_bem_solution(fname)
    _compare_bem_solutions(solution, solution_c)
    write_bem_solution(fname_temp, solution)
    solution_read = read_bem_solution(fname_temp)
    _compare_bem_solutions(solution, solution_c)
    _compare_bem_solutions(solution_read, solution_c)


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
    info['dig'][0]['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    pytest.raises(RuntimeError, fit_sphere_to_headshape, info, units='m')
    info['dig'][0]['coord_frame'] = FIFF.FIFFV_COORD_HEAD

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


@requires_nibabel()
@requires_freesurfer('mri_convert')
@testing.requires_testing_data
def test_make_flash_bem(tmpdir):
    """Test computing bem from flash images."""
    tmp = str(tmpdir)
    bemdir = op.join(subjects_dir, 'sample', 'bem')
    flash_path = op.join(subjects_dir, 'sample', 'mri', 'flash')

    for surf in ('inner_skull', 'outer_skull', 'outer_skin'):
        copy(op.join(bemdir, surf + '.surf'), tmp)
        copy(op.join(bemdir, surf + '.tri'), tmp)
    copy(op.join(bemdir, 'inner_skull_tmp.tri'), tmp)
    copy(op.join(bemdir, 'outer_skin_from_testing.surf'), tmp)

    # This function deletes the tri files at the end.
    try:
        make_flash_bem('sample', overwrite=True, subjects_dir=subjects_dir,
                       flash_path=flash_path)
        for surf in ('inner_skull', 'outer_skull', 'outer_skin'):
            coords, faces = read_surface(op.join(bemdir, surf + '.surf'))
            surf = 'outer_skin_from_testing' if surf == 'outer_skin' else surf
            coords_c, faces_c = read_surface(op.join(tmp, surf + '.surf'))
            assert_equal(0, faces.min())
            assert_equal(coords.shape[0], faces.max() + 1)
            assert_allclose(coords, coords_c)
            assert_allclose(faces, faces_c)
    finally:
        for surf in ('inner_skull', 'outer_skull', 'outer_skin'):
            remove(op.join(bemdir, surf + '.surf'))  # delete symlinks
            copy(op.join(tmp, surf + '.tri'), bemdir)  # return deleted tri
            copy(op.join(tmp, surf + '.surf'), bemdir)  # return moved surf
        copy(op.join(tmp, 'inner_skull_tmp.tri'), bemdir)
        copy(op.join(tmp, 'outer_skin_from_testing.surf'), bemdir)
    plt.close('all')


run_tests_if_main()
