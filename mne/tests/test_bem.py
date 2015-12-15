# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD 3 clause

import os.path as op
from copy import deepcopy

import numpy as np
from nose.tools import assert_raises, assert_true
from numpy.testing import assert_equal, assert_allclose

from mne import (make_bem_model, read_bem_surfaces, write_bem_surfaces,
                 make_bem_solution, read_bem_solution, write_bem_solution,
                 make_sphere_model, Transform)
from mne.preprocessing.maxfilter import fit_sphere_to_headshape
from mne.io.constants import FIFF
from mne.transforms import translation
from mne.datasets import testing
from mne.utils import run_tests_if_main, _TempDir, slow_test, catch_logging
from mne.bem import (_ico_downsample, _get_ico_map, _order_surfaces,
                     _assert_complete_surface, _assert_inside,
                     _check_surface_size, _bem_find_surface)
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
    """Helper to compare BEM surfaces"""
    names = ['id', 'nn', 'rr', 'coord_frame', 'tris', 'sigma', 'ntri', 'np']
    ignores = ['tri_cent', 'tri_nn', 'tri_area', 'neighbor_tri']
    for s0, s1 in zip(surfs_1, surfs_2):
        assert_equal(set(names), set(s0.keys()) - set(ignores))
        assert_equal(set(names), set(s1.keys()) - set(ignores))
        for name in names:
            assert_allclose(s0[name], s1[name], rtol=1e-3, atol=1e-6,
                            err_msg='Mismatch: "%s"' % name)


def _compare_bem_solutions(sol_a, sol_b):
    """Helper to compare BEM solutions"""
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
def test_io_bem():
    """Test reading and writing of bem surfaces and solutions
    """
    tempdir = _TempDir()
    temp_bem = op.join(tempdir, 'temp-bem.fif')
    assert_raises(ValueError, read_bem_surfaces, fname_raw)
    assert_raises(ValueError, read_bem_surfaces, fname_bem_3, s_id=10)
    surf = read_bem_surfaces(fname_bem_3, patch_stats=True)
    surf = read_bem_surfaces(fname_bem_3, patch_stats=False)
    write_bem_surfaces(temp_bem, surf[0])
    surf_read = read_bem_surfaces(temp_bem, patch_stats=False)
    _compare_bem_surfaces(surf, surf_read)

    assert_raises(RuntimeError, read_bem_solution, fname_bem_3)
    temp_sol = op.join(tempdir, 'temp-sol.fif')
    sol = read_bem_solution(fname_bem_sol_3)
    assert_true('BEM' in repr(sol))
    write_bem_solution(temp_sol, sol)
    sol_read = read_bem_solution(temp_sol)
    _compare_bem_solutions(sol, sol_read)
    sol = read_bem_solution(fname_bem_sol_1)
    assert_raises(RuntimeError, _bem_find_surface, sol, 3)


def test_make_sphere_model():
    """Test making a sphere model"""
    info = read_info(fname_raw)
    assert_raises(ValueError, make_sphere_model, 'foo', 'auto', info)
    assert_raises(ValueError, make_sphere_model, 'auto', 'auto', None)
    # here we just make sure it works -- the functionality is actually
    # tested more extensively e.g. in the forward and dipole code
    bem = make_sphere_model('auto', 'auto', info)
    assert_true('3 layers' in repr(bem))
    assert_true('Sphere ' in repr(bem))
    assert_true(' mm' in repr(bem))
    bem = make_sphere_model('auto', None, info)
    assert_true('no layers' in repr(bem))
    assert_true('Sphere ' in repr(bem))


@testing.requires_testing_data
def test_bem_model():
    """Test BEM model creation from Python with I/O"""
    tempdir = _TempDir()
    fname_temp = op.join(tempdir, 'temp-bem.fif')
    for kwargs, fname in zip((dict(), dict(conductivity=[0.3])),
                             [fname_bem_3, fname_bem_1]):
        model = make_bem_model('sample', ico=2, subjects_dir=subjects_dir,
                               **kwargs)
        model_c = read_bem_surfaces(fname)
        _compare_bem_surfaces(model, model_c)
        write_bem_surfaces(fname_temp, model)
        model_read = read_bem_surfaces(fname_temp)
        _compare_bem_surfaces(model, model_c)
        _compare_bem_surfaces(model_read, model_c)
    assert_raises(ValueError, make_bem_model, 'sample',  # bad conductivity
                  conductivity=[0.3, 0.006], subjects_dir=subjects_dir)


@slow_test
@testing.requires_testing_data
def test_bem_solution():
    """Test making a BEM solution from Python with I/O"""
    # test degenerate conditions
    surf = read_bem_surfaces(fname_bem_1)[0]
    assert_raises(RuntimeError, _ico_downsample, surf, 10)  # bad dec grade
    s_bad = dict(tris=surf['tris'][1:], ntri=surf['ntri'] - 1, rr=surf['rr'])
    assert_raises(RuntimeError, _ico_downsample, s_bad, 1)  # not isomorphic
    s_bad = dict(tris=surf['tris'].copy(), ntri=surf['ntri'],
                 rr=surf['rr'])  # bad triangulation
    s_bad['tris'][0] = [0, 0, 0]
    assert_raises(RuntimeError, _ico_downsample, s_bad, 1)
    s_bad['id'] = 1
    assert_raises(RuntimeError, _assert_complete_surface, s_bad)
    s_bad = dict(tris=surf['tris'], ntri=surf['ntri'], rr=surf['rr'].copy())
    s_bad['rr'][0] = 0.
    assert_raises(RuntimeError, _get_ico_map, surf, s_bad)

    surfs = read_bem_surfaces(fname_bem_3)
    assert_raises(RuntimeError, _assert_inside, surfs[0], surfs[1])  # outside
    surfs[0]['id'] = 100  # bad surfs
    assert_raises(RuntimeError, _order_surfaces, surfs)
    surfs[1]['rr'] /= 1000.
    assert_raises(RuntimeError, _check_surface_size, surfs[1])

    # actually test functionality
    tempdir = _TempDir()
    fname_temp = op.join(tempdir, 'temp-bem-sol.fif')
    # use a model and solution made in Python
    conductivities = [(0.3,), (0.3, 0.006, 0.3)]
    fnames = [fname_bem_sol_1, fname_bem_sol_3]
    for cond, fname in zip(conductivities, fnames):
        for model_type in ('python', 'c'):
            if model_type == 'python':
                model = make_bem_model('sample', conductivity=cond, ico=2,
                                       subjects_dir=subjects_dir)
            else:
                model = fname_bem_1 if len(cond) == 1 else fname_bem_3
        solution = make_bem_solution(model)
        solution_c = read_bem_solution(fname)
        _compare_bem_solutions(solution, solution_c)
        write_bem_solution(fname_temp, solution)
        solution_read = read_bem_solution(fname_temp)
        _compare_bem_solutions(solution, solution_c)
        _compare_bem_solutions(solution_read, solution_c)


def test_fit_sphere_to_headshape():
    """Test fitting a sphere to digitization points"""
    # Create points of various kinds
    rad = 90.  # mm
    big_rad = 120.
    center = np.array([0.5, -10., 40.])  # mm
    dev_trans = np.array([0., -0.005, -10.])
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
         'r': np.array([0.0, 0.0, 1.0])},

        # EEG points
        # Fz
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([0, .72, .69])},
        # F3
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([-.55, .67, .50])},
        # F4
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([.55, .67, .50])},
        # Cz
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([0.0, 0.0, 1.0])},
        # Pz
        {'coord_frame': FIFF.FIFFV_COORD_HEAD,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([0, -.72, .69])},
    ]
    for d in dig:
        d['r'] *= rad / 1000.
        d['r'] += center / 1000.

    # Device to head transformation (rotate .2 rad over X-axis)
    dev_head_t = Transform('meg', 'head', translation(*(dev_trans / 1000.)))

    info = {'dig': dig, 'dev_head_t': dev_head_t}

    # Degenerate conditions
    assert_raises(ValueError, fit_sphere_to_headshape, info,
                  dig_kinds=(FIFF.FIFFV_POINT_HPI,))
    info['dig'][0]['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    assert_raises(RuntimeError, fit_sphere_to_headshape, info)
    info['dig'][0]['coord_frame'] = FIFF.FIFFV_COORD_HEAD

    #  # Test with 4 points that match a perfect sphere
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA)
    r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds)
    kwargs = dict(rtol=1e-3, atol=1e-2)  # in mm
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, dev_center, **kwargs)

    # Test with all points
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA,
                 FIFF.FIFFV_POINT_EEG)
    kwargs = dict(rtol=1e-3, atol=1.)  # in mm
    r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds)
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, dev_center, **kwargs)

    # Test with some noisy EEG points only.
    dig_kinds = (FIFF.FIFFV_POINT_EEG,)
    r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds)
    kwargs = dict(rtol=1e-3, atol=10.)  # in mm
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, center, **kwargs)

    # Test big size
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA)
    info_big = deepcopy(info)
    for d in info_big['dig']:
        d['r'] -= center / 1000.
        d['r'] *= big_rad / rad
        d['r'] += center / 1000.
    with catch_logging() as log_file:
        r, oh, od = fit_sphere_to_headshape(info_big, dig_kinds=dig_kinds,
                                            verbose='warning')
    log_file = log_file.getvalue().strip()
    assert_equal(len(log_file.split('\n')), 1)
    assert_true(log_file.startswith('Estimated head size'))
    assert_allclose(oh, center, atol=1e-3)
    assert_allclose(r, big_rad, atol=1e-3)
    del info_big

    # Test offcenter
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA)
    info_shift = deepcopy(info)
    shift_center = np.array([0., -30, 0.])
    for d in info_shift['dig']:
        d['r'] -= center / 1000.
        d['r'] += shift_center / 1000.
    with catch_logging() as log_file:
        r, oh, od = fit_sphere_to_headshape(info_shift, dig_kinds=dig_kinds,
                                            verbose='warning')
    log_file = log_file.getvalue().strip()
    assert_equal(len(log_file.split('\n')), 1)
    assert_true('from head frame origin' in log_file)
    assert_allclose(oh, shift_center, atol=1e-3)
    assert_allclose(r, rad, atol=1e-3)


run_tests_if_main()
