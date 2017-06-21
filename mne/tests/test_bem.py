# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD 3 clause

from copy import deepcopy
from os import remove
import os.path as op
from shutil import copy
import warnings

import numpy as np
from nose.tools import assert_raises, assert_true
from numpy.testing import assert_equal, assert_allclose

from mne import (make_bem_model, read_bem_surfaces, write_bem_surfaces,
                 make_bem_solution, read_bem_solution, write_bem_solution,
                 make_sphere_model, Transform, Info)
from mne.preprocessing.maxfilter import fit_sphere_to_headshape
from mne.io.constants import FIFF
from mne.transforms import translation
from mne.datasets import testing
from mne.utils import (run_tests_if_main, _TempDir, slow_test, catch_logging,
                       requires_freesurfer)
from mne.bem import (_ico_downsample, _get_ico_map, _order_surfaces,
                     _assert_complete_surface, _assert_inside,
                     _check_surface_size, _bem_find_surface, make_flash_bem)
from mne.surface import read_surface
from mne.io import read_info

import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')

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
    """Test reading and writing of bem surfaces and solutions"""
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
    assert_raises(ValueError, make_sphere_model, 'auto', 'auto', info,
                  relative_radii=(), sigmas=())
    assert_raises(ValueError, make_sphere_model, 'auto', 'auto', info,
                  relative_radii=(1,))  # wrong number of radii
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
        d['r'] *= rad
        d['r'] += center

    # Device to head transformation (rotate .2 rad over X-axis)
    dev_head_t = Transform('meg', 'head', translation(*(dev_trans)))
    info = Info(dig=dig, dev_head_t=dev_head_t)

    # Degenerate conditions
    assert_raises(ValueError, fit_sphere_to_headshape, info,
                  dig_kinds=(FIFF.FIFFV_POINT_HPI,))
    assert_raises(ValueError, fit_sphere_to_headshape, info,
                  dig_kinds='foo', units='m')
    info['dig'][0]['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    assert_raises(RuntimeError, fit_sphere_to_headshape, info, units='m')
    info['dig'][0]['coord_frame'] = FIFF.FIFFV_COORD_HEAD

    #  # Test with 4 points that match a perfect sphere
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA)
    with warnings.catch_warnings(record=True):  # not enough points
        r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds,
                                            units='m')
    kwargs = dict(rtol=1e-3, atol=1e-5)
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, dev_center, **kwargs)

    # Test with all points
    dig_kinds = ('cardinal', FIFF.FIFFV_POINT_EXTRA, 'eeg')
    kwargs = dict(rtol=1e-3, atol=1e-3)
    with warnings.catch_warnings(record=True):  # not enough points
        r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds,
                                            units='m')
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, dev_center, **kwargs)

    # Test with some noisy EEG points only.
    dig_kinds = 'eeg'
    with warnings.catch_warnings(record=True):  # not enough points
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
    with warnings.catch_warnings(record=True):  # fit
        with catch_logging() as log_file:
            r, oh, od = fit_sphere_to_headshape(info_big, dig_kinds=dig_kinds,
                                                verbose='warning', units='mm')
    log_file = log_file.getvalue().strip()
    assert_equal(len(log_file.split('\n')), 2)
    assert_true('Estimated head size' in log_file)
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
    with warnings.catch_warnings(record=True):
        with catch_logging() as log_file:
            r, oh, od = fit_sphere_to_headshape(
                info_shift, dig_kinds=dig_kinds, verbose='warning', units='m')
    log_file = log_file.getvalue().strip()
    assert_equal(len(log_file.split('\n')), 2)
    assert_true('from head frame origin' in log_file)
    assert_allclose(oh, shift_center, atol=1e-6)
    assert_allclose(r, rad, atol=1e-6)

    # Test "auto" mode (default)
    # Should try "extra", fail, and go on to EEG
    with warnings.catch_warnings(record=True):  # not enough points
        r, oh, od = fit_sphere_to_headshape(info, units='m')
    kwargs = dict(rtol=1e-3, atol=1e-3)
    assert_allclose(r, rad, **kwargs)
    assert_allclose(oh, center, **kwargs)
    assert_allclose(od, dev_center, **kwargs)
    with warnings.catch_warnings(record=True):  # not enough points
        r2, oh2, od2 = fit_sphere_to_headshape(info, units='m')
    assert_allclose(r, r2, atol=1e-7)
    assert_allclose(oh, oh2, atol=1e-7)
    assert_allclose(od, od2, atol=1e-7)
    # this one should pass, 1 EXTRA point and 3 EEG (but the fit is terrible)
    info = Info(dig=dig[:7], dev_head_t=dev_head_t)
    with warnings.catch_warnings(record=True):  # bad fit
        r, oh, od = fit_sphere_to_headshape(info, units='m')
    # this one should fail, 1 EXTRA point and 3 EEG (but the fit is terrible)
    info = Info(dig=dig[:6], dev_head_t=dev_head_t)
    assert_raises(ValueError, fit_sphere_to_headshape, info, units='m')
    assert_raises(TypeError, fit_sphere_to_headshape, 1, units='m')


@requires_freesurfer
@testing.requires_testing_data
def test_make_flash_bem():
    """Test computing bem from flash images."""
    import matplotlib.pyplot as plt
    tmp = _TempDir()
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
