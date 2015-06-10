# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD 3 clause

import os.path as op
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose

from mne import (make_bem_model, read_bem_surfaces, write_bem_surfaces,
                 make_bem_solution, read_bem_solution, write_bem_solution)
from mne.preprocessing.maxfilter import fit_sphere_to_headshape
from mne.io.constants import FIFF
from mne.transforms import rotation
from mne.datasets import testing
from mne.utils import run_tests_if_main, _TempDir, slow_test

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
        assert_allclose(sol_a[key], sol_b[key], rtol=1e-4, atol=1e-6,
                        err_msg='Mismatch: %s' % key)


@testing.requires_testing_data
def test_io_bem_surfaces():
    """Test reading and writing of bem surfaces
    """
    tempdir = _TempDir()
    surf = read_bem_surfaces(fname_bem_3, patch_stats=True)
    surf = read_bem_surfaces(fname_bem_3, patch_stats=False)
    write_bem_surfaces(op.join(tempdir, 'bem_surf.fif'), surf[0])
    surf_read = read_bem_surfaces(op.join(tempdir, 'bem_surf.fif'),
                                  patch_stats=False)
    _compare_bem_surfaces(surf, surf_read)


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


@slow_test
@testing.requires_testing_data
def test_bem_solution():
    """Test making a BEM solution from Python with I/O"""
    tempdir = _TempDir()
    fname_temp = op.join(tempdir, 'temp-bem-sol.fif')
    # use a model and solution made in Python
    conductivities = [(0.3,), ]  # (0.3, 0.006, 0.3)]
    fnames = [fname_bem_sol_1, ]  # fname_bem_sol_3]
    for cond, fname in zip(conductivities, fnames):
        model = make_bem_model('sample', conductivity=cond, ico=2)
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
    dig = [
        # Left auricular
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'ident': FIFF.FIFFV_POINT_LPA,
         'kind': FIFF.FIFFV_POINT_CARDINAL,
         'r': np.array([-1.0, 0.0, 0.0])},
        # Nasion
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'ident': FIFF.FIFFV_POINT_NASION,
         'kind': FIFF.FIFFV_POINT_CARDINAL,
         'r': np.array([0.0, 1.0, 0.0])},
        # Right auricular
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'ident': FIFF.FIFFV_POINT_RPA,
         'kind': FIFF.FIFFV_POINT_CARDINAL,
         'r': np.array([1.0, 0.0, 0.0])},

        # Top of the head (extra point)
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EXTRA,
         'r': np.array([0.0, 0.0, 1.0])},

        # EEG points
        # Fz
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([0, .72, .69])},
        # F3
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([-.55, .67, .50])},
        # F4
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([.55, .67, .50])},
        # Cz
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([0.0, 0.0, 1.0])},
        # Pz
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([0, -.72, .69])},
    ]

    # Device to head transformation (rotate .2 rad over X-axis)
    dev_head_t = {
        'from': FIFF.FIFFV_COORD_DEVICE,
        'to': FIFF.FIFFV_COORD_HEAD,
        'trans': rotation(x=0.2),
    }

    info = {'dig': dig, 'dev_head_t': dev_head_t}

    #  # Test with 4 points that match a perfect sphere
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA)
    r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds)
    assert_almost_equal(r / 1000, 1.0, decimal=10)
    assert_almost_equal(oh / 1000, [0.0, 0.0, 0.0], decimal=10)
    assert_almost_equal(od / 1000, [0.0, 0.0, 0.0], decimal=10)

    # Test with all points. Digitization points are no longer perfect, so
    # allow for a wider margin of error.
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA,
                 FIFF.FIFFV_POINT_EXTRA)
    r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds)
    assert_almost_equal(r / 1000, 1.0, decimal=3)
    assert_almost_equal(oh / 1000, [0.0, 0.0, 0.0], decimal=3)
    assert_almost_equal(od / 1000, [0.0, 0.0, 0.0], decimal=3)

    # Test with some noisy EEG points only.
    dig_kinds = (FIFF.FIFFV_POINT_EEG,)
    r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds)
    assert_almost_equal(r / 1000, 1.0, decimal=2)
    assert_almost_equal(oh / 1000, [0.0, 0.0, 0.0], decimal=2)
    assert_almost_equal(od / 1000, [0.0, 0.0, 0.0], decimal=2)


run_tests_if_main()
