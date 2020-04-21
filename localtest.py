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
from mne.utils import (run_tests_if_main, catch_logging, requires_freesurfer,
                       requires_nibabel)
from mne.bem import (_ico_downsample, _get_ico_map, _order_surfaces,
                     _assert_complete_surface, _assert_inside,
                     _check_surface_size, _bem_find_surface, make_flash_bem,
                     make_watershed_bem)
from mne.surface import read_surface
from mne.io import read_info

fname_raw = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data',
                    'test_raw.fif')
subjects_dir = op.join(testing.data_path(download=False), 'subjects')
fname_bem_3 = op.join(subjects_dir, 'sample', 'bem',
                      'sample-320-320-320-bem.fif')
fname_bem_1 = op.join(subjects_dir, 'sample', 'bem', 'sample-320-bem.fif')
fname_bem_sol_3 = op.join(subjects_dir, 'sample', 'bem',
                          'sample-320-320-320-bem-sol.fif')
fname_bem_sol_1 = op.join(subjects_dir, 'sample', 'bem',
                          'sample-320-bem-sol.fif')


def test_make_watershed_bem(tmpdir):
    tmp = str(tmpdir)
    bemdir = op.join(subjects_dir, 'sample', 'bem')

    for surf in ('inner_skull', 'outer_skull', 'outer_skin'):
        copy(op.join(bemdir, surf + '.surf'), tmp)
    copy(op.join(bemdir, 'outer_skin_from_testing.surf'), tmp)

    try:
        make_watershed_bem('sample', subjects_dir=subjects_dir, overwrite=True)
        for surf in ('inner_skull', 'outer_skull', 'outer_skin'):
            surf_out = '%s.surf' % surf
            coords, faces, vol_info = read_surface(op.join(bemdir, surf_out),
                                                   read_metadata=True)
            surf = 'outer_skin_from_testing' if surf == 'outer_skin' else surf
            # should testing data include the computed watershed bems ?
            _, _, vol_info_c = read_surface(op.join(tmp, surf_out),
                                            read_metadata=True)
            # compare to the flash bems
            assert_allclose(
                [vol_info['xras'], vol_info['yras'], vol_info['zras']],
                [vol_info_c['xras'], vol_info_c['yras'], vol_info_c['zras']],
                atol=1e-4)
            assert_allclose(vol_info['cras'], vol_info_c['cras'], atol=1e-4)
            assert_equal(0, faces.min())
            assert_equal(coords.shape[0], faces.max() + 1)
    finally:
        for surf in ('inner_skull', 'outer_skull', 'outer_skin'):
            remove(op.join(bemdir, surf_out))  # delete symlinks
            copy(op.join(tmp, surf_out), bemdir)  # return moved surf
        copy(op.join(tmp, 'outer_skin_from_testing.surf'), bemdir)


tmpdir = op.join(testing.data_path(download=False), 'subjects', 'sample',
                 'bem', 'tmp')
test_make_watershed_bem(tmpdir)
