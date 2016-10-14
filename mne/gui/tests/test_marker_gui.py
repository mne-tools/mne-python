# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import warnings

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_false

from mne.io.kit.tests import data_dir as kit_data_dir
from mne.io.kit import read_mrk
from mne.utils import _TempDir, requires_mayavi, run_tests_if_main

mrk_pre_path = os.path.join(kit_data_dir, 'test_mrk_pre.sqd')
mrk_post_path = os.path.join(kit_data_dir, 'test_mrk_post.sqd')
mrk_avg_path = os.path.join(kit_data_dir, 'test_mrk.sqd')

warnings.simplefilter('always')


@requires_mayavi
def test_combine_markers_model():
    """Test CombineMarkersModel Traits Model"""
    from mne.gui._marker_gui import CombineMarkersModel, CombineMarkersPanel
    tempdir = _TempDir()
    tgt_fname = os.path.join(tempdir, 'test.txt')

    model = CombineMarkersModel()

    # set one marker file
    assert_false(model.mrk3.can_save)
    model.mrk1.file = mrk_pre_path
    assert_true(model.mrk3.can_save)
    assert_array_equal(model.mrk3.points, model.mrk1.points)

    # setting second marker file
    model.mrk2.file = mrk_pre_path
    assert_array_equal(model.mrk3.points, model.mrk1.points)

    # set second marker
    model.mrk2.clear = True
    model.mrk2.file = mrk_post_path
    assert_true(np.any(model.mrk3.points))
    points_interpolate_mrk1_mrk2 = model.mrk3.points

    # change interpolation method
    model.mrk3.method = 'Average'
    mrk_avg = read_mrk(mrk_avg_path)
    assert_array_equal(model.mrk3.points, mrk_avg)

    # clear second marker
    model.mrk2.clear = True
    assert_array_equal(model.mrk1.points, model.mrk3.points)

    # I/O
    model.mrk2.file = mrk_post_path
    model.mrk3.save(tgt_fname)
    mrk_io = read_mrk(tgt_fname)
    assert_array_equal(mrk_io, model.mrk3.points)

    # exlude an individual marker
    model.mrk1.use = [1, 2, 3, 4]
    assert_array_equal(model.mrk3.points[0], model.mrk2.points[0])
    assert_array_equal(model.mrk3.points[1:], mrk_avg[1:])

    # reset model
    model.clear = True
    model.mrk1.file = mrk_pre_path
    model.mrk2.file = mrk_post_path
    assert_array_equal(model.mrk3.points, points_interpolate_mrk1_mrk2)

    os.environ['_MNE_GUI_TESTING_MODE'] = 'true'
    try:
        with warnings.catch_warnings(record=True):  # traits warnings
            warnings.simplefilter('always')
            CombineMarkersPanel()
    finally:
        del os.environ['_MNE_GUI_TESTING_MODE']


run_tests_if_main()
