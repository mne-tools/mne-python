# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import os

import numpy as np
from numpy.testing import assert_array_equal

from mne.io.kit.tests import data_dir as kit_data_dir
from mne.io.kit import read_mrk
from mne.utils import requires_mayavi, traits_test, modified_env

mrk_pre_path = os.path.join(kit_data_dir, 'test_mrk_pre.sqd')
mrk_post_path = os.path.join(kit_data_dir, 'test_mrk_post.sqd')
mrk_avg_path = os.path.join(kit_data_dir, 'test_mrk.sqd')


@requires_mayavi
@traits_test
def test_combine_markers_model(tmp_path):
    """Test CombineMarkersModel Traits Model."""
    from mne.gui._marker_gui import CombineMarkersModel
    tempdir = str(tmp_path)
    tgt_fname = os.path.join(tempdir, 'test.txt')

    model = CombineMarkersModel()

    # set one marker file
    assert not model.mrk3.can_save
    model.mrk1.file = mrk_pre_path
    assert model.mrk3.can_save
    assert_array_equal(model.mrk3.points, model.mrk1.points)

    # setting second marker file
    model.mrk2.file = mrk_pre_path
    assert_array_equal(model.mrk3.points, model.mrk1.points)

    # set second marker
    model.mrk2.clear = True
    model.mrk2.file = mrk_post_path
    assert np.any(model.mrk3.points)
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

    # exclude an individual marker
    model.mrk1.use = [1, 2, 3, 4]
    assert_array_equal(model.mrk3.points[0], model.mrk2.points[0])
    assert_array_equal(model.mrk3.points[1:], mrk_avg[1:])

    # reset model
    model.clear = True
    model.mrk1.file = mrk_pre_path
    model.mrk2.file = mrk_post_path
    assert_array_equal(model.mrk3.points, points_interpolate_mrk1_mrk2)


@requires_mayavi
@traits_test
def test_combine_markers_panel(check_gui_ci):
    """Test CombineMarkersPanel."""
    from mne.gui._marker_gui import CombineMarkersPanel
    with modified_env(_MNE_GUI_TESTING_MODE='true'):
        CombineMarkersPanel()
