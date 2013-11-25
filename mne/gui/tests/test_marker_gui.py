# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_false

from mne.fiff.kit.tests import data_dir as kit_data_dir
from mne.fiff.kit import read_mrk
from mne.utils import _TempDir, requires_traits

mrk_pre_path = os.path.join(kit_data_dir, 'test_mrk_pre.sqd')
mrk_post_path = os.path.join(kit_data_dir, 'test_mrk_post.sqd')
mrk_avg_path = os.path.join(kit_data_dir, 'test_mrk.sqd')

tempdir = _TempDir()
tgt_fname = os.path.join(tempdir, 'test.txt')


@requires_traits
def test_combine_markers_model():
    """Test CombineMarkersModel Traits Model"""
    from mne.gui._marker_gui import CombineMarkersModel

    model = CombineMarkersModel()
    assert_false(model.mrk3.can_save)
    model.mrk1.file = mrk_pre_path
    assert_true(model.mrk3.can_save)
    assert_array_equal(model.mrk1.points, model.mrk3.points)

    model.mrk2.file = mrk_pre_path
    assert_array_equal(model.mrk1.points, model.mrk3.points)

    model.mrk2._clear_fired()
    model.mrk2.file = mrk_post_path
    assert_true(np.any(model.mrk3.points))

    model.mrk3.method = 'Average'
    mrk_avg = read_mrk(mrk_avg_path)
    assert_array_equal(model.mrk3.points, mrk_avg)

    model.mrk3.save(tgt_fname)
    mrk_io = read_mrk(tgt_fname)
    assert_array_equal(mrk_io, model.mrk3.points)
