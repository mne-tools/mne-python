# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_false, assert_equal

from mne.datasets import testing
from mne.utils import _TempDir, requires_mayavi, run_tests_if_main

sample_path = testing.data_path(download=False)
subjects_dir = os.path.join(sample_path, 'subjects')


@testing.requires_testing_data
@requires_mayavi
def test_mri_model():
    """Test MRIHeadWithFiducialsModel Traits Model"""
    from mne.gui._fiducials_gui import MRIHeadWithFiducialsModel
    tempdir = _TempDir()
    tgt_fname = os.path.join(tempdir, 'test-fiducials.fif')

    model = MRIHeadWithFiducialsModel(subjects_dir=subjects_dir)
    model.subject = 'sample'
    assert_equal(model.default_fid_fname[-20:], "sample-fiducials.fif")
    assert_false(model.can_reset)
    assert_false(model.can_save)
    model.lpa = [[-1, 0, 0]]
    model.nasion = [[0, 1, 0]]
    model.rpa = [[1, 0, 0]]
    assert_false(model.can_reset)
    assert_true(model.can_save)

    bem_fname = os.path.basename(model.bem.file)
    assert_false(model.can_reset)
    assert_equal(bem_fname, 'sample-head-dense.fif')

    model.save(tgt_fname)
    assert_equal(model.fid_file, tgt_fname)

    # resetting the file should not affect the model's fiducials
    model.fid_file = ''
    assert_array_equal(model.lpa, [[-1, 0, 0]])
    assert_array_equal(model.nasion, [[0, 1, 0]])
    assert_array_equal(model.rpa, [[1, 0, 0]])

    # reset model
    model.lpa = [[0, 0, 0]]
    model.nasion = [[0, 0, 0]]
    model.rpa = [[0, 0, 0]]
    assert_array_equal(model.lpa, [[0, 0, 0]])
    assert_array_equal(model.nasion, [[0, 0, 0]])
    assert_array_equal(model.rpa, [[0, 0, 0]])

    # loading the file should assign the model's fiducials
    model.fid_file = tgt_fname
    assert_array_equal(model.lpa, [[-1, 0, 0]])
    assert_array_equal(model.nasion, [[0, 1, 0]])
    assert_array_equal(model.rpa, [[1, 0, 0]])

    # after changing from file model should be able to reset
    model.nasion = [[1, 1, 1]]
    assert_true(model.can_reset)
    model.reset = True
    assert_array_equal(model.nasion, [[0, 1, 0]])


run_tests_if_main()
