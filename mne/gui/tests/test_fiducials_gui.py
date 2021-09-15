# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import os
import os.path as op

from numpy.testing import assert_array_equal

from mne.utils import requires_mayavi, traits_test


@requires_mayavi
@traits_test
def test_mri_model(subjects_dir_tmp):
    """Test MRIHeadWithFiducialsModel Traits Model."""
    from mne.gui._fiducials_gui import MRIHeadWithFiducialsModel
    tgt_fname = op.join(subjects_dir_tmp, 'test-fiducials.fif')

    # Remove the two files that will make the fiducials okay via MNI estimation
    os.remove(op.join(subjects_dir_tmp, 'sample', 'bem',
                      'sample-fiducials.fif'))
    os.remove(op.join(subjects_dir_tmp, 'sample', 'mri', 'transforms',
                      'talairach.xfm'))

    model = MRIHeadWithFiducialsModel(subjects_dir=subjects_dir_tmp)
    model.subject = 'sample'
    assert model.default_fid_fname[-20:] == "sample-fiducials.fif"
    assert not model.can_reset
    assert not model.can_save
    model.lpa = [[-1, 0, 0]]
    model.nasion = [[0, 1, 0]]
    model.rpa = [[1, 0, 0]]
    assert not model.can_reset
    assert model.can_save

    bem_fname = op.basename(model.bem_high_res.file)
    assert not model.can_reset
    assert bem_fname == 'sample-head-dense.fif'

    model.save(tgt_fname)
    assert model.fid_file == tgt_fname

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
    assert model.can_reset
    model.reset = True
    assert_array_equal(model.nasion, [[0, 1, 0]])
