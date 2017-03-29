# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

from numpy import array
from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_false, assert_raises, assert_true

from mne.datasets import testing
from mne.io.tests import data_dir as fiff_data_dir
from mne.utils import (_TempDir, requires_freesurfer, requires_mayavi,
                       run_tests_if_main)

data_path = testing.data_path(download=False)
subjects_dir = os.path.join(data_path, 'subjects')
bem_path = os.path.join(subjects_dir, 'sample', 'bem', 'sample-1280-bem.fif')
inst_path = os.path.join(data_path, 'MEG', 'sample',
                         'sample_audvis_trunc_raw.fif')
fid_path = os.path.join(fiff_data_dir, 'fsaverage-fiducials.fif')


@testing.requires_testing_data
@requires_mayavi
def test_bem_source():
    """Test SurfaceSource"""
    from mne.gui._file_traits import SurfaceSource

    bem = SurfaceSource()
    assert_equal(bem.points.shape, (0, 3))
    assert_equal(bem.tris.shape, (0, 3))

    bem.file = bem_path
    assert_equal(bem.points.shape, (642, 3))
    assert_equal(bem.tris.shape, (1280, 3))


@testing.requires_testing_data
@requires_mayavi
def test_fiducials_source():
    """Test FiducialsSource"""
    from mne.gui._file_traits import FiducialsSource

    fid = FiducialsSource()
    fid.file = fid_path

    points = array([[-0.08061612, -0.02908875, -0.04131077],
                    [0.00146763, 0.08506715, -0.03483611],
                    [0.08436285, -0.02850276, -0.04127743]])
    assert_allclose(fid.points, points, 1e-6)

    fid.file = ''
    assert_equal(fid.points, None)


@testing.requires_testing_data
@requires_mayavi
def test_inst_source():
    """Test DigSource"""
    from mne.gui._file_traits import DigSource

    inst = DigSource()
    assert_equal(inst.inst_fname, '-')

    inst.file = inst_path
    assert_equal(inst.inst_dir, os.path.dirname(inst_path))

    lpa = array([[-7.13766068e-02, 0.00000000e+00, 5.12227416e-09]])
    nasion = array([[3.72529030e-09, 1.02605611e-01, 4.19095159e-09]])
    rpa = array([[7.52676800e-02, 0.00000000e+00, 5.58793545e-09]])
    assert_allclose(inst.lpa, lpa)
    assert_allclose(inst.nasion, nasion)
    assert_allclose(inst.rpa, rpa)


@testing.requires_testing_data
@requires_mayavi
def test_subject_source():
    """Test SubjectSelector"""
    from mne.gui._file_traits import MRISubjectSource

    mri = MRISubjectSource()
    mri.subjects_dir = subjects_dir
    assert_true('sample' in mri.subjects)
    mri.subject = 'sample'


@testing.requires_testing_data
@requires_mayavi
@requires_freesurfer
def test_subject_source_with_fsaverage():
    """Test SubjectSelector"""
    from mne.gui._file_traits import MRISubjectSource
    tempdir = _TempDir()

    mri = MRISubjectSource()
    assert_false(mri.can_create_fsaverage)
    assert_raises(RuntimeError, mri.create_fsaverage)

    mri.subjects_dir = tempdir
    assert_true(mri.can_create_fsaverage)
    mri.create_fsaverage()


run_tests_if_main()
