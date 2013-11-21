from math import pi
import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_equal, assert_equal, assert_allclose

from mne.datasets import sample
from mne import read_trans, write_trans
from mne.utils import _TempDir
from mne.transforms import (_get_mri_head_t_from_trans_file, invert_transform,
                            rotation, rotation3d, rotation_angles)

data_path = sample.data_path(download=False)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
fname_trans = op.join(op.split(__file__)[0], '..', 'fiff', 'tests',
                      'data', 'sample-audvis-raw-trans.txt')

tempdir = _TempDir()


@sample.requires_sample_data
def test_get_mri_head_t():
    """Test converting '-trans.txt' to '-trans.fif'"""
    trans = read_trans(fname)
    trans = invert_transform(trans)  # starts out as head->MRI, so invert
    trans_2 = _get_mri_head_t_from_trans_file(fname_trans)
    assert_equal(trans['from'], trans_2['from'])
    assert_equal(trans['to'], trans_2['to'])
    assert_allclose(trans['trans'], trans_2['trans'], rtol=1e-5, atol=1e-5)


@sample.requires_sample_data
def test_io_trans():
    """Test reading and writing of trans files
    """
    info0 = read_trans(fname)
    fname1 = op.join(tempdir, 'test-trans.fif')
    write_trans(fname1, info0)
    info1 = read_trans(fname1)

    # check all properties
    assert_true(info0['from'] == info1['from'])
    assert_true(info0['to'] == info1['to'])
    assert_array_equal(info0['trans'], info1['trans'])
    for d0, d1 in zip(info0['dig'], info1['dig']):
        assert_array_equal(d0['r'], d1['r'])
        for name in ['kind', 'ident', 'coord_frame']:
            assert_true(d0[name] == d1[name])


def test_rotation():
    """Test conversion between rotation angles and transformation matrix"""
    tests = [(0, 0, 1), (.5, .5, .5), (pi, 0, -1.5)]
    for rot in tests:
        x, y, z = rot
        m = rotation3d(x, y, z)
        m4 = rotation(x, y, z)
        assert_array_equal(m, m4[:3, :3])
        back = rotation_angles(m)
        assert_equal(back, rot)
        back4 = rotation_angles(m4)
        assert_equal(back4, rot)
