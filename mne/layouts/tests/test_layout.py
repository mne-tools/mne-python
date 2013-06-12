import os.path as op
import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne.layouts import make_eeg_layout, make_grid_layout, read_layout
from mne.fiff import Raw
from mne.utils import _TempDir

fif_fname = op.join(op.dirname(__file__), '..', '..', 'fiff',
                   'tests', 'data', 'test_raw.fif')

lout_path = op.join(op.dirname(__file__), '..', '..', 'fiff',
                    'tests', 'data')

test_info = {'ch_names': ['ICA 001', 'ICA 002', 'EOG 061'],
 'chs': [{'cal': 1,
   'ch_name': 'ICA 001',
   'coil_trans': None,
   'coil_type': 0,
   'coord_Frame': 0,
   'eeg_loc': None,
   'kind': 502,
   'loc': np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
                   dtype=np.float32),
   'logno': 1,
   'range': 1.0,
   'scanno': 1,
   'unit': -1,
   'unit_mul': 0},
  {'cal': 1,
   'ch_name': 'ICA 002',
   'coil_trans': None,
   'coil_type': 0,
   'coord_Frame': 0,
   'eeg_loc': None,
   'kind': 502,
   'loc': np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
                    dtype=np.float32),
   'logno': 2,
   'range': 1.0,
   'scanno': 2,
   'unit': -1,
   'unit_mul': 0},
  {'cal': 0.002142000012099743,
   'ch_name': 'EOG 061',
   'coil_trans': None,
   'coil_type': 1,
   'coord_frame': 0,
   'eeg_loc': None,
   'kind': 202,
   'loc': np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
                    dtype=np.float32),
   'logno': 61,
   'range': 1.0,
   'scanno': 376,
   'unit': 107,
   'unit_mul': 0}],
   'nchan': 3}

tempdir = _TempDir()


def test_io_layout_lout():
    """Test IO with .lout files"""
    layout = read_layout('Vectorview-all', scale=False)
    layout.save(op.join(tempdir, 'foobar.lout'))
    layout_read = read_layout(op.join(tempdir, 'foobar.lout'), path='./',
                              scale=False)
    assert_array_almost_equal(layout.pos, layout_read.pos, decimal=2)
    assert_true(layout.names, layout_read.names)


def test_io_layout_lay():
    """Test IO with .lay files"""
    layout = read_layout('CTF151', scale=False)
    layout.save(op.join(tempdir, 'foobar.lay'))
    layout_read = read_layout(op.join(tempdir, 'foobar.lay'), path='./',
                              scale=False)
    assert_array_almost_equal(layout.pos, layout_read.pos, decimal=2)
    assert_true(layout.names, layout_read.names)


def test_make_eeg_layout():
    """ Test creation of EEG layout """
    tmp_name = 'foo'
    lout_name = 'test_raw'
    lout_orig = read_layout(kind=lout_name, path=lout_path)
    layout = make_eeg_layout(Raw(fif_fname).info)
    layout.save(op.join(tempdir, tmp_name + '.lout'))
    lout_new = read_layout(kind=tmp_name, path=tempdir)
    assert_array_equal(lout_new.kind, tmp_name)
    assert_array_equal(lout_orig.pos, lout_new.pos)
    assert_array_equal(lout_orig.names, lout_new.names)


def test_make_grid_layout():
    """ Test creation of grid layout """
    tmp_name = 'bar'
    lout_name = 'test_ica'
    lout_orig = read_layout(kind=lout_name, path=lout_path)
    layout = make_grid_layout(test_info)
    layout.save(op.join(tempdir, tmp_name + '.lout'))
    lout_new = read_layout(kind=tmp_name, path=tempdir)
    assert_array_equal(lout_new.kind, tmp_name)
    assert_array_equal(lout_orig.pos, lout_new.pos)
    assert_array_equal(lout_orig.names, lout_new.names)
