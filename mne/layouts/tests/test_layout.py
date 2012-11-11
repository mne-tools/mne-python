import os.path as op
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne.layouts import make_eeg_layout, make_grid_layout, read_layout
from mne.fiff import Raw

fif_fname = op.join(op.dirname(__file__), '..', '..', 'fiff',
                   'tests', 'data', 'test_raw.fif')

ica_fif_fname = op.join(op.dirname(__file__), '..', '..', 'fiff',
                   'tests', 'data', 'test_ica_raw.fif')

lout_path = op.join(op.dirname(__file__), '..', '..', 'fiff',
                    'tests', 'data')


def test_io_layout():
    """Test IO with .lout files"""
    layout = read_layout('Vectorview-all', scale=False)
    layout.save('foobar.lout')
    layout_read = read_layout('foobar.lout', path='./', scale=False)
    assert_array_almost_equal(layout.pos, layout_read.pos, decimal=2)
    assert_true(layout.names, layout_read.names)


def test_make_eeg_layout():
    """ Test creation of EEG layout """
    tmp_name = 'foo'
    lout_name = 'test_raw'
    lout_orig = read_layout(kind=lout_name, path=lout_path)
    layout = make_eeg_layout(Raw(fif_fname).info)
    layout.save(tmp_name + '.lout')
    lout_new = read_layout(kind=tmp_name, path='.')
    assert_array_equal(lout_new.kind, tmp_name)
    assert_array_equal(lout_orig.pos, lout_new.pos)
    assert_array_equal(lout_orig.names, lout_new.names)


def test_make_grid_layout():
    """ Test creation of grid layout """
    tmp_name = 'bar'
    lout_name = 'test_ica'
    lout_orig = read_layout(kind=lout_name, path=lout_path)
    layout = make_grid_layout(Raw(ica_fif_fname).info)
    layout.save(tmp_name + '.lout')
    lout_new = read_layout(kind=tmp_name, path='.')
    assert_array_equal(lout_new.kind, tmp_name)
    assert_array_equal(lout_orig.pos, lout_new.pos)
    assert_array_equal(lout_orig.names, lout_new.names)
