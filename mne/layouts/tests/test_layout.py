from __future__ import print_function
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import copy
import os.path as op
import warnings

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from nose.tools import assert_true, assert_raises

from mne.layouts import (make_eeg_layout, make_grid_layout, read_layout,
                         find_layout)
from mne.layouts.layout import _box_size

from mne import pick_types, pick_info
from mne.io import Raw
from mne.io import read_raw_kit
from mne.utils import _TempDir

warnings.simplefilter('always')

fif_fname = op.join(op.dirname(__file__), '..', '..', 'io',
                   'tests', 'data', 'test_raw.fif')

lout_path = op.join(op.dirname(__file__), '..', '..', 'io',
                    'tests', 'data')

bti_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'bti',
                  'tests', 'data')

fname_ctf_raw = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                        'data', 'test_ctf_comp_raw.fif')

fname_kit_157 = op.join(op.dirname(__file__), '..', '..', 'io', 'kit',
                        'tests', 'data', 'test.sqd')

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

    print(layout)  # test repr


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
    lout_new = read_layout(kind=tmp_name, path=tempdir, scale=False)
    assert_array_equal(lout_new.kind, tmp_name)
    assert_allclose(layout.pos, lout_new.pos, atol=0.1)
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

def test_find_layout():
    """Test finding layout"""
    assert_raises(ValueError, find_layout, test_info, ch_type='meep')

    sample_info = Raw(fif_fname).info
    grads = pick_types(sample_info, meg='grad')
    sample_info2 = pick_info(sample_info, grads)

    mags = pick_types(sample_info, meg='mag')
    sample_info3 = pick_info(sample_info, mags)

    # mock new convention
    sample_info4 = copy.deepcopy(sample_info)
    for ii, name in enumerate(sample_info4['ch_names']):
        new = name.replace(' ', '')
        sample_info4['ch_names'][ii] = new
        sample_info4['chs'][ii]['ch_name'] = new

    mags = pick_types(sample_info, meg=False, eeg=True)
    sample_info5 = pick_info(sample_info, mags)

    lout = find_layout(sample_info, ch_type=None)
    assert_true(lout.kind == 'Vectorview-all')
    assert_true(all(' ' in k for k in lout.names))

    lout = find_layout(sample_info2, ch_type='meg')
    assert_true(lout.kind == 'Vectorview-all')

    # test new vector-view
    lout = find_layout(sample_info4, ch_type=None)
    assert_true(lout.kind == 'Vectorview-all')
    assert_true(all(not ' ' in k for k in lout.names))

    lout = find_layout(sample_info, ch_type='grad')
    assert_true(lout.kind == 'Vectorview-grad')
    lout = find_layout(sample_info2)
    assert_true(lout.kind == 'Vectorview-grad')
    lout = find_layout(sample_info2, ch_type='grad')
    assert_true(lout.kind == 'Vectorview-grad')
    lout = find_layout(sample_info2, ch_type='meg')
    assert_true(lout.kind == 'Vectorview-all')


    lout = find_layout(sample_info, ch_type='mag')
    assert_true(lout.kind == 'Vectorview-mag')
    lout = find_layout(sample_info3)
    assert_true(lout.kind == 'Vectorview-mag')
    lout = find_layout(sample_info3, ch_type='mag')
    assert_true(lout.kind == 'Vectorview-mag')
    lout = find_layout(sample_info3, ch_type='meg')
    assert_true(lout.kind == 'Vectorview-all')
    #
    lout = find_layout(sample_info, ch_type='eeg')
    assert_true(lout.kind == 'EEG')
    lout = find_layout(sample_info5)
    assert_true(lout.kind == 'EEG')
    lout = find_layout(sample_info5, ch_type='eeg')
    assert_true(lout.kind == 'EEG')
    # no common layout, 'meg' option not supported

    fname_bti_raw = op.join(bti_dir, 'exported4D_linux_raw.fif')
    lout = find_layout(Raw(fname_bti_raw).info)
    assert_true(lout.kind == 'magnesWH3600')

    lout = find_layout(Raw(fname_ctf_raw).info)
    assert_true(lout.kind == 'CTF-275')

    lout = find_layout(read_raw_kit(fname_kit_157).info)
    assert_true(lout.kind == 'KIT-157')

    sample_info5['dig'] = []
    assert_raises(RuntimeError, find_layout, sample_info5)

def test_box_size():
    """Test calculation of box sizes."""
    # No points. Box size should be 1,1.
    assert_allclose(_box_size([]), (1.0, 1.0))

    # Create one point. Box size should be 1,1.
    point = [(0,0)]
    assert_allclose(_box_size(point), (1.0, 1.0))

    # Create two points. Box size should be 0.5,1.
    points = [(0.25, 0.5), (0.75, 0.5)]
    assert_allclose(_box_size(points), (0.5, 1.0))

    # Create three points. Box size should be (0.5, 0.5).
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points), (0.5, 0.5))

    # Create a grid of points. Box size should be (0.1, 0.1).
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 11), np.linspace(-0.5, 0.5, 11))
    x, y = x.ravel(), y.ravel()
    assert_allclose(_box_size(np.c_[x,y]), (0.1, 0.1))

    # Create a random set of points. This should never break the function.
    points = np.random.rand(100, 2)
    width, height = _box_size(points)
    assert_true(width != None)
    assert_true(height != None)

    # Test specifying an existing width.
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points, width=0.4), (0.4, 0.5))

    # Test specifying an existing width that has influence on the calculated
    # height.
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points, width=0.2), (0.2, 1.0))

    # Test specifying an existing height.
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points, height=0.4), (0.5, 0.4))

    # Test specifying an existing height that has influence on the calculated
    # width.
    points = [(0.25, 0.25), (0.75, 0.45), (0.5, 0.75)]
    assert_allclose(_box_size(points, height=0.1), (1.0, 0.1))

    # Test specifying both width and height. The function should simply return
    # these.
    points = [(0.25, 0.25), (0.75, 0.45), (0.5, 0.75)]
    assert_array_equal(_box_size(points, width=0.1, height=0.1), (0.1, 0.1))

    # Test specifying a width that will cause unfixable horizontal overlap and
    # essentially breaks the function (height will be 0).
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_array_equal(_box_size(points, width=1), (1, 0))

    # Test adding some padding.
    # Create three points. Box size should be a little less than (0.5, 0.5).
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points, padding=0.9), (0.9*0.5, 0.9*0.5))
