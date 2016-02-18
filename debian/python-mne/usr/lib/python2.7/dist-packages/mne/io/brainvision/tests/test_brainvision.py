"""Data Equivalence Tests"""
from __future__ import print_function

# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import inspect

from nose.tools import assert_equal, assert_raises, assert_true
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)

from mne.utils import _TempDir, run_tests_if_main
from mne import pick_types, concatenate_raws, find_events
from mne.io.constants import FIFF
from mne.io import Raw, read_raw_brainvision

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
vhdr_path = op.join(data_dir, 'test.vhdr')
vmrk_path = op.join(data_dir, 'test.vmrk')
vhdr_highpass_path = op.join(data_dir, 'test_highpass.vhdr')
montage = op.join(data_dir, 'test.hpts')
eeg_bin = op.join(data_dir, 'test_bin_raw.fif')
eog = ['HL', 'HR', 'Vb']


def test_brainvision_data_filters():
    """Test reading raw Brain Vision files
    """
    raw = read_raw_brainvision(vhdr_highpass_path, montage, eog=eog,
                               preload=True)
    assert_equal(raw.info['highpass'], 0.1)
    assert_equal(raw.info['lowpass'], 250.)
    raw.info["lowpass"] = None
    raw.filter(1, 30)


def test_brainvision_data():
    """Test reading raw Brain Vision files
    """
    assert_raises(IOError, read_raw_brainvision, vmrk_path)
    assert_raises(ValueError, read_raw_brainvision, vhdr_path, montage,
                  preload=True, scale="foo")
    raw_py = read_raw_brainvision(vhdr_path, montage, eog=eog, preload=True)
    raw_py.load_data()  # currently does nothing
    assert_true('RawBrainVision' in repr(raw_py))

    assert_equal(raw_py.info['highpass'], 0.)
    assert_equal(raw_py.info['lowpass'], 250.)

    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, times_py = raw_py[picks]

    print(raw_py)  # to test repr
    print(raw_py.info)  # to test Info repr

    # compare with a file that was generated using MNE-C
    raw_bin = Raw(eeg_bin, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_bin, times_bin = raw_bin[picks]

    assert_array_almost_equal(data_py, data_bin)
    assert_array_almost_equal(times_py, times_bin)

    # Make sure EOG channels are marked correctly
    raw_py = read_raw_brainvision(vhdr_path, montage, eog=eog,
                                  preload=True)
    for ch in raw_py.info['chs']:
        if ch['ch_name'] in eog:
            assert_equal(ch['kind'], FIFF.FIFFV_EOG_CH)
        elif ch['ch_name'] == 'STI 014':
            assert_equal(ch['kind'], FIFF.FIFFV_STIM_CH)
        elif ch['ch_name'] in raw_py.info['ch_names']:
            assert_equal(ch['kind'], FIFF.FIFFV_EEG_CH)
        else:
            raise RuntimeError("Unknown Channel: %s" % ch['ch_name'])

    # Make sure concatenation works
    raw_concat = concatenate_raws([raw_py.copy(), raw_py])
    assert_equal(raw_concat.n_times, 2 * raw_py.n_times)


def test_events():
    """Test reading and modifying events"""
    tempdir = _TempDir()

    # check that events are read and stim channel is synthesized correcly
    raw = read_raw_brainvision(vhdr_path, eog=eog, preload=True)
    events = raw.get_brainvision_events()
    assert_array_equal(events, [[487, 1, 253],
                                [497, 1, 255],
                                [1770, 1, 254],
                                [1780, 1, 255],
                                [3253, 1, 254],
                                [3263, 1, 255],
                                [4936, 1, 253],
                                [4946, 1, 255],
                                [6000, 1, 255],
                                [6620, 1, 254],
                                [6630, 1, 255]])

    # check that events are read and stim channel is synthesized correcly and
    # response triggers are shifted like they're supposed to be.
    raw = read_raw_brainvision(vhdr_path, eog=eog, preload=True,
                               response_trig_shift=1000)
    events = raw.get_brainvision_events()
    assert_array_equal(events, [[487, 1, 253],
                                [497, 1, 255],
                                [1770, 1, 254],
                                [1780, 1, 255],
                                [3253, 1, 254],
                                [3263, 1, 255],
                                [4936, 1, 253],
                                [4946, 1, 255],
                                [6000, 1, 1255],
                                [6620, 1, 254],
                                [6630, 1, 255]])

    # check that events are read and stim channel is synthesized correcly and
    # response triggers are ignored.
    raw = read_raw_brainvision(vhdr_path, eog=eog, preload=True,
                               response_trig_shift=None)
    events = raw.get_brainvision_events()
    assert_array_equal(events, [[487, 1, 253],
                                [497, 1, 255],
                                [1770, 1, 254],
                                [1780, 1, 255],
                                [3253, 1, 254],
                                [3263, 1, 255],
                                [4936, 1, 253],
                                [4946, 1, 255],
                                [6620, 1, 254],
                                [6630, 1, 255]])

    assert_raises(TypeError, read_raw_brainvision, vhdr_path, eog=eog,
                  preload=True, response_trig_shift=0.1)
    assert_raises(TypeError, read_raw_brainvision, vhdr_path, eog=eog,
                  preload=True, response_trig_shift=np.nan)

    mne_events = find_events(raw, stim_channel='STI 014')
    assert_array_equal(events[:, [0, 2]], mne_events[:, [0, 2]])

    # modify events and check that stim channel is updated
    index = events[:, 2] == 255
    events = events[index]
    raw.set_brainvision_events(events)
    mne_events = find_events(raw, stim_channel='STI 014')
    assert_array_equal(events[:, [0, 2]], mne_events[:, [0, 2]])

    # remove events
    nchan = raw.info['nchan']
    ch_name = raw.info['chs'][-2]['ch_name']
    events = np.empty((0, 3))
    raw.set_brainvision_events(events)
    assert_equal(raw.info['nchan'], nchan)
    assert_equal(len(raw._data), nchan)
    assert_equal(raw.info['chs'][-2]['ch_name'], ch_name)
    assert_equal(len(find_events(raw, 'STI 014')), 0)
    assert_allclose(raw[-1][0], 0.)
    fname = op.join(tempdir, 'evt_raw.fif')
    raw.save(fname)

    # add events back in
    events = [[10, 1, 2]]
    raw.set_brainvision_events(events)
    assert_equal(raw.info['nchan'], nchan)
    assert_equal(len(raw._data), nchan)
    assert_equal(raw.info['chs'][-1]['ch_name'], 'STI 014')


def test_read_segment():
    """Test writing raw eeg files when preload is False
    """
    tempdir = _TempDir()
    raw1 = read_raw_brainvision(vhdr_path, eog=eog, preload=False)
    raw1_file = op.join(tempdir, 'test1-raw.fif')
    raw1.save(raw1_file, overwrite=True)
    raw11 = Raw(raw1_file, preload=True)
    data1, times1 = raw1[:, :]
    data11, times11 = raw11[:, :]
    assert_array_almost_equal(data1, data11, 8)
    assert_array_almost_equal(times1, times11)
    assert_equal(sorted(raw1.info.keys()), sorted(raw11.info.keys()))

    raw2 = read_raw_brainvision(vhdr_path, eog=eog, preload=True)
    raw2_file = op.join(tempdir, 'test2-raw.fif')
    raw2.save(raw2_file, overwrite=True)
    data2, times2 = raw2[:, :]
    assert_array_equal(data1, data2)
    assert_array_equal(times1, times2)

    raw1 = Raw(raw1_file, preload=True)
    raw2 = Raw(raw2_file, preload=True)
    assert_array_equal(raw1._data, raw2._data)

    # save with buffer size smaller than file
    raw3_file = op.join(tempdir, 'test3-raw.fif')
    raw3 = read_raw_brainvision(vhdr_path, eog=eog)
    raw3.save(raw3_file, buffer_size_sec=2)
    raw3 = Raw(raw3_file, preload=True)
    assert_array_equal(raw3._data, raw1._data)

run_tests_if_main()
