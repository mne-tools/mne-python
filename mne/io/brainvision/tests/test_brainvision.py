"""Data Equivalence Tests"""
from __future__ import print_function

# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import inspect
import warnings

from nose.tools import assert_equal, assert_raises, assert_true
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)

from mne.utils import _TempDir, run_tests_if_main
from mne import pick_types, find_events
from mne.io.constants import FIFF
from mne.io import Raw, read_raw_brainvision
from mne.io.tests.test_raw import _test_raw_reader

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
vhdr_path = op.join(data_dir, 'test.vhdr')
vmrk_path = op.join(data_dir, 'test.vmrk')
vhdr_v2_path = op.join(data_dir, 'testv2.vhdr')
vmrk_v2_path = op.join(data_dir, 'testv2.vmrk')
vhdr_highpass_path = op.join(data_dir, 'test_highpass.vhdr')
montage = op.join(data_dir, 'test.hpts')
eeg_bin = op.join(data_dir, 'test_bin_raw.fif')
eog = ['HL', 'HR', 'Vb']

warnings.simplefilter('always')


def test_brainvision_data_filters():
    """Test reading raw Brain Vision files
    """
    with warnings.catch_warnings(record=True) as w:  # event parsing
        raw = _test_raw_reader(
            read_raw_brainvision, vhdr_fname=vhdr_highpass_path,
            montage=montage, eog=eog)
    assert_true(all('parse triggers that' in str(ww.message) for ww in w))

    assert_equal(raw.info['highpass'], 0.1)
    assert_equal(raw.info['lowpass'], 250.)


def test_brainvision_data():
    """Test reading raw Brain Vision files
    """
    assert_raises(IOError, read_raw_brainvision, vmrk_path)
    assert_raises(ValueError, read_raw_brainvision, vhdr_path, montage,
                  preload=True, scale="foo")
    with warnings.catch_warnings(record=True) as w:  # event parsing
        raw_py = _test_raw_reader(
            read_raw_brainvision, vhdr_fname=vhdr_path, montage=montage,
            eog=eog)
    assert_true(all('parse triggers that' in str(ww.message) for ww in w))
    assert_true('RawBrainVision' in repr(raw_py))

    assert_equal(raw_py.info['highpass'], 0.)
    assert_equal(raw_py.info['lowpass'], 250.)

    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, times_py = raw_py[picks]

    # compare with a file that was generated using MNE-C
    raw_bin = Raw(eeg_bin, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_bin, times_bin = raw_bin[picks]

    assert_array_almost_equal(data_py, data_bin)
    assert_array_almost_equal(times_py, times_bin)

    # Make sure EOG channels are marked correctly
    for ch in raw_py.info['chs']:
        if ch['ch_name'] in eog:
            assert_equal(ch['kind'], FIFF.FIFFV_EOG_CH)
        elif ch['ch_name'] == 'STI 014':
            assert_equal(ch['kind'], FIFF.FIFFV_STIM_CH)
        elif ch['ch_name'] in raw_py.info['ch_names']:
            assert_equal(ch['kind'], FIFF.FIFFV_EEG_CH)
        else:
            raise RuntimeError("Unknown Channel: %s" % ch['ch_name'])

    # test loading v2
    read_raw_brainvision(vhdr_v2_path, eog=eog, preload=True,
                         response_trig_shift=1000)


def test_events():
    """Test reading and modifying events"""
    tempdir = _TempDir()

    # check that events are read and stim channel is synthesized correcly
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw = read_raw_brainvision(vhdr_path, eog=eog, preload=True)
        events = raw._get_brainvision_events()
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
        assert_equal(len(w), 1)  # for dropping Sync & R255 events

    # check that events are read and stim channel is synthesized correcly and
    # response triggers are shifted like they're supposed to be.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw = read_raw_brainvision(vhdr_path, eog=eog, preload=True,
                                   response_trig_shift=1000)
        events = raw._get_brainvision_events()
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
        assert_equal(len(w), 1)  # for dropping Sync & R255 events

    # check that events are read and stim channel is synthesized correcly and
    # response triggers are ignored.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw = read_raw_brainvision(vhdr_path, eog=eog, preload=True,
                                   response_trig_shift=None)
        events = raw._get_brainvision_events()
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
        assert_equal(len(w), 1)  # for dropping Sync & R255 events

    # check that events are read properly when event_id is specified for
    # auxiliary events
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw = read_raw_brainvision(vhdr_path, eog=eog, preload=True,
                                   response_trig_shift=None,
                                   event_id={'Sync On': 5})
        events = raw._get_brainvision_events()
        assert_array_equal(events, [[487, 1, 253],
                                    [497, 1, 255],
                                    [1770, 1, 254],
                                    [1780, 1, 255],
                                    [3253, 1, 254],
                                    [3263, 1, 255],
                                    [4936, 1, 253],
                                    [4946, 1, 255],
                                    [6620, 1, 254],
                                    [6630, 1, 255],
                                    [7630, 1, 5]])
        assert_equal(len(w), 1)  # parsing Sync event, missing R255

    assert_raises(TypeError, read_raw_brainvision, vhdr_path, eog=eog,
                  preload=True, response_trig_shift=0.1)
    assert_raises(TypeError, read_raw_brainvision, vhdr_path, eog=eog,
                  preload=True, response_trig_shift=np.nan)

    # Test that both response_trig_shit and event_id can be set
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        read_raw_brainvision(vhdr_path, eog=eog, preload=False,
                             response_trig_shift=100,
                             event_id={'Sync On': 5})

        mne_events = find_events(raw, stim_channel='STI 014')
        assert_array_equal(events[:, [0, 2]], mne_events[:, [0, 2]])
        assert_equal(len(w), 0)  # parsing the Sync event

    # modify events and check that stim channel is updated
    index = events[:, 2] == 255
    events = events[index]
    raw._set_brainvision_events(events)
    mne_events = find_events(raw, stim_channel='STI 014')
    assert_array_equal(events[:, [0, 2]], mne_events[:, [0, 2]])

    # remove events
    nchan = raw.info['nchan']
    ch_name = raw.info['chs'][-2]['ch_name']
    events = np.empty((0, 3))
    raw._set_brainvision_events(events)
    assert_equal(raw.info['nchan'], nchan)
    assert_equal(len(raw._data), nchan)
    assert_equal(raw.info['chs'][-2]['ch_name'], ch_name)
    assert_equal(len(find_events(raw, 'STI 014')), 0)
    assert_allclose(raw[-1][0], 0.)
    fname = op.join(tempdir, 'evt_raw.fif')
    raw.save(fname)

    # add events back in
    events = [[10, 1, 2]]
    raw._set_brainvision_events(events)
    assert_equal(raw.info['nchan'], nchan)
    assert_equal(len(raw._data), nchan)
    assert_equal(raw.info['chs'][-1]['ch_name'], 'STI 014')


run_tests_if_main()
