from __future__ import print_function

# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import matplotlib

from numpy.testing import assert_array_almost_equal, assert_allclose
from nose.tools import assert_equal, assert_raises, assert_true
from mne import find_events, Epochs, pick_types, concatenate_raws
from mne.io import Raw
from mne.io.array import RawArray
from mne.io.meas_info import create_info, _kind_dict
from mne.utils import _TempDir, slow_test, requires_scipy_version

matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests might throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'tests', 'data')
fif_fname = op.join(base_dir, 'test_raw.fif')


@slow_test
@requires_scipy_version('0.12')
def test_array_raw():
    """Test creating raw from array
    """
    tempdir = _TempDir()
    # creating
    raw = Raw(fif_fname).crop(2, 5, copy=False)
    data, times = raw[:, :]
    sfreq = raw.info['sfreq']
    ch_names = [(ch[4:] if 'STI' not in ch else ch)
                for ch in raw.info['ch_names']]  # change them, why not
    # del raw
    types = list()
    for ci in range(102):
        types.extend(('grad', 'grad', 'mag'))
    types.extend(['stim'] * 9)
    types.extend(['eeg'] * 60)
    # wrong length
    assert_raises(ValueError, create_info, ch_names, sfreq, types)
    # bad entry
    types.append('foo')
    assert_raises(KeyError, create_info, ch_names, sfreq, types)
    types[-1] = 'eog'
    # default type
    info = create_info(ch_names, sfreq)
    assert_equal(info['chs'][0]['kind'], _kind_dict['misc'][0])
    # use real types
    info = create_info(ch_names, sfreq, types)
    raw2 = RawArray(data, info)
    data2, times2 = raw2[:, :]
    assert_allclose(data, data2)
    assert_allclose(times, times2)
    # Make sure concatenation works
    raw_concat = concatenate_raws([raw2.copy(), raw2])
    assert_equal(raw_concat.n_times, 2 * raw2.n_times)

    # saving
    temp_fname = op.join(tempdir, 'raw.fif')
    raw2.save(temp_fname)
    raw3 = Raw(temp_fname)
    data3, times3 = raw3[:, :]
    assert_allclose(data, data3)
    assert_allclose(times, times3)

    # filtering
    picks = pick_types(raw2.info, misc=True, exclude='bads')[:4]
    assert_equal(len(picks), 4)
    raw_lp = raw2.copy()
    with warnings.catch_warnings(record=True):
        raw_lp.filter(0., 4.0 - 0.25, picks=picks, n_jobs=2)
    raw_hp = raw2.copy()
    with warnings.catch_warnings(record=True):
        raw_hp.filter(8.0 + 0.25, None, picks=picks, n_jobs=2)
    raw_bp = raw2.copy()
    with warnings.catch_warnings(record=True):
        raw_bp.filter(4.0 + 0.25, 8.0 - 0.25, picks=picks)
    raw_bs = raw2.copy()
    with warnings.catch_warnings(record=True):
        raw_bs.filter(8.0 + 0.25, 4.0 - 0.25, picks=picks, n_jobs=2)
    data, _ = raw2[picks, :]
    lp_data, _ = raw_lp[picks, :]
    hp_data, _ = raw_hp[picks, :]
    bp_data, _ = raw_bp[picks, :]
    bs_data, _ = raw_bs[picks, :]
    sig_dec = 11
    assert_array_almost_equal(data, lp_data + bp_data + hp_data, sig_dec)
    assert_array_almost_equal(data, bp_data + bs_data, sig_dec)

    # plotting
    raw2.plot()
    raw2.plot_psd()

    # epoching
    events = find_events(raw2, stim_channel='STI 014')
    events[:, 2] = 1
    assert_true(len(events) > 2)
    epochs = Epochs(raw2, events, 1, -0.2, 0.4, preload=True)
    epochs.plot_drop_log(return_fig=True)
    epochs.plot()
    evoked = epochs.average()
    evoked.plot()
