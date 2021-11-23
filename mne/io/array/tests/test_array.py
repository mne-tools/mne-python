# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_equal)
import pytest
import matplotlib.pyplot as plt

from mne import find_events, Epochs, pick_types
from mne.io import read_raw_fif
from mne.io.array import RawArray
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.meas_info import create_info
from mne.io.pick import get_channel_type_constants
from mne.channels import make_dig_montage

base_dir = op.join(op.dirname(__file__), '..', '..', 'tests', 'data')
fif_fname = op.join(base_dir, 'test_raw.fif')


def test_long_names():
    """Test long name support."""
    info = create_info(['a' * 15 + 'b', 'a' * 16], 1000., verbose='error')
    data = np.empty((2, 1000))
    raw = RawArray(data, info)
    assert raw.ch_names == ['a' * 15 + 'b', 'a' * 16]
    # and a way to get the old behavior
    raw.rename_channels({k: k[:13] for k in raw.ch_names},
                        allow_duplicates=True, verbose='error')
    assert raw.ch_names == ['a' * 13 + '-0', 'a' * 13 + '-1']
    info = create_info(['a' * 16] * 11, 1000., verbose='error')
    data = np.empty((11, 1000))
    raw = RawArray(data, info)
    assert raw.ch_names == ['a' * 16 + '-%s' % ii for ii in range(11)]


def test_array_copy():
    """Test copying during construction."""
    info = create_info(1, 1000.)
    data = np.empty((1, 1000))
    # 'auto' (default)
    raw = RawArray(data, info)
    assert raw._data is data
    assert raw.info is not info
    raw = RawArray(data.astype(np.float32), info)
    assert raw._data is not data
    assert raw.info is not info
    # 'info' (more restrictive)
    raw = RawArray(data, info, copy='info')
    assert raw._data is data
    assert raw.info is not info
    with pytest.raises(ValueError, match="data copying was not .* copy='info"):
        RawArray(data.astype(np.float32), info, copy='info')
    # 'data'
    raw = RawArray(data, info, copy='data')
    assert raw._data is not data
    assert raw.info is info
    # 'both'
    raw = RawArray(data, info, copy='both')
    assert raw._data is not data
    assert raw.info is not info
    raw = RawArray(data.astype(np.float32), info, copy='both')
    assert raw._data is not data
    assert raw.info is not info
    # None
    raw = RawArray(data, info, copy=None)
    assert raw._data is data
    assert raw.info is info
    with pytest.raises(ValueError, match='data copying was not .* copy=None'):
        RawArray(data.astype(np.float32), info, copy=None)


@pytest.mark.slowtest
def test_array_raw():
    """Test creating raw from array."""
    # creating
    raw = read_raw_fif(fif_fname).crop(2, 5)
    data, times = raw[:, :]
    sfreq = raw.info['sfreq']
    ch_names = [(ch[4:] if 'STI' not in ch else ch)
                for ch in raw.info['ch_names']]  # change them, why not
    types = list()
    for ci in range(101):
        types.extend(('grad', 'grad', 'mag'))
    types.extend(['ecog', 'seeg', 'hbo'])  # really 4 meg channels
    types.extend(['stim'] * 9)
    types.extend(['dbs'])  # really eeg channel
    types.extend(['eeg'] * 60)
    picks = np.concatenate([pick_types(raw.info, meg=True)[::20],
                            pick_types(raw.info, meg=False, stim=True),
                            pick_types(raw.info, meg=False, eeg=True)[::20]])
    del raw
    data = data[picks]
    ch_names = np.array(ch_names)[picks].tolist()
    types = np.array(types)[picks].tolist()
    types.pop(-1)
    # wrong length
    pytest.raises(ValueError, create_info, ch_names, sfreq, types)
    # bad entry
    types.append('foo')
    pytest.raises(KeyError, create_info, ch_names, sfreq, types)
    types[-1] = 'eog'
    # default type
    info = create_info(ch_names, sfreq)
    assert_equal(info['chs'][0]['kind'],
                 get_channel_type_constants()['misc']['kind'])
    # use real types
    info = create_info(ch_names, sfreq, types)
    raw2 = _test_raw_reader(RawArray, test_preloading=False,
                            data=data, info=info, first_samp=2 * data.shape[1])
    data2, times2 = raw2[:, :]
    assert_allclose(data, data2)
    assert_allclose(times, times2)
    assert ('RawArray' in repr(raw2))
    pytest.raises(TypeError, RawArray, info, data)

    # filtering
    picks = pick_types(raw2.info, meg=True, misc=True, exclude='bads')[:4]
    assert_equal(len(picks), 4)
    raw_lp = raw2.copy()
    kwargs = dict(fir_design='firwin', picks=picks)
    raw_lp.filter(None, 4.0, h_trans_bandwidth=4., **kwargs)
    raw_hp = raw2.copy()
    raw_hp.filter(16.0, None, l_trans_bandwidth=4., **kwargs)
    raw_bp = raw2.copy()
    raw_bp.filter(8.0, 12.0, l_trans_bandwidth=4., h_trans_bandwidth=4.,
                  **kwargs)
    raw_bs = raw2.copy()
    raw_bs.filter(16.0, 4.0, l_trans_bandwidth=4., h_trans_bandwidth=4.,
                  **kwargs)
    data, _ = raw2[picks, :]
    lp_data, _ = raw_lp[picks, :]
    hp_data, _ = raw_hp[picks, :]
    bp_data, _ = raw_bp[picks, :]
    bs_data, _ = raw_bs[picks, :]
    sig_dec = 15
    assert_array_almost_equal(data, lp_data + bp_data + hp_data, sig_dec)
    assert_array_almost_equal(data, bp_data + bs_data, sig_dec)

    # plotting
    raw2.plot()
    raw2.plot_psd(tmax=2., average=True, n_fft=1024,
                  spatial_colors=False)
    plt.close('all')

    # epoching
    events = find_events(raw2, stim_channel='STI 014')
    events[:, 2] = 1
    assert len(events) > 2
    epochs = Epochs(raw2, events, 1, -0.2, 0.4, preload=True)
    evoked = epochs.average()
    assert_equal(evoked.nave, len(events) - 1)

    # complex data
    rng = np.random.RandomState(0)
    data = rng.randn(1, 100) + 1j * rng.randn(1, 100)
    raw = RawArray(data, create_info(1, 1000., 'eeg'))
    assert_allclose(raw._data, data)

    # Using digital montage to give MNI electrode coordinates
    n_elec = 10
    ts_size = 10000
    Fs = 512.
    ch_names = [str(i) for i in range(n_elec)]
    ch_pos_loc = np.random.randint(60, size=(n_elec, 3)).tolist()

    data = np.random.rand(n_elec, ts_size)
    montage = make_dig_montage(
        ch_pos=dict(zip(ch_names, ch_pos_loc)),
        coord_frame='head'
    )
    info = create_info(ch_names, Fs, 'ecog')

    raw = RawArray(data, info)
    raw.set_montage(montage)
    raw.plot_psd(average=False)  # looking for nonexistent layout
    raw.plot_psd_topo()
