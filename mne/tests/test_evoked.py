# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Denis Engemann <denis.engemann@gmail.com>
#         Andrew Dykstra <andrew.r.dykstra@gmail.com>
#         Mads Jensen <mje.mads@gmail.com>
#
# License: BSD-3-Clause

from copy import deepcopy
import os.path as op
import pickle

import numpy as np
from scipy import fftpack
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_array_equal, assert_allclose)
import pytest

from mne import (equalize_channels, pick_types, read_evokeds, write_evokeds,
                 combine_evoked, create_info, read_events,
                 Epochs, EpochsArray)
from mne.evoked import _get_peak, Evoked, EvokedArray
from mne.io import read_raw_fif
from mne.io.constants import FIFF
from mne.utils import requires_pandas, grand_average

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname = op.join(base_dir, 'test-ave.fif')
fname_gz = op.join(base_dir, 'test-ave.fif.gz')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')


def test_get_data():
    """Test the get_data method for Evoked."""
    evoked = read_evokeds(fname, 0)
    d1 = evoked.get_data()
    d2 = evoked.data
    assert_array_equal(d1, d2)

    eeg_idxs = np.array([i == "eeg" for i in evoked.get_channel_types()])
    assert_array_equal(
        evoked.data[eeg_idxs],
        evoked.get_data(picks="eeg")
    )

    # Get a specific time window using tmin and tmax
    d3 = evoked.get_data(tmin=0)
    assert np.all(d3.shape[1] ==
                  evoked.data.shape[1] -
                  np.nonzero(evoked.times == 0)[0])

    assert evoked.get_data(tmin=0, tmax=0).size == 0

    with pytest.raises(TypeError, match='tmin .* float, None'):
        evoked.get_data(tmin=[1], tmax=1)

    with pytest.raises(TypeError, match='tmax .* float, None'):
        evoked.get_data(tmin=1, tmax=np.ones(5))

    # Test units
    # more tests in mne/io/tests/test_raw.py::test_get_data_units
    # EEG is already in V, so no conversion should take place
    d1 = evoked.get_data(picks="eeg", units=None)
    d2 = evoked.get_data(picks="eeg", units="V")
    assert_array_equal(d1, d2)

    # Convert to µV
    d3 = evoked.get_data(picks="eeg", units="µV")
    assert_array_equal(d1 * 1e6, d3)


def test_decim():
    """Test evoked decimation."""
    rng = np.random.RandomState(0)
    n_channels, n_times = 10, 20
    dec_1, dec_2 = 2, 3
    decim = dec_1 * dec_2
    sfreq = 10.
    sfreq_new = sfreq / decim
    data = rng.randn(n_channels, n_times)
    info = create_info(n_channels, sfreq, 'eeg')
    with info._unlock():
        info['lowpass'] = sfreq_new / float(decim)
    evoked = EvokedArray(data, info, tmin=-1)
    evoked_dec = evoked.copy().decimate(decim)
    evoked_dec_2 = evoked.copy().decimate(decim, offset=1)
    evoked_dec_3 = evoked.decimate(dec_1).decimate(dec_2)
    assert_array_equal(evoked_dec.data, data[:, ::decim])
    assert_array_equal(evoked_dec_2.data, data[:, 1::decim])
    assert_array_equal(evoked_dec.data, evoked_dec_3.data)

    # Check proper updating of various fields
    assert evoked_dec.first == -2
    assert evoked_dec.last == 1
    assert_array_equal(evoked_dec.times, [-1, -0.4, 0.2, 0.8])
    assert evoked_dec_2.first == -2
    assert evoked_dec_2.last == 1
    assert_array_equal(evoked_dec_2.times, [-0.9, -0.3, 0.3, 0.9])
    assert evoked_dec_3.first == -2
    assert evoked_dec_3.last == 1
    assert_array_equal(evoked_dec_3.times, [-1, -0.4, 0.2, 0.8])

    # make sure the time nearest zero is also sample number 0.
    for ev in (evoked_dec, evoked_dec_2, evoked_dec_3):
        lowest_index = np.argmin(np.abs(np.arange(ev.first, ev.last)))
        idxs_of_times_nearest_zero = \
            np.where(np.abs(ev.times) == np.min(np.abs(ev.times)))[0]
        # we use `in` here in case two times are equidistant from 0.
        assert lowest_index in idxs_of_times_nearest_zero
        assert len(idxs_of_times_nearest_zero) in (1, 2)

    # Now let's do it with some real data
    raw = read_raw_fif(raw_fname)
    events = read_events(event_name)
    sfreq_new = raw.info['sfreq'] / decim
    with raw.info._unlock():
        raw.info['lowpass'] = sfreq_new / 4.  # suppress aliasing warnings
    picks = pick_types(raw.info, meg=True, eeg=True, exclude=())
    epochs = Epochs(raw, events, 1, -0.2, 0.5, picks=picks, preload=True)
    for offset in (0, 1):
        ev_ep_decim = epochs.copy().decimate(decim, offset).average()
        ev_decim = epochs.average().decimate(decim, offset)
        expected_times = epochs.times[offset::decim]
        assert_allclose(ev_decim.times, expected_times)
        assert_allclose(ev_ep_decim.times, expected_times)
        expected_data = epochs.get_data()[:, :, offset::decim].mean(axis=0)
        assert_allclose(ev_decim.data, expected_data)
        assert_allclose(ev_ep_decim.data, expected_data)
        assert_equal(ev_decim.info['sfreq'], sfreq_new)
        assert_array_equal(ev_decim.times, expected_times)


def test_savgol_filter():
    """Test savgol filtering."""
    h_freq = 10.
    evoked = read_evokeds(fname, 0)
    freqs = fftpack.fftfreq(len(evoked.times), 1. / evoked.info['sfreq'])
    data = np.abs(fftpack.fft(evoked.data))
    match_mask = np.logical_and(freqs >= 0, freqs <= h_freq / 2.)
    mismatch_mask = np.logical_and(freqs >= h_freq * 2, freqs < 50.)
    pytest.raises(ValueError, evoked.savgol_filter, evoked.info['sfreq'])
    evoked_sg = evoked.copy().savgol_filter(h_freq)
    data_filt = np.abs(fftpack.fft(evoked_sg.data))
    # decent in pass-band
    assert_allclose(np.mean(data[:, match_mask], 0),
                    np.mean(data_filt[:, match_mask], 0),
                    rtol=1e-4, atol=1e-2)
    # suppression in stop-band
    assert (np.mean(data[:, mismatch_mask]) >
            np.mean(data_filt[:, mismatch_mask]) * 5)
    # original preserved
    assert_allclose(data, np.abs(fftpack.fft(evoked.data)), atol=1e-16)


def test_hash_evoked():
    """Test evoked hashing."""
    ave = read_evokeds(fname, 0)
    ave_2 = read_evokeds(fname, 0)
    assert hash(ave) == hash(ave_2)
    assert ave == ave_2
    # do NOT use assert_equal here, failing output is terrible
    assert pickle.dumps(ave) == pickle.dumps(ave_2)

    ave_2.data[0, 0] -= 1
    assert hash(ave) != hash(ave_2)


def _aspect_kinds():
    """Yield evoked aspect kinds."""
    kinds = list()
    for key in FIFF:
        if not key.startswith('FIFFV_ASPECT_'):
            continue
        kinds.append(getattr(FIFF, str(key)))
    return kinds


@pytest.mark.parametrize('aspect_kind', _aspect_kinds())
def test_evoked_aspects(aspect_kind, tmp_path):
    """Test handling of evoked aspects."""
    # gh-6359
    ave = read_evokeds(fname, 0)
    ave._aspect_kind = aspect_kind
    assert 'Evoked' in repr(ave)
    # for completeness let's try a round-trip
    temp_fname = op.join(str(tmp_path), 'test-ave.fif')
    ave.save(temp_fname)
    ave_2 = read_evokeds(temp_fname, condition=0)
    assert_allclose(ave.data, ave_2.data)
    assert ave.kind == ave_2.kind


@pytest.mark.slowtest
def test_io_evoked(tmp_path):
    """Test IO for evoked data (fif + gz) with integer and str args."""
    ave = read_evokeds(fname, 0)
    ave_double = ave.copy()
    ave_double.comment = ave.comment + ' doubled nave'
    ave_double.nave = ave.nave * 2

    write_evokeds(tmp_path / 'evoked-ave.fif', [ave, ave_double])
    ave2, ave_double = read_evokeds(op.join(tmp_path, 'evoked-ave.fif'))
    assert ave2.nave * 2 == ave_double.nave

    # This not being assert_array_equal due to windows rounding
    assert (np.allclose(ave.data, ave2.data, atol=1e-16, rtol=1e-3))
    assert_array_almost_equal(ave.times, ave2.times)
    assert_equal(ave.nave, ave2.nave)
    assert_equal(ave._aspect_kind, ave2._aspect_kind)
    assert_equal(ave.kind, ave2.kind)
    assert_equal(ave.last, ave2.last)
    assert_equal(ave.first, ave2.first)
    assert (repr(ave))

    # test compressed i/o
    ave2 = read_evokeds(fname_gz, 0)
    assert (np.allclose(ave.data, ave2.data, atol=1e-16, rtol=1e-8))

    # test str access
    condition = 'Left Auditory'
    pytest.raises(ValueError, read_evokeds, fname, condition, kind='stderr')
    pytest.raises(ValueError, read_evokeds, fname, condition,
                  kind='standard_error')
    ave3 = read_evokeds(fname, condition)
    assert_array_almost_equal(ave.data, ave3.data, 19)

    # test read_evokeds and write_evokeds
    aves1 = read_evokeds(fname)[1::2]
    aves2 = read_evokeds(fname, [1, 3])
    aves3 = read_evokeds(fname, ['Right Auditory', 'Right visual'])
    write_evokeds(tmp_path / 'evoked-ave.fif', aves1)
    aves4 = read_evokeds(tmp_path / 'evoked-ave.fif')
    for aves in [aves2, aves3, aves4]:
        for [av1, av2] in zip(aves1, aves):
            assert_array_almost_equal(av1.data, av2.data)
            assert_array_almost_equal(av1.times, av2.times)
            assert_equal(av1.nave, av2.nave)
            assert_equal(av1.kind, av2.kind)
            assert_equal(av1._aspect_kind, av2._aspect_kind)
            assert_equal(av1.last, av2.last)
            assert_equal(av1.first, av2.first)
            assert_equal(av1.comment, av2.comment)

    # test saving and reading complex numbers in evokeds
    ave_complex = ave.copy()
    ave_complex._data = 1j * ave_complex.data
    fname_temp = str(tmp_path / 'complex-ave.fif')
    ave_complex.save(fname_temp)
    ave_complex = read_evokeds(fname_temp)[0]
    assert_allclose(ave.data, ave_complex.data.imag)

    # test warnings on bad filenames
    fname2 = tmp_path / 'test-bad-name.fif'
    with pytest.warns(RuntimeWarning, match='-ave.fif'):
        write_evokeds(fname2, ave)
    with pytest.warns(RuntimeWarning, match='-ave.fif'):
        read_evokeds(fname2)

    # test writing when order of bads doesn't match
    fname3 = tmp_path / 'test-bad-order-ave.fif'
    condition = 'Left Auditory'
    ave4 = read_evokeds(fname, condition)
    ave4.info['bads'] = ave4.ch_names[:3]
    ave5 = ave4.copy()
    ave5.info['bads'] = ave4.info['bads'][::-1]
    write_evokeds(fname3, [ave4, ave5])

    # constructor
    pytest.raises(TypeError, Evoked, fname)

    # MaxShield
    fname_ms = tmp_path / 'test-ave.fif'
    assert (ave.info['maxshield'] is False)
    with ave.info._unlock():
        ave.info['maxshield'] = True
    ave.save(fname_ms)
    pytest.raises(ValueError, read_evokeds, fname_ms)
    with pytest.warns(RuntimeWarning, match='Elekta'):
        aves = read_evokeds(fname_ms, allow_maxshield=True)
    assert all(ave.info['maxshield'] is True for ave in aves)
    aves = read_evokeds(fname_ms, allow_maxshield='yes')
    assert (all(ave.info['maxshield'] is True for ave in aves))


def test_shift_time_evoked(tmp_path):
    """Test for shifting of time scale."""
    tempdir = str(tmp_path)
    # Shift backward
    ave = read_evokeds(fname, 0).shift_time(-0.1, relative=True)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave)

    # Shift forward twice the amount
    ave_bshift = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)
    ave_bshift.shift_time(0.2, relative=True)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave_bshift)

    # Shift backward again
    ave_fshift = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)
    ave_fshift.shift_time(-0.1, relative=True)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave_fshift)

    ave_normal = read_evokeds(fname, 0)
    ave_relative = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)

    assert_allclose(ave_normal.data, ave_relative.data, atol=1e-16, rtol=1e-3)
    assert_array_almost_equal(ave_normal.times, ave_relative.times, 8)

    assert_equal(ave_normal.last, ave_relative.last)
    assert_equal(ave_normal.first, ave_relative.first)

    # Absolute time shift
    ave = read_evokeds(fname, 0)
    ave.shift_time(-0.3, relative=False)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave)

    ave_absolute = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)

    assert_allclose(ave_normal.data, ave_absolute.data, atol=1e-16, rtol=1e-3)
    assert_equal(ave_absolute.first, int(-0.3 * ave.info['sfreq']))

    # subsample shift
    shift = 1e-6  # 1 µs, should be well below 1/sfreq
    ave = read_evokeds(fname, 0)
    times = ave.times
    ave.shift_time(shift)
    assert_allclose(times + shift, ave.times, atol=1e-16, rtol=1e-12)

    # test handling of Evoked.first, Evoked.last
    ave = read_evokeds(fname, 0)
    first_last = np.array([ave.first, ave.last])
    # should shift by 0 samples
    ave.shift_time(1e-6)
    assert_array_equal(first_last, np.array([ave.first, ave.last]))
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave)
    ave_loaded = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)
    assert_array_almost_equal(ave.times, ave_loaded.times, 8)
    # should shift by 57 samples
    ave.shift_time(57. / ave.info['sfreq'])
    assert_array_equal(first_last + 57, np.array([ave.first, ave.last]))
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave)
    ave_loaded = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)
    assert_array_almost_equal(ave.times, ave_loaded.times, 8)


def test_tmin_tmax():
    """Test that the tmin and tmax attributes return the correct time."""
    evoked = read_evokeds(fname, 0)
    assert evoked.times[0] == evoked.tmin
    assert evoked.times[-1] == evoked.tmax


def test_evoked_resample(tmp_path):
    """Test resampling evoked data."""
    tempdir = str(tmp_path)
    # upsample, write it out, read it in
    ave = read_evokeds(fname, 0)
    orig_lp = ave.info['lowpass']
    sfreq_normal = ave.info['sfreq']
    ave.resample(2 * sfreq_normal, npad=100)
    assert ave.info['lowpass'] == orig_lp
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave)
    ave_up = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)

    # compare it to the original
    ave_normal = read_evokeds(fname, 0)

    # and compare the original to the downsampled upsampled version
    ave_new = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)
    ave_new.resample(sfreq_normal, npad=100)
    assert ave.info['lowpass'] == orig_lp

    assert_array_almost_equal(ave_normal.data, ave_new.data, 2)
    assert_array_almost_equal(ave_normal.times, ave_new.times)
    assert_equal(ave_normal.nave, ave_new.nave)
    assert_equal(ave_normal._aspect_kind, ave_new._aspect_kind)
    assert_equal(ave_normal.kind, ave_new.kind)
    assert_equal(ave_normal.last, ave_new.last)
    assert_equal(ave_normal.first, ave_new.first)

    # for the above to work, the upsampling just about had to, but
    # we'll add a couple extra checks anyway
    assert (len(ave_up.times) == 2 * len(ave_normal.times))
    assert (ave_up.data.shape[1] == 2 * ave_normal.data.shape[1])

    ave_new.resample(50)
    assert ave_new.info['sfreq'] == 50.
    assert ave_new.info['lowpass'] == 25.


def test_evoked_filter():
    """Test filtering evoked data."""
    # this is mostly a smoke test as the Epochs and raw tests are more complete
    ave = read_evokeds(fname, 0).pick_types(meg='grad')
    ave.data[:] = 1.
    assert round(ave.info['lowpass']) == 172
    ave_filt = ave.copy().filter(None, 40., fir_design='firwin')
    assert ave_filt.info['lowpass'] == 40.
    assert_allclose(ave.data, 1., atol=1e-6)


def test_evoked_detrend():
    """Test for detrending evoked data."""
    ave = read_evokeds(fname, 0)
    ave_normal = read_evokeds(fname, 0)
    ave.detrend(0)
    ave_normal.data -= np.mean(ave_normal.data, axis=1)[:, np.newaxis]
    picks = pick_types(ave.info, meg=True, eeg=True, exclude='bads')
    assert_allclose(ave.data[picks], ave_normal.data[picks],
                    rtol=1e-8, atol=1e-16)


@requires_pandas
def test_to_data_frame():
    """Test evoked Pandas exporter."""
    ave = read_evokeds(fname, 0)
    # test index checking
    with pytest.raises(ValueError, match='options. Valid index options are'):
        ave.to_data_frame(index=['foo', 'bar'])
    with pytest.raises(ValueError, match='"qux" is not a valid option'):
        ave.to_data_frame(index='qux')
    with pytest.raises(TypeError, match='index must be `None` or a string or'):
        ave.to_data_frame(index=np.arange(400))
    # test setting index
    df = ave.to_data_frame(index='time')
    assert 'time' not in df.columns
    assert 'time' in df.index.names
    # test wide and long formats
    df_wide = ave.to_data_frame()
    assert all(np.in1d(ave.ch_names, df_wide.columns))
    df_long = ave.to_data_frame(long_format=True)
    expected = ('time', 'channel', 'ch_type', 'value')
    assert set(expected) == set(df_long.columns)
    assert set(ave.ch_names) == set(df_long['channel'])
    assert(len(df_long) == ave.data.size)
    del df_wide, df_long
    # test scalings
    df = ave.to_data_frame(index='time')
    assert ((df.columns == ave.ch_names).all())
    assert_array_equal(df.values[:, 0], ave.data[0] * 1e13)
    assert_array_equal(df.values[:, 2], ave.data[2] * 1e15)


@requires_pandas
@pytest.mark.parametrize('time_format', (None, 'ms', 'timedelta'))
def test_to_data_frame_time_format(time_format):
    """Test time conversion in evoked Pandas exporter."""
    from pandas import Timedelta
    ave = read_evokeds(fname, 0)
    # test time_format
    df = ave.to_data_frame(time_format=time_format)
    dtypes = {None: np.float64, 'ms': np.int64, 'timedelta': Timedelta}
    assert isinstance(df['time'].iloc[0], dtypes[time_format])


def test_evoked_proj():
    """Test SSP proj operations."""
    for proj in [True, False]:
        ave = read_evokeds(fname, condition=0, proj=proj)
        assert (all(p['active'] == proj for p in ave.info['projs']))

        # test adding / deleting proj
        if proj:
            pytest.raises(ValueError, ave.add_proj, [],
                          {'remove_existing': True})
            pytest.raises(ValueError, ave.del_proj, 0)
        else:
            projs = deepcopy(ave.info['projs'])
            n_proj = len(ave.info['projs'])
            ave.del_proj(0)
            assert (len(ave.info['projs']) == n_proj - 1)
            # Test that already existing projections are not added.
            ave.add_proj(projs, remove_existing=False)
            assert (len(ave.info['projs']) == n_proj)
            ave.add_proj(projs[:-1], remove_existing=True)
            assert (len(ave.info['projs']) == n_proj - 1)

    ave = read_evokeds(fname, condition=0, proj=False)
    data = ave.data.copy()
    ave.apply_proj()
    assert_allclose(np.dot(ave._projector, data), ave.data)


def test_get_peak():
    """Test peak getter."""
    evoked = read_evokeds(fname, condition=0, proj=True)
    pytest.raises(ValueError, evoked.get_peak, ch_type='mag', tmin=1)
    pytest.raises(ValueError, evoked.get_peak, ch_type='mag', tmax=0.9)
    pytest.raises(ValueError, evoked.get_peak, ch_type='mag', tmin=0.02,
                  tmax=0.01)
    pytest.raises(ValueError, evoked.get_peak, ch_type='mag', mode='foo')
    pytest.raises(RuntimeError, evoked.get_peak, ch_type=None, mode='foo')
    pytest.raises(ValueError, evoked.get_peak, ch_type='misc', mode='foo')

    ch_name, time_idx = evoked.get_peak(ch_type='mag')
    assert (ch_name in evoked.ch_names)
    assert (time_idx in evoked.times)

    ch_name, time_idx, max_amp = evoked.get_peak(ch_type='mag',
                                                 time_as_index=True,
                                                 return_amplitude=True)
    assert (time_idx < len(evoked.times))
    assert_equal(ch_name, 'MEG 1421')
    assert_allclose(max_amp, 7.17057e-13, rtol=1e-5)

    pytest.raises(ValueError, evoked.get_peak, ch_type='mag',
                  merge_grads=True)
    ch_name, time_idx = evoked.get_peak(ch_type='grad', merge_grads=True)
    assert_equal(ch_name, 'MEG 244X')

    data = np.array([[0., 1., 2.],
                     [0., -3., 0]])

    times = np.array([.1, .2, .3])

    ch_idx, time_idx, max_amp = _get_peak(data, times, mode='abs')
    assert_equal(ch_idx, 1)
    assert_equal(time_idx, 1)
    assert_allclose(max_amp, -3.)

    ch_idx, time_idx, max_amp = _get_peak(data * -1, times, mode='neg')
    assert_equal(ch_idx, 0)
    assert_equal(time_idx, 2)
    assert_allclose(max_amp, -2.)

    ch_idx, time_idx, max_amp = _get_peak(data, times, mode='pos')
    assert_equal(ch_idx, 0)
    assert_equal(time_idx, 2)
    assert_allclose(max_amp, 2.)

    pytest.raises(ValueError, _get_peak, data + 1e3, times, mode='neg')
    pytest.raises(ValueError, _get_peak, data - 1e3, times, mode='pos')


def test_drop_channels_mixin():
    """Test channels-dropping functionality."""
    evoked = read_evokeds(fname, condition=0, proj=True)
    drop_ch = evoked.ch_names[:3]
    ch_names = evoked.ch_names[3:]

    ch_names_orig = evoked.ch_names
    dummy = evoked.copy().drop_channels(drop_ch)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, evoked.ch_names)
    assert_equal(len(ch_names_orig), len(evoked.data))
    dummy2 = evoked.copy().drop_channels([drop_ch[0]])
    assert_equal(dummy2.ch_names, ch_names_orig[1:])

    evoked.drop_channels(drop_ch)
    assert_equal(ch_names, evoked.ch_names)
    assert_equal(len(ch_names), len(evoked.data))

    for ch_names in ([1, 2], "fake", ["fake"]):
        pytest.raises(ValueError, evoked.drop_channels, ch_names)


def test_pick_channels_mixin():
    """Test channel-picking functionality."""
    evoked = read_evokeds(fname, condition=0, proj=True)
    ch_names = evoked.ch_names[:3]

    ch_names_orig = evoked.ch_names
    dummy = evoked.copy().pick_channels(ch_names)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, evoked.ch_names)
    assert_equal(len(ch_names_orig), len(evoked.data))

    evoked.pick_channels(ch_names)
    assert_equal(ch_names, evoked.ch_names)
    assert_equal(len(ch_names), len(evoked.data))

    evoked = read_evokeds(fname, condition=0, proj=True)
    assert ('meg' in evoked)
    assert ('eeg' in evoked)
    evoked.pick_types(meg=False, eeg=True)
    assert ('meg' not in evoked)
    assert ('eeg' in evoked)
    assert (len(evoked.ch_names) == 60)


def test_equalize_channels():
    """Test equalization of channels."""
    evoked1 = read_evokeds(fname, condition=0, proj=True)
    evoked2 = evoked1.copy()
    ch_names = evoked1.ch_names[2:]
    evoked1.drop_channels(evoked1.ch_names[:1])
    evoked2.drop_channels(evoked2.ch_names[1:2])
    my_comparison = [evoked1, evoked2]
    my_comparison = equalize_channels(my_comparison)
    for e in my_comparison:
        assert_equal(ch_names, e.ch_names)


def test_arithmetic():
    """Test evoked arithmetic."""
    ev = read_evokeds(fname, condition=0)
    ev20 = EvokedArray(np.ones_like(ev.data), ev.info, ev.times[0], nave=20)
    ev30 = EvokedArray(np.ones_like(ev.data), ev.info, ev.times[0], nave=30)

    tol = dict(rtol=1e-9, atol=0)
    # test subtraction
    sub1 = combine_evoked([ev, ev], weights=[1, -1])
    sub2 = combine_evoked([ev, -ev], weights=[1, 1])
    assert np.allclose(sub1.data, np.zeros_like(sub1.data), atol=1e-20)
    assert np.allclose(sub2.data, np.zeros_like(sub2.data), atol=1e-20)
    # test nave weighting. Expect signal ampl.: 1*(20/50) + 1*(30/50) == 1
    # and expect nave == ev1.nave + ev2.nave
    ev = combine_evoked([ev20, ev30], weights='nave')
    assert np.allclose(ev.nave, ev20.nave + ev30.nave)
    assert np.allclose(ev.data, np.ones_like(ev.data), **tol)
    # test equal-weighted sum. Expect signal ampl. == 2
    # and expect nave == 1/sum(1/naves) == 1/(1/20 + 1/30) == 12
    ev = combine_evoked([ev20, ev30], weights=[1, 1])
    assert np.allclose(ev.nave, 12.)
    assert np.allclose(ev.data, ev20.data + ev30.data, **tol)
    # test equal-weighted average. Expect signal ampl. == 1
    # and expect nave == 1/sum(weights²/naves) == 1/(0.5²/20 + 0.5²/30) == 48
    ev = combine_evoked([ev20, ev30], weights='equal')
    assert np.allclose(ev.nave, 48.)
    assert np.allclose(ev.data, np.mean([ev20.data, ev30.data], axis=0), **tol)
    # test zero weights
    ev = combine_evoked([ev20, ev30], weights=[1, 0])
    assert ev.nave == ev20.nave
    assert np.allclose(ev.data, ev20.data, **tol)

    # default comment behavior if evoked.comment is None
    old_comment1 = ev20.comment
    ev20.comment = None
    ev = combine_evoked([ev20, -ev30], weights=[1, -1])
    assert_equal(ev.comment.count('unknown'), 2)
    assert ev.comment == 'unknown + unknown'
    ev20.comment = old_comment1

    with pytest.raises(ValueError, match="Invalid value for the 'weights'"):
        combine_evoked([ev20, ev30], weights='foo')
    with pytest.raises(ValueError, match='weights must be the same size as'):
        combine_evoked([ev20, ev30], weights=[1])

    # grand average
    evoked1, evoked2 = read_evokeds(fname, condition=[0, 1], proj=True)
    ch_names = evoked1.ch_names[2:]
    evoked1.info['bads'] = ['EEG 008']  # test interpolation
    evoked1.drop_channels(evoked1.ch_names[:1])
    evoked2.drop_channels(evoked2.ch_names[1:2])
    gave = grand_average([evoked1, evoked2])
    assert_equal(gave.data.shape, [len(ch_names), evoked1.data.shape[1]])
    assert_equal(ch_names, gave.ch_names)
    assert_equal(gave.nave, 2)
    with pytest.raises(TypeError, match='All elements must be an instance of'):
        grand_average([1, evoked1])
    gave = grand_average([ev20, ev20, -ev30])  # (1 + 1 + -1) / 3  =  1/3
    assert_allclose(gave.data, np.full_like(gave.data, 1. / 3.))

    # test channel (re)ordering
    evoked1, evoked2 = read_evokeds(fname, condition=[0, 1], proj=True)
    data2 = evoked2.data  # assumes everything is ordered to the first evoked
    data = (evoked1.data + evoked2.data) / 2.
    evoked2.reorder_channels(evoked2.ch_names[::-1])
    assert not np.allclose(data2, evoked2.data)
    with pytest.warns(RuntimeWarning, match='reordering'):
        evoked3 = combine_evoked([evoked1, evoked2], weights=[0.5, 0.5])
    assert np.allclose(evoked3.data, data)
    assert evoked1.ch_names != evoked2.ch_names
    assert evoked1.ch_names == evoked3.ch_names


def test_array_epochs(tmp_path):
    """Test creating evoked from array."""
    tempdir = str(tmp_path)

    # creating
    rng = np.random.RandomState(42)
    data1 = rng.randn(20, 60)
    sfreq = 1e3
    ch_names = ['EEG %03d' % (i + 1) for i in range(20)]
    types = ['eeg'] * 20
    info = create_info(ch_names, sfreq, types)
    evoked1 = EvokedArray(data1, info, tmin=-0.01)

    # save, read, and compare evokeds
    tmp_fname = op.join(tempdir, 'evkdary-ave.fif')
    evoked1.save(tmp_fname)
    evoked2 = read_evokeds(tmp_fname)[0]
    data2 = evoked2.data
    assert_allclose(data1, data2)
    assert_array_almost_equal(evoked1.times, evoked2.times, 8)
    assert_equal(evoked1.first, evoked2.first)
    assert_equal(evoked1.last, evoked2.last)
    assert_equal(evoked1.kind, evoked2.kind)
    assert_equal(evoked1.nave, evoked2.nave)

    # now compare with EpochsArray (with single epoch)
    data3 = data1[np.newaxis, :, :]
    events = np.c_[10, 0, 1]
    evoked3 = EpochsArray(data3, info, events=events, tmin=-0.01).average()
    assert_allclose(evoked1.data, evoked3.data)
    assert_allclose(evoked1.times, evoked3.times)
    assert_equal(evoked1.first, evoked3.first)
    assert_equal(evoked1.last, evoked3.last)
    assert_equal(evoked1.kind, evoked3.kind)
    assert_equal(evoked1.nave, evoked3.nave)

    # test kind check
    with pytest.raises(ValueError, match='Invalid value'):
        EvokedArray(data1, info, tmin=0, kind=1)
    with pytest.raises(ValueError, match='Invalid value'):
        EvokedArray(data1, info, kind='mean')

    # test match between channels info and data
    ch_names = ['EEG %03d' % (i + 1) for i in range(19)]
    types = ['eeg'] * 19
    info = create_info(ch_names, sfreq, types)
    pytest.raises(ValueError, EvokedArray, data1, info, tmin=-0.01)


def test_time_as_index_and_crop():
    """Test time as index and cropping."""
    tmin, tmax = -0.1, 0.1
    evoked = read_evokeds(fname, condition=0).crop(tmin, tmax)
    delta = 1. / evoked.info['sfreq']
    atol = 0.5 * delta
    assert_allclose(evoked.times[[0, -1]], [tmin, tmax], atol=atol)
    assert_array_equal(evoked.time_as_index([-.1, .1], use_rounding=True),
                       [0, len(evoked.times) - 1])
    evoked.crop(evoked.tmin, evoked.tmax, include_tmax=False)
    n_times = len(evoked.times)
    with pytest.warns(RuntimeWarning, match='tmax is set to'):
        evoked.crop(tmin, tmax, include_tmax=False)
    assert len(evoked.times) == n_times
    assert_allclose(evoked.times[[0, -1]], [tmin, tmax - delta], atol=atol)


def test_add_channels():
    """Test evoked splitting / re-appending channel types."""
    evoked = read_evokeds(fname, condition=0)
    hpi_coils = [{'event_bits': []},
                 {'event_bits': np.array([256, 0, 256, 256])},
                 {'event_bits': np.array([512, 0, 512, 512])}]
    with evoked.info._unlock():
        evoked.info['hpi_subsystem'] = dict(hpi_coils=hpi_coils, ncoil=2)
    evoked_eeg = evoked.copy().pick_types(meg=False, eeg=True)
    evoked_meg = evoked.copy().pick_types(meg=True)
    evoked_stim = evoked.copy().pick_types(meg=False, stim=True)
    evoked_eeg_meg = evoked.copy().pick_types(meg=True, eeg=True)
    evoked_new = evoked_meg.copy().add_channels([evoked_eeg, evoked_stim])
    assert (all(ch in evoked_new.ch_names
                for ch in evoked_stim.ch_names + evoked_meg.ch_names))
    evoked_new = evoked_meg.copy().add_channels([evoked_eeg])

    assert (ch in evoked_new.ch_names for ch in evoked.ch_names)
    assert_array_equal(evoked_new.data, evoked_eeg_meg.data)
    assert (all(ch not in evoked_new.ch_names
                for ch in evoked_stim.ch_names))

    # Now test errors
    evoked_badsf = evoked_eeg.copy()
    with evoked_badsf.info._unlock():
        evoked_badsf.info['sfreq'] = 3.1415927
    evoked_eeg = evoked_eeg.crop(-.1, .1)

    pytest.raises(RuntimeError, evoked_meg.add_channels, [evoked_badsf])
    pytest.raises(AssertionError, evoked_meg.add_channels, [evoked_eeg])
    pytest.raises(ValueError, evoked_meg.add_channels, [evoked_meg])
    pytest.raises(TypeError, evoked_meg.add_channels, evoked_badsf)


def test_evoked_baseline(tmp_path):
    """Test evoked baseline."""
    evoked = read_evokeds(fname, condition=0, baseline=None)

    # Here we create a data_set with constant data.
    evoked = EvokedArray(np.ones_like(evoked.data), evoked.info,
                         evoked.times[0])
    assert evoked.baseline is None

    evoked_baselined = EvokedArray(np.ones_like(evoked.data), evoked.info,
                                   evoked.times[0], baseline=(None, 0))
    assert_allclose(evoked_baselined.baseline, (evoked_baselined.tmin, 0))
    del evoked_baselined

    # Mean baseline correction is applied, since the data is equal to its mean
    # the resulting data should be a matrix of zeroes.
    baseline = (None, None)
    evoked.apply_baseline(baseline)
    assert_allclose(evoked.baseline, (evoked.tmin, evoked.tmax))
    assert_allclose(evoked.data, np.zeros_like(evoked.data))

    # Test that the .baseline attribute changes if we apply a different
    # baseline now.
    baseline = (None, 0)
    evoked.apply_baseline(baseline)
    assert_allclose(evoked.baseline, (evoked.tmin, 0))

    # By default for our test file, no baseline should be set upon reading
    evoked = read_evokeds(fname, condition=0)
    assert evoked.baseline is None

    # Test that the .baseline attribute is set when we call read_evokeds()
    # with a `baseline` parameter.
    baseline = (-0.2, -0.1)
    evoked = read_evokeds(fname, condition=0, baseline=baseline)
    assert_allclose(evoked.baseline, baseline)

    # Test that the .baseline attribute survives an I/O roundtrip.
    evoked = read_evokeds(fname, condition=0)
    baseline = (-0.2, -0.1)
    evoked.apply_baseline(baseline)
    assert_allclose(evoked.baseline, baseline)

    tmp_fname = tmp_path / 'test-ave.fif'
    evoked.save(tmp_fname)
    evoked_read = read_evokeds(tmp_fname, condition=0)
    assert_allclose(evoked_read.baseline, evoked.baseline)

    # We shouldn't be able to remove a baseline correction after it has been
    # applied.
    evoked = read_evokeds(fname, condition=0)
    baseline = (-0.2, -0.1)
    evoked.apply_baseline(baseline)
    with pytest.raises(ValueError, match='already been baseline-corrected'):
        evoked.apply_baseline(None)


def test_hilbert():
    """Test hilbert on raw, epochs, and evoked."""
    raw = read_raw_fif(raw_fname).load_data()
    raw.del_proj()
    raw.pick_channels(raw.ch_names[:2])
    events = read_events(event_name)
    epochs = Epochs(raw, events)
    with pytest.raises(RuntimeError, match='requires epochs data to be load'):
        epochs.apply_hilbert()
    epochs.load_data()
    evoked = epochs.average()
    raw_hilb = raw.apply_hilbert()
    epochs_hilb = epochs.apply_hilbert()
    evoked_hilb = evoked.copy().apply_hilbert()
    evoked_hilb_2_data = epochs_hilb.get_data().mean(0)
    assert_allclose(evoked_hilb.data, evoked_hilb_2_data)
    # This one is only approximate because of edge artifacts
    evoked_hilb_3 = Epochs(raw_hilb, events).average()
    corr = np.corrcoef(np.abs(evoked_hilb_3.data.ravel()),
                       np.abs(evoked_hilb.data.ravel()))[0, 1]
    assert 0.96 < corr < 0.98
    # envelope=True mode
    evoked_hilb_env = evoked.apply_hilbert(envelope=True)
    assert_allclose(evoked_hilb_env.data, np.abs(evoked_hilb.data))


def test_apply_function_evk():
    """Check the apply_function method for evoked data."""
    # create fake evoked data to use for checking apply_function
    data = np.random.rand(10, 1000)
    info = create_info(10, 1000., 'eeg')
    evoked = EvokedArray(data, info)
    evoked_data = evoked.data.copy()
    # check apply_function channel-wise

    def fun(data, multiplier):
        return data * multiplier

    mult = -1
    applied = evoked.apply_function(fun, n_jobs=1, multiplier=mult)
    assert np.shape(applied.data) == np.shape(evoked_data)
    assert np.equal(applied.data, evoked_data * mult).all()
