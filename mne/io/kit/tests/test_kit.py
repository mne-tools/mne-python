# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_allclose)
import pytest
from scipy import linalg
import scipy.io

import mne
from mne import pick_types, Epochs, find_events, read_events
from mne.datasets.testing import requires_testing_data
from mne.transforms import apply_trans
from mne.utils import assert_dig_allclose
from mne.io import read_raw_fif, read_raw_kit, read_epochs_kit
from mne.io.constants import FIFF
from mne.io.kit.coreg import read_sns
from mne.io.kit.constants import KIT
from mne.io.tests.test_raw import _test_raw_reader
from mne.surface import _get_ico_surface
from mne.io.kit import __file__ as _KIT_INIT_FILE

data_dir = op.join(op.dirname(_KIT_INIT_FILE), 'tests', 'data')
sqd_path = op.join(data_dir, 'test.sqd')
sqd_umd_path = op.join(data_dir, 'test_umd-raw.sqd')
epochs_path = op.join(data_dir, 'test-epoch.raw')
events_path = op.join(data_dir, 'test-eve.txt')
mrk_path = op.join(data_dir, 'test_mrk.sqd')
mrk2_path = op.join(data_dir, 'test_mrk_pre.sqd')
mrk3_path = op.join(data_dir, 'test_mrk_post.sqd')
elp_txt_path = op.join(data_dir, 'test_elp.txt')
hsp_txt_path = op.join(data_dir, 'test_hsp.txt')
elp_path = op.join(data_dir, 'test.elp')
hsp_path = op.join(data_dir, 'test.hsp')

data_path = mne.datasets.testing.data_path(download=False)
sqd_as_path = op.join(data_path, 'KIT', 'test_as-raw.con')
yokogawa_path = op.join(
    data_path, 'KIT', 'ArtificalSignalData_Yokogawa_1khz.con')
ricoh_path = op.join(
    data_path, 'KIT', 'ArtificalSignalData_RICOH_1khz.con')
ricoh_systems_paths = [op.join(
    data_path, 'KIT', 'Example_PQA160C_1001-export_anonymyze.con')]
ricoh_systems_paths += [op.join(
    data_path, 'KIT', 'Example_RICOH160-1_10020-export_anonymyze.con')]
ricoh_systems_paths += [op.join(
    data_path, 'KIT', 'Example_RICOH160-1_10021-export_anonymyze.con')]
berlin_path = op.join(data_path, 'KIT', 'data_berlin.con')


@requires_testing_data
def test_data(tmpdir):
    """Test reading raw kit files."""
    pytest.raises(TypeError, read_raw_kit, epochs_path)
    pytest.raises(TypeError, read_epochs_kit, sqd_path)
    pytest.raises(ValueError, read_raw_kit, sqd_path, mrk_path, elp_txt_path)
    pytest.raises(ValueError, read_raw_kit, sqd_path, None, None, None,
                  list(range(200, 190, -1)))
    pytest.raises(ValueError, read_raw_kit, sqd_path, None, None, None,
                  list(range(167, 159, -1)), '*', 1, True)
    # check functionality
    raw_mrk = read_raw_kit(sqd_path, [mrk2_path, mrk3_path], elp_txt_path,
                           hsp_txt_path)
    assert raw_mrk.info['description'] == \
        'NYU 160ch System since Jan24 2009 (34) V2R004 EQ1160C'
    raw_py = _test_raw_reader(read_raw_kit, input_fname=sqd_path, mrk=mrk_path,
                              elp=elp_txt_path, hsp=hsp_txt_path,
                              stim=list(range(167, 159, -1)), slope='+',
                              stimthresh=1)
    assert 'RawKIT' in repr(raw_py)
    assert_equal(raw_mrk.info['kit_system_id'], KIT.SYSTEM_NYU_2010)

    # check number/kind of channels
    assert_equal(len(raw_py.info['chs']), 193)
    kit_channels = (('kind', {FIFF.FIFFV_MEG_CH: 157, FIFF.FIFFV_REF_MEG_CH: 3,
                              FIFF.FIFFV_MISC_CH: 32, FIFF.FIFFV_STIM_CH: 1}),
                    ('coil_type', {FIFF.FIFFV_COIL_KIT_GRAD: 157,
                                   FIFF.FIFFV_COIL_KIT_REF_MAG: 3,
                                   FIFF.FIFFV_COIL_NONE: 33}))
    for label, target in kit_channels:
        actual = {id_: sum(ch[label] == id_ for ch in raw_py.info['chs']) for
                  id_ in target.keys()}
        assert_equal(actual, target)

    # Test stim channel
    raw_stim = read_raw_kit(sqd_path, mrk_path, elp_txt_path, hsp_txt_path,
                            stim='<', preload=False)
    for raw in [raw_py, raw_stim, raw_mrk]:
        stim_pick = pick_types(raw.info, meg=False, ref_meg=False,
                               stim=True, exclude='bads')
        stim1, _ = raw[stim_pick]
        stim2 = np.array(raw.read_stim_ch(), ndmin=2)
        assert_array_equal(stim1, stim2)

    # Binary file only stores the sensor channels
    py_picks = pick_types(raw_py.info, meg=True, exclude='bads')
    raw_bin = op.join(data_dir, 'test_bin_raw.fif')
    raw_bin = read_raw_fif(raw_bin, preload=True)
    bin_picks = pick_types(raw_bin.info, meg=True, stim=True, exclude='bads')
    data_bin, _ = raw_bin[bin_picks]
    data_py, _ = raw_py[py_picks]

    # this .mat was generated using the Yokogawa MEG Reader
    data_Ykgw = op.join(data_dir, 'test_Ykgw.mat')
    data_Ykgw = scipy.io.loadmat(data_Ykgw)['data']
    data_Ykgw = data_Ykgw[py_picks]

    assert_array_almost_equal(data_py, data_Ykgw)

    py_picks = pick_types(raw_py.info, meg=True, stim=True, ref_meg=False,
                          exclude='bads')
    data_py, _ = raw_py[py_picks]
    assert_array_almost_equal(data_py, data_bin)

    # KIT-UMD data
    _test_raw_reader(read_raw_kit, input_fname=sqd_umd_path, test_rank='less')
    raw = read_raw_kit(sqd_umd_path)
    assert raw.info['description'] == \
        'University of Maryland/Kanazawa Institute of Technology/160-channel MEG System (53) V2R004 PQ1160R'  # noqa: E501
    assert_equal(raw.info['kit_system_id'], KIT.SYSTEM_UMD_2014_12)
    # check number/kind of channels
    assert_equal(len(raw.info['chs']), 193)
    for label, target in kit_channels:
        actual = {id_: sum(ch[label] == id_ for ch in raw.info['chs']) for
                  id_ in target.keys()}
        assert_equal(actual, target)

    # KIT Academia Sinica
    raw = read_raw_kit(sqd_as_path, slope='+')
    assert raw.info['description'] == \
        'Academia Sinica/Institute of Linguistics//Magnetoencephalograph System (261) V2R004 PQ1160R-N2'  # noqa: E501
    assert_equal(raw.info['kit_system_id'], KIT.SYSTEM_AS_2008)
    assert_equal(raw.info['chs'][100]['ch_name'], 'MEG 101')
    assert_equal(raw.info['chs'][100]['kind'], FIFF.FIFFV_MEG_CH)
    assert_equal(raw.info['chs'][100]['coil_type'], FIFF.FIFFV_COIL_KIT_GRAD)
    assert_equal(raw.info['chs'][157]['ch_name'], 'MEG 158')
    assert_equal(raw.info['chs'][157]['kind'], FIFF.FIFFV_REF_MEG_CH)
    assert_equal(raw.info['chs'][157]['coil_type'],
                 FIFF.FIFFV_COIL_KIT_REF_MAG)
    assert_equal(raw.info['chs'][160]['ch_name'], 'EEG 001')
    assert_equal(raw.info['chs'][160]['kind'], FIFF.FIFFV_EEG_CH)
    assert_equal(raw.info['chs'][160]['coil_type'], FIFF.FIFFV_COIL_EEG)
    assert_array_equal(find_events(raw), [[91, 0, 2]])


def _assert_sinusoid(data, t, freq, amp, msg):
    __tracebackhide__ = True
    sinusoid = np.exp(2j * np.pi * freq * t) * amp
    phase = np.angle(np.dot(data, sinusoid))
    sinusoid = np.cos(2 * np.pi * freq * t - phase) * amp
    assert_allclose(data, sinusoid, rtol=0.05, atol=amp * 1e-3, err_msg=msg)


@requires_testing_data
@pytest.mark.parametrize('fname, desc', [
    (yokogawa_path, 'Meg160/Analysis (1001) V3R000 PQA160C'),
    (ricoh_path, 'Meg160/Analysis (1001) V3R000 PQA160C'),
])
def test_ricoh_data(tmpdir, fname, desc):
    """Test reading channel names and dig information from Ricoh systems."""
    raw = read_raw_kit(fname, standardize_names=True)
    assert raw.ch_names[0] == 'MEG 001'
    raw = read_raw_kit(fname, standardize_names=False, verbose='debug')
    assert raw.info['description'] == desc
    assert_allclose(raw.times[-1], 5. - 1. / raw.info['sfreq'])
    assert raw.ch_names[0] == 'LF31'
    eeg_picks = pick_types(raw.info, meg=False, eeg=True)
    assert len(eeg_picks) == 45
    assert len(raw.info['dig']) == 8 + len(eeg_picks) - 2  # EKG+ and E no pos
    bad_dig = [ch['ch_name'] for ci, ch in enumerate(raw.info['chs'])
               if ci in eeg_picks and (ch['loc'][:3] == 0).all()]
    assert bad_dig == ['EKG+', 'E']
    assert not any(np.allclose(d['r'], 0.) for d in raw.info['dig'])
    assert_allclose(
        raw.info['dev_head_t']['trans'],
        [[0.998311, -0.056923, 0.01164, 0.001403],
         [0.054469, 0.986653, 0.153458, 0.0044],
         [-0.02022, -0.152564, 0.988087, 0.018634],
         [0., 0., 0., 1.]], atol=1e-5)
    data = raw.get_data()
    # 1 pT 10 Hz on the first channel
    assert raw.info['chs'][0]['coil_type'] == FIFF.FIFFV_COIL_KIT_GRAD
    _assert_sinusoid(data[0], raw.times, 10, 1e-12, '1 pT 10 Hz MEG')
    assert_allclose(data[1:160], 0., atol=1e-13)
    # 1 V 5 Hz analog
    assert raw.info['chs'][186]['coil_type'] == FIFF.FIFFV_COIL_EEG
    _assert_sinusoid(data[160], raw.times, 5, 1, '1 V 5 Hz analog')
    assert_allclose(data[161:185], 0., atol=1e-20)
    # 50 uV 8 Hz plus 1.6 mV offset
    assert raw.info['chs'][186]['coil_type'] == FIFF.FIFFV_COIL_EEG
    eeg_data = data[186]
    assert_allclose(eeg_data.mean(), 1.6e-3, atol=1e-5)  # offset
    eeg_data = eeg_data - eeg_data.mean()
    _assert_sinusoid(eeg_data, raw.times, 8, 50e-6, '50 uV 8 Hz EEG')
    assert_allclose(data[187:-1], 0., atol=1e-20)
    assert_allclose(data[-1], 254.5, atol=0.51)


def test_epochs():
    """Test reading epoched SQD file."""
    raw = read_raw_kit(sqd_path, stim=None)
    events = read_events(events_path)
    raw_epochs = Epochs(raw, events, None, tmin=0, tmax=.099, baseline=None)
    data1 = raw_epochs.get_data()
    epochs = read_epochs_kit(epochs_path, events_path)
    data11 = epochs.get_data()
    assert_array_equal(data1, data11)


def test_raw_events():
    """Test creating stim channel from raw SQD file."""
    def evts(a, b, c, d, e, f=None):
        out = [[269, a, b], [281, b, c], [1552, c, d], [1564, d, e]]
        if f is not None:
            out.append([2000, e, f])
        return out

    raw = read_raw_kit(sqd_path)
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(255, 254, 255, 254, 255, 0))

    raw = read_raw_kit(sqd_path, slope='+')
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(0, 1, 0, 1, 0))

    raw = read_raw_kit(sqd_path, stim='<', slope='+')
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(0, 128, 0, 128, 0))

    raw = read_raw_kit(sqd_path, stim='<', slope='+', stim_code='channel')
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(0, 160, 0, 160, 0))

    raw = read_raw_kit(sqd_path, stim=range(160, 162), slope='+',
                       stim_code='channel')
    assert_array_equal(find_events(raw, output='step', consecutive=True),
                       evts(0, 160, 0, 160, 0))


def test_ch_loc():
    """Test raw kit loc."""
    raw_py = read_raw_kit(sqd_path, mrk_path, elp_txt_path, hsp_txt_path,
                          stim='<')
    raw_bin = read_raw_fif(op.join(data_dir, 'test_bin_raw.fif'))

    ch_py = np.array([ch['loc'] for ch in
                      raw_py._raw_extras[0]['channels'][:160]])
    # ch locs stored as m, not mm
    ch_py[:, :3] *= 1e3
    ch_sns = read_sns(op.join(data_dir, 'sns.txt'))
    assert_array_almost_equal(ch_py, ch_sns, 2)

    assert_array_almost_equal(raw_py.info['dev_head_t']['trans'],
                              raw_bin.info['dev_head_t']['trans'], 4)
    for py_ch, bin_ch in zip(raw_py.info['chs'], raw_bin.info['chs']):
        if bin_ch['ch_name'].startswith('MEG'):
            # the stored ch locs have more precision than the sns.txt
            assert_array_almost_equal(py_ch['loc'], bin_ch['loc'], decimal=2)

    # test when more than one marker file provided
    mrks = [mrk_path, mrk2_path, mrk3_path]
    read_raw_kit(sqd_path, mrks, elp_txt_path, hsp_txt_path, preload=False)
    # this dataset does not have the equivalent set of points :(
    raw_bin.info['dig'] = raw_bin.info['dig'][:8]
    raw_py.info['dig'] = raw_py.info['dig'][:8]
    assert_dig_allclose(raw_py.info, raw_bin.info)


def test_hsp_elp():
    """Test KIT usage of *.elp and *.hsp files against *.txt files."""
    raw_txt = read_raw_kit(sqd_path, mrk_path, elp_txt_path, hsp_txt_path)
    raw_elp = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path)

    # head points
    pts_txt = np.array([dig_point['r'] for dig_point in raw_txt.info['dig']])
    pts_elp = np.array([dig_point['r'] for dig_point in raw_elp.info['dig']])
    assert_array_almost_equal(pts_elp, pts_txt, decimal=5)

    # transforms
    trans_txt = raw_txt.info['dev_head_t']['trans']
    trans_elp = raw_elp.info['dev_head_t']['trans']
    assert_array_almost_equal(trans_elp, trans_txt, decimal=5)

    # head points in device space
    pts_txt_in_dev = apply_trans(linalg.inv(trans_txt), pts_txt)
    pts_elp_in_dev = apply_trans(linalg.inv(trans_elp), pts_elp)
    assert_array_almost_equal(pts_elp_in_dev, pts_txt_in_dev, decimal=5)


def test_decimate(tmpdir):
    """Test decimation of digitizer headshapes with too many points."""
    # load headshape and convert to meters
    hsp_mm = _get_ico_surface(5)['rr'] * 100
    hsp_m = hsp_mm / 1000.

    # save headshape to a file in mm in temporary directory
    tempdir = str(tmpdir)
    sphere_hsp_path = op.join(tempdir, 'test_sphere.txt')
    np.savetxt(sphere_hsp_path, hsp_mm)

    # read in raw data using spherical hsp, and extract new hsp
    with pytest.warns(RuntimeWarning,
                      match='was automatically downsampled .* FastScan'):
        raw = read_raw_kit(sqd_path, mrk_path, elp_txt_path, sphere_hsp_path)
    # collect headshape from raw (should now be in m)
    hsp_dec = np.array([dig['r'] for dig in raw.info['dig']])[8:]

    # with 10242 points and _decimate_points set to resolution of 5 mm, hsp_dec
    # should be a bit over 5000 points. If not, something is wrong or
    # decimation resolution has been purposefully changed
    assert len(hsp_dec) > 5000

    # should have similar size, distance from center
    dist = np.sqrt(np.sum((hsp_m - np.mean(hsp_m, axis=0))**2, axis=1))
    dist_dec = np.sqrt(np.sum((hsp_dec - np.mean(hsp_dec, axis=0))**2, axis=1))
    hsp_rad = np.mean(dist)
    hsp_dec_rad = np.mean(dist_dec)
    assert_array_almost_equal(hsp_rad, hsp_dec_rad, decimal=3)


@requires_testing_data
@pytest.mark.parametrize('fname, desc, system_id', [
    (ricoh_systems_paths[0],
        'Meg160/Analysis (1001) V2R004 PQA160C', 1001),
    (ricoh_systems_paths[1],
        'RICOH MEG System (10020) V3R000 RICOH160-1', 10020),
    (ricoh_systems_paths[2],
        'RICOH MEG System (10021) V3R000 RICOH160-1', 10021),
])
def test_ricoh_systems(tmpdir, fname, desc, system_id):
    """Test reading channel names and dig information from Ricoh systems."""
    raw = read_raw_kit(fname, standardize_names=False)
    assert raw.info['description'] == desc
    assert raw.info['kit_system_id'] == system_id


@requires_testing_data
def test_berlin():
    """Test data from Berlin."""
    # gh-8535
    raw = read_raw_kit(berlin_path)
    assert raw.info['description'] == 'Physikalisch Technische Bundesanstalt, Berlin/128-channel MEG System (124) V2R004 PQ1128R-N2'  # noqa: E501
    assert raw.info['kit_system_id'] == 124
    assert raw.info['highpass'] == 0.
    assert raw.info['lowpass'] == 200.
    assert raw.info['sfreq'] == 500.
    n = int(round(28.77 * raw.info['sfreq']))
    meg = raw.get_data('MEG 003', n, n + 1)[0, 0]
    assert_allclose(meg, -8.89e-12, rtol=1e-3)
    eeg = raw.get_data('E14', n, n + 1)[0, 0]
    assert_allclose(eeg, -2.55, rtol=1e-3)
