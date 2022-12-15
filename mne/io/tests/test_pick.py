from copy import deepcopy
import os.path as op

from numpy.testing import assert_array_equal, assert_equal
import pytest
import numpy as np

from mne import (pick_channels_regexp, pick_types, Epochs,
                 read_forward_solution, rename_channels,
                 pick_info, pick_channels, create_info, make_ad_hoc_cov)
from mne import __file__ as _root_init_fname
from mne.io import (read_raw_fif, RawArray, read_raw_bti, read_raw_kit,
                    read_info)
from mne.channels import make_standard_montage
from mne.preprocessing import compute_current_source_density
from mne.io.pick import (channel_indices_by_type, channel_type,
                         pick_types_forward, _picks_by_type, _picks_to_idx,
                         _contains_ch_type, pick_channels_cov,
                         _get_channel_types, get_channel_type_constants,
                         _DATA_CH_TYPES_SPLIT)
from mne.io.constants import FIFF
from mne.datasets import testing
from mne.utils import catch_logging, assert_object_equal

data_path = testing.data_path(download=False)
fname_meeg = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_mc = op.join(data_path, 'SSS', 'test_move_anon_movecomp_raw_sss.fif')

io_dir = op.join(op.dirname(__file__), '..')
ctf_fname = op.join(io_dir, 'tests', 'data', 'test_ctf_raw.fif')
fif_fname = op.join(io_dir, 'tests', 'data', 'test_raw.fif')


def _picks_by_type_old(info, meg_combined=False, ref_meg=False,
                       exclude='bads'):
    """Use the old, slower _picks_by_type code."""
    picks_list = []
    has = [_contains_ch_type(info, k) for k in _DATA_CH_TYPES_SPLIT]
    has = dict(zip(_DATA_CH_TYPES_SPLIT, has))
    if has['mag'] and (meg_combined is not True or not has['grad']):
        picks_list.append(
            ('mag', pick_types(info, meg='mag', eeg=False, stim=False,
                               ref_meg=ref_meg, exclude=exclude))
        )
    if has['grad'] and (meg_combined is not True or not has['mag']):
        picks_list.append(
            ('grad', pick_types(info, meg='grad', eeg=False, stim=False,
                                ref_meg=ref_meg, exclude=exclude))
        )
    if has['mag'] and has['grad'] and meg_combined is True:
        picks_list.append(
            ('meg', pick_types(info, meg=True, eeg=False, stim=False,
                               ref_meg=ref_meg, exclude=exclude))
        )
    for ch_type in _DATA_CH_TYPES_SPLIT:
        if ch_type in ['grad', 'mag']:  # exclude just MEG channels
            continue
        if has[ch_type]:
            picks_list.append(
                (ch_type, pick_types(info, meg=False, stim=False,
                                     ref_meg=ref_meg, exclude=exclude,
                                     **{ch_type: True}))
            )
    return picks_list


def _channel_type_old(info, idx):
    """Get channel type using old, slower scheme."""
    ch = info['chs'][idx]

    # iterate through all defined channel types until we find a match with ch
    # go in order from most specific (most rules entries) to least specific
    channel_types = sorted(get_channel_type_constants().items(),
                           key=lambda x: len(x[1]), reverse=True)
    for t, rules in channel_types:
        for key, vals in rules.items():  # all keys must match the values
            if ch.get(key, None) not in np.array(vals):
                break  # not channel type t, go to next iteration
        else:
            return t

    raise ValueError(f'Unknown channel type for {ch["ch_name"]}')


def _assert_channel_types(info):
    for k in range(info['nchan']):
        a, b = channel_type(info, k), _channel_type_old(info, k)
        assert a == b


def test_pick_refs():
    """Test picking of reference sensors."""
    infos = list()
    # KIT
    kit_dir = op.join(io_dir, 'kit', 'tests', 'data')
    sqd_path = op.join(kit_dir, 'test.sqd')
    mrk_path = op.join(kit_dir, 'test_mrk.sqd')
    elp_path = op.join(kit_dir, 'test_elp.txt')
    hsp_path = op.join(kit_dir, 'test_hsp.txt')
    raw_kit = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path)
    infos.append(raw_kit.info)
    # BTi
    bti_dir = op.join(io_dir, 'bti', 'tests', 'data')
    bti_pdf = op.join(bti_dir, 'test_pdf_linux')
    bti_config = op.join(bti_dir, 'test_config_linux')
    bti_hs = op.join(bti_dir, 'test_hs_linux')
    raw_bti = read_raw_bti(bti_pdf, bti_config, bti_hs, preload=False)
    infos.append(raw_bti.info)
    # CTF
    fname_ctf_raw = op.join(io_dir, 'tests', 'data', 'test_ctf_comp_raw.fif')
    raw_ctf = read_raw_fif(fname_ctf_raw)
    raw_ctf.apply_gradient_compensation(2)
    for info in infos:
        info['bads'] = []
        _assert_channel_types(info)
        with pytest.raises(ValueError, match="'planar2'] or bool, not foo"):
            pick_types(info, meg='foo')
        with pytest.raises(ValueError, match="'planar2', 'auto'] or bool,"):
            pick_types(info, ref_meg='foo')
        picks_meg_ref = pick_types(info, meg=True, ref_meg=True)
        picks_meg = pick_types(info, meg=True, ref_meg=False)
        picks_ref = pick_types(info, meg=False, ref_meg=True)
        assert_array_equal(picks_meg_ref,
                           np.sort(np.concatenate([picks_meg, picks_ref])))
        picks_grad = pick_types(info, meg='grad', ref_meg=False)
        picks_ref_grad = pick_types(info, meg=False, ref_meg='grad')
        picks_meg_ref_grad = pick_types(info, meg='grad', ref_meg='grad')
        assert_array_equal(picks_meg_ref_grad,
                           np.sort(np.concatenate([picks_grad,
                                                   picks_ref_grad])))
        picks_mag = pick_types(info, meg='mag', ref_meg=False)
        picks_ref_mag = pick_types(info, meg=False, ref_meg='mag')
        picks_meg_ref_mag = pick_types(info, meg='mag', ref_meg='mag')
        assert_array_equal(picks_meg_ref_mag,
                           np.sort(np.concatenate([picks_mag,
                                                   picks_ref_mag])))
        assert_array_equal(picks_meg,
                           np.sort(np.concatenate([picks_mag, picks_grad])))
        assert_array_equal(picks_ref,
                           np.sort(np.concatenate([picks_ref_mag,
                                                   picks_ref_grad])))
        assert_array_equal(picks_meg_ref, np.sort(np.concatenate(
            [picks_grad, picks_mag, picks_ref_grad, picks_ref_mag])))

        for pick in (picks_meg_ref, picks_meg, picks_ref,
                     picks_grad, picks_ref_grad, picks_meg_ref_grad,
                     picks_mag, picks_ref_mag, picks_meg_ref_mag):
            if len(pick) > 0:
                pick_info(info, pick)

    # test CTF expected failures directly
    info = raw_ctf.info
    info['bads'] = []
    picks_meg_ref = pick_types(info, meg=True, ref_meg=True)
    picks_meg = pick_types(info, meg=True, ref_meg=False)
    picks_ref = pick_types(info, meg=False, ref_meg=True)
    picks_mag = pick_types(info, meg='mag', ref_meg=False)
    picks_ref_mag = pick_types(info, meg=False, ref_meg='mag')
    picks_meg_ref_mag = pick_types(info, meg='mag', ref_meg='mag')
    for pick in (picks_meg_ref, picks_ref, picks_ref_mag, picks_meg_ref_mag):
        if len(pick) > 0:
            pick_info(info, pick)

    for pick in (picks_meg, picks_mag):
        if len(pick) > 0:
            with catch_logging() as log:
                pick_info(info, pick, verbose=True)
            assert ('Removing {} compensators'.format(len(info['comps']))
                    in log.getvalue())
    picks_ref_grad = pick_types(info, meg=False, ref_meg='grad')
    assert set(picks_ref_mag) == set(picks_ref)
    assert len(picks_ref_grad) == 0
    all_meg = np.arange(3, 306)
    assert_array_equal(np.concatenate([picks_ref, picks_meg]), all_meg)
    assert_array_equal(picks_meg_ref_mag, all_meg)


def test_pick_channels_regexp():
    """Test pick with regular expression."""
    ch_names = ['MEG 2331', 'MEG 2332', 'MEG 2333']
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...1'), [0])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...[2-3]'), [1, 2])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG *'), [0, 1, 2])


def assert_indexing(info, picks_by_type, ref_meg=False, all_data=True):
    """Assert our indexing functions work properly."""
    # First that our old and new channel typing functions are equivalent
    _assert_channel_types(info)
    # Next that channel_indices_by_type works
    if not ref_meg:
        idx = channel_indices_by_type(info)
        for key in idx:
            for p in picks_by_type:
                if key == p[0]:
                    assert_array_equal(idx[key], p[1])
                    break
            else:
                assert len(idx[key]) == 0
    # Finally, picks_by_type (if relevant)
    if not all_data:
        picks_by_type = [p for p in picks_by_type
                         if p[0] in _DATA_CH_TYPES_SPLIT]
    picks_by_type = [(p[0], np.array(p[1], int)) for p in picks_by_type]
    actual = _picks_by_type(info, ref_meg=ref_meg)
    assert_object_equal(actual, picks_by_type)
    if not ref_meg and idx['hbo']:  # our old code had a bug
        with pytest.raises(TypeError, match='unexpected keyword argument'):
            _picks_by_type_old(info, ref_meg=ref_meg)
    else:
        old = _picks_by_type_old(info, ref_meg=ref_meg)
        assert_object_equal(old, picks_by_type)
    # test bads
    info = info.copy()
    info['bads'] = [info['chs'][picks_by_type[0][1][0]]['ch_name']]
    picks_by_type = deepcopy(picks_by_type)
    picks_by_type[0] = (picks_by_type[0][0], picks_by_type[0][1][1:])
    actual = _picks_by_type(info, ref_meg=ref_meg)
    assert_object_equal(actual, picks_by_type)


def test_pick_seeg_ecog():
    """Test picking with sEEG and ECoG."""
    names = 'A1 A2 Fz O OTp1 OTp2 E1 OTp3 E2 E3'.split()
    types = 'mag mag eeg eeg seeg seeg ecog seeg ecog ecog'.split()
    info = create_info(names, 1024., types)
    picks_by_type = [('mag', [0, 1]), ('eeg', [2, 3]),
                     ('seeg', [4, 5, 7]), ('ecog', [6, 8, 9])]
    assert_indexing(info, picks_by_type)
    assert_array_equal(pick_types(info, meg=False, seeg=True), [4, 5, 7])
    for i, t in enumerate(types):
        assert_equal(channel_type(info, i), types[i])
    raw = RawArray(np.zeros((len(names), 10)), info)
    events = np.array([[1, 0, 0], [2, 0, 0]])
    epochs = Epochs(raw, events=events, event_id={'event': 0},
                    tmin=-1e-5, tmax=1e-5,
                    baseline=(0, 0))  # only one sample
    evoked = epochs.average(pick_types(epochs.info, meg=True, seeg=True))
    e_seeg = evoked.copy().pick_types(meg=False, seeg=True)
    for lt, rt in zip(e_seeg.ch_names, [names[4], names[5], names[7]]):
        assert lt == rt
    # Deal with constant debacle
    raw = read_raw_fif(op.join(io_dir, 'tests', 'data',
                               'test_chpi_raw_sss.fif'))
    assert_equal(len(pick_types(raw.info, meg=False, seeg=True, ecog=True)), 0)


def test_pick_dbs():
    """Test picking with DBS."""
    # gh-8739
    names = 'A1 A2 Fz O OTp1 OTp2 OTp3'.split()
    types = 'mag mag eeg eeg dbs dbs dbs'.split()
    info = create_info(names, 1024., types)
    picks_by_type = [('mag', [0, 1]), ('eeg', [2, 3]), ('dbs', [4, 5, 6])]
    assert_indexing(info, picks_by_type)
    assert_array_equal(pick_types(info, meg=False, dbs=True), [4, 5, 6])
    for i, t in enumerate(types):
        assert channel_type(info, i) == types[i]
    raw = RawArray(np.zeros((len(names), 7)), info)
    events = np.array([[1, 0, 0], [2, 0, 0]])
    epochs = Epochs(raw, events=events, event_id={'event': 0},
                    tmin=-1e-5, tmax=1e-5,
                    baseline=(0, 0))  # only one sample
    evoked = epochs.average(pick_types(epochs.info, meg=True, dbs=True))
    e_dbs = evoked.copy().pick_types(meg=False, dbs=True)
    for lt, rt in zip(e_dbs.ch_names, [names[4], names[5], names[6]]):
        assert lt == rt
    raw = read_raw_fif(op.join(io_dir, 'tests', 'data',
                               'test_chpi_raw_sss.fif'))
    assert len(pick_types(raw.info, meg=False, dbs=True)) == 0


def test_pick_chpi():
    """Test picking cHPI."""
    # Make sure we don't mis-classify cHPI channels
    info = read_info(op.join(io_dir, 'tests', 'data', 'test_chpi_raw_sss.fif'))
    _assert_channel_types(info)
    channel_types = _get_channel_types(info)
    assert 'chpi' in channel_types
    assert 'seeg' not in channel_types
    assert 'ecog' not in channel_types


def test_pick_csd():
    """Test picking current source density channels."""
    # Make sure we don't mis-classify cHPI channels
    names = ['MEG 2331', 'MEG 2332', 'MEG 2333', 'A1', 'A2', 'Fz']
    types = 'mag mag grad csd csd csd'.split()
    info = create_info(names, 1024., types)
    picks_by_type = [('mag', [0, 1]), ('grad', [2]), ('csd', [3, 4, 5])]
    assert_indexing(info, picks_by_type, all_data=False)


def test_pick_bio():
    """Test picking BIO channels."""
    names = 'A1 A2 Fz O BIO1 BIO2 BIO3'.split()
    types = 'mag mag eeg eeg bio bio bio'.split()
    info = create_info(names, 1024., types)
    picks_by_type = [('mag', [0, 1]), ('eeg', [2, 3]), ('bio', [4, 5, 6])]
    assert_indexing(info, picks_by_type, all_data=False)


def test_pick_fnirs():
    """Test picking fNIRS channels."""
    names = 'A1 A2 Fz O hbo1 hbo2 hbr1 fnirsRaw1 fnirsRaw2 fnirsOD1'.split()
    types = 'mag mag eeg eeg hbo hbo hbr fnirs_cw_' \
            'amplitude fnirs_cw_amplitude fnirs_od'.split()
    info = create_info(names, 1024., types)
    picks_by_type = [('mag', [0, 1]), ('eeg', [2, 3]),
                     ('hbo', [4, 5]), ('hbr', [6]),
                     ('fnirs_cw_amplitude', [7, 8]), ('fnirs_od', [9])]
    assert_indexing(info, picks_by_type)


def test_pick_ref():
    """Test picking ref_meg channels."""
    info = read_info(ctf_fname)
    picks_by_type = [('stim', [0]), ('eog', [306, 307]), ('ecg', [308]),
                     ('misc', [1]),
                     ('mag', np.arange(31, 306)),
                     ('ref_meg', np.arange(2, 31))]
    assert_indexing(info, picks_by_type, all_data=False)
    picks_by_type.append(('mag', np.concatenate([picks_by_type.pop(-1)[1],
                                                 picks_by_type.pop(-1)[1]])))
    assert_indexing(info, picks_by_type, ref_meg=True, all_data=False)


def _check_fwd_n_chan_consistent(fwd, n_expected):
    n_ok = len(fwd['info']['ch_names'])
    n_sol = fwd['sol']['data'].shape[0]
    assert_equal(n_expected, n_sol)
    assert_equal(n_expected, n_ok)


@testing.requires_testing_data
def test_pick_forward_seeg_ecog():
    """Test picking forward with SEEG and ECoG."""
    fwd = read_forward_solution(fname_meeg)
    counts = channel_indices_by_type(fwd['info'])
    for key in counts.keys():
        counts[key] = len(counts[key])
    counts['meg'] = counts['mag'] + counts['grad']
    fwd_ = pick_types_forward(fwd, meg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['meg'])
    fwd_ = pick_types_forward(fwd, meg=False, eeg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['eeg'])
    # should raise exception related to emptiness
    pytest.raises(ValueError, pick_types_forward, fwd, meg=False, seeg=True)
    pytest.raises(ValueError, pick_types_forward, fwd, meg=False, ecog=True)
    # change last chan from EEG to sEEG, second-to-last to ECoG
    ecog_name = 'E1'
    seeg_name = 'OTp1'
    rename_channels(fwd['info'], {'EEG 059': ecog_name})
    rename_channels(fwd['info'], {'EEG 060': seeg_name})
    for ch in fwd['info']['chs']:
        if ch['ch_name'] == seeg_name:
            ch['kind'] = FIFF.FIFFV_SEEG_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_EEG
        elif ch['ch_name'] == ecog_name:
            ch['kind'] = FIFF.FIFFV_ECOG_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_EEG
    fwd['sol']['row_names'][-1] = fwd['info']['chs'][-1]['ch_name']
    fwd['sol']['row_names'][-2] = fwd['info']['chs'][-2]['ch_name']
    counts['eeg'] -= 2
    counts['seeg'] += 1
    counts['ecog'] += 1
    # repick & check
    fwd_seeg = pick_types_forward(fwd, meg=False, seeg=True)
    assert_equal(fwd_seeg['sol']['row_names'], [seeg_name])
    assert_equal(fwd_seeg['info']['ch_names'], [seeg_name])
    # should work fine
    fwd_ = pick_types_forward(fwd, meg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['meg'])
    fwd_ = pick_types_forward(fwd, meg=False, eeg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['eeg'])
    fwd_ = pick_types_forward(fwd, meg=False, seeg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['seeg'])
    fwd_ = pick_types_forward(fwd, meg=False, ecog=True)
    _check_fwd_n_chan_consistent(fwd_, counts['ecog'])


def test_picks_by_channels():
    """Test creating pick_lists."""
    rng = np.random.RandomState(909)

    test_data = rng.random_sample((4, 2000))
    ch_names = ['MEG %03d' % i for i in [1, 2, 3, 4]]
    ch_types = ['grad', 'mag', 'mag', 'eeg']
    sfreq = 250.0
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    _assert_channel_types(info)
    raw = RawArray(test_data, info)

    pick_list = _picks_by_type(raw.info)
    assert_equal(len(pick_list), 3)
    assert_equal(pick_list[0][0], 'mag')
    pick_list2 = _picks_by_type(raw.info, meg_combined=False)
    assert_equal(len(pick_list), len(pick_list2))
    assert_equal(pick_list2[0][0], 'mag')

    pick_list2 = _picks_by_type(raw.info, meg_combined=True)
    assert_equal(len(pick_list), len(pick_list2) + 1)
    assert_equal(pick_list2[0][0], 'meg')

    test_data = rng.random_sample((4, 2000))
    ch_names = ['MEG %03d' % i for i in [1, 2, 3, 4]]
    ch_types = ['mag', 'mag', 'mag', 'mag']
    sfreq = 250.0
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = RawArray(test_data, info)
    # This acts as a set, not an order
    assert_array_equal(pick_channels(info['ch_names'], ['MEG 002', 'MEG 001']),
                       [0, 1])

    # Make sure checks for list input work.
    pytest.raises(ValueError, pick_channels, ch_names, 'MEG 001')
    pytest.raises(ValueError, pick_channels, ch_names, ['MEG 001'], 'hi')

    pick_list = _picks_by_type(raw.info)
    assert_equal(len(pick_list), 1)
    assert_equal(pick_list[0][0], 'mag')
    pick_list2 = _picks_by_type(raw.info, meg_combined=True)
    assert_equal(len(pick_list), len(pick_list2))
    assert_equal(pick_list2[0][0], 'mag')

    # pick_types type check
    with pytest.raises(ValueError, match='must be of type'):
        raw.pick_types(eeg='string')

    # duplicate check
    names = ['MEG 002', 'MEG 002']
    assert len(pick_channels(raw.info['ch_names'], names)) == 1
    assert len(raw.copy().pick_channels(names)[0][0]) == 1


def test_clean_info_bads():
    """Test cleaning info['bads'] when bad_channels are excluded."""
    raw_file = op.join(op.dirname(_root_init_fname), 'io', 'tests', 'data',
                       'test_raw.fif')
    raw = read_raw_fif(raw_file)
    _assert_channel_types(raw.info)

    # select eeg channels
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)

    # select 3 eeg channels as bads
    idx_eeg_bad_ch = picks_eeg[[1, 5, 14]]
    eeg_bad_ch = [raw.info['ch_names'][k] for k in idx_eeg_bad_ch]

    # select meg channels
    picks_meg = pick_types(raw.info, meg=True, eeg=False)

    # select randomly 3 meg channels as bads
    idx_meg_bad_ch = picks_meg[[0, 15, 34]]
    meg_bad_ch = [raw.info['ch_names'][k] for k in idx_meg_bad_ch]

    # simulate the bad channels
    raw.info['bads'] = eeg_bad_ch + meg_bad_ch

    # simulate the call to pick_info excluding the bad eeg channels
    info_eeg = pick_info(raw.info, picks_eeg)

    # simulate the call to pick_info excluding the bad meg channels
    info_meg = pick_info(raw.info, picks_meg)

    assert_equal(info_eeg['bads'], eeg_bad_ch)
    assert_equal(info_meg['bads'], meg_bad_ch)

    info = pick_info(raw.info, picks_meg)
    info._check_consistency()
    info['bads'] += ['EEG 053']
    pytest.raises(RuntimeError, info._check_consistency)
    with pytest.raises(ValueError, match='unique'):
        pick_info(raw.info, [0, 0])


@testing.requires_testing_data
def test_picks_to_idx():
    """Test checking type integrity checks of picks."""
    info = create_info(12, 1000., 'eeg')
    _assert_channel_types(info)
    picks = np.arange(info['nchan'])
    # Array and list
    assert_array_equal(picks, _picks_to_idx(info, picks))
    assert_array_equal(picks, _picks_to_idx(info, list(picks)))
    with pytest.raises(TypeError, match='data type float64 is invalid'):
        _picks_to_idx(info, 1.)
    # None
    assert_array_equal(picks, _picks_to_idx(info, None))
    # Type indexing
    assert_array_equal(picks, _picks_to_idx(info, 'eeg'))
    assert_array_equal(picks, _picks_to_idx(info, ['eeg']))
    # Negative indexing
    assert_array_equal([len(picks) - 1], _picks_to_idx(info, len(picks) - 1))
    assert_array_equal([len(picks) - 1], _picks_to_idx(info, -1))
    assert_array_equal([len(picks) - 1], _picks_to_idx(info, [-1]))
    # Name indexing
    assert_array_equal([2], _picks_to_idx(info, info['ch_names'][2]))
    assert_array_equal(np.arange(5, 9),
                       _picks_to_idx(info, info['ch_names'][5:9]))
    with pytest.raises(ValueError, match='must be >= '):
        _picks_to_idx(info, -len(picks) - 1)
    with pytest.raises(ValueError, match='must be < '):
        _picks_to_idx(info, len(picks))
    with pytest.raises(ValueError, match='could not be interpreted'):
        _picks_to_idx(info, ['a', 'b'])
    with pytest.raises(ValueError, match='could not be interpreted'):
        _picks_to_idx(info, 'b')
    # bads behavior
    info['bads'] = info['ch_names'][1:2]
    picks_good = np.array([0] + list(range(2, 12)))
    assert_array_equal(picks_good, _picks_to_idx(info, None))
    assert_array_equal(picks_good, _picks_to_idx(info, None,
                                                 exclude=info['bads']))
    assert_array_equal(picks, _picks_to_idx(info, None, exclude=()))
    with pytest.raises(ValueError, match=' 1D, got'):
        _picks_to_idx(info, [[1]])
    # MEG types
    info = read_info(fname_mc)
    meg_picks = np.arange(306)
    mag_picks = np.arange(2, 306, 3)
    grad_picks = np.setdiff1d(meg_picks, mag_picks)
    assert_array_equal(meg_picks, _picks_to_idx(info, 'meg'))
    assert_array_equal(meg_picks, _picks_to_idx(info, ('mag', 'grad')))
    assert_array_equal(mag_picks, _picks_to_idx(info, 'mag'))
    assert_array_equal(grad_picks, _picks_to_idx(info, 'grad'))

    info = create_info(['eeg', 'foo'], 1000., 'eeg')
    with pytest.raises(RuntimeError, match='equivalent to channel types'):
        _picks_to_idx(info, 'eeg')
    with pytest.raises(ValueError, match='same length'):
        create_info(['a', 'b'], 1000., dict(hbo=['a'], hbr=['b']))
    info = create_info(['a', 'b'], 1000., ['hbo', 'hbr'])
    assert_array_equal(np.arange(2), _picks_to_idx(info, 'fnirs'))
    assert_array_equal([0], _picks_to_idx(info, 'hbo'))
    assert_array_equal([1], _picks_to_idx(info, 'hbr'))
    info = create_info(['a', 'b'], 1000., ['hbo', 'misc'])
    assert_array_equal(np.arange(len(info['ch_names'])),
                       _picks_to_idx(info, 'all'))
    assert_array_equal([0], _picks_to_idx(info, 'data'))
    info = create_info(['a', 'b'], 1000., ['fnirs_cw_amplitude', 'fnirs_od'])
    assert_array_equal(np.arange(2), _picks_to_idx(info, 'fnirs'))
    assert_array_equal([0], _picks_to_idx(info, 'fnirs_cw_amplitude'))
    assert_array_equal([1], _picks_to_idx(info, 'fnirs_od'))
    info = create_info(['a', 'b'], 1000., ['fnirs_cw_amplitude', 'misc'])
    assert_array_equal(np.arange(len(info['ch_names'])),
                       _picks_to_idx(info, 'all'))
    assert_array_equal([0], _picks_to_idx(info, 'data'))
    info = create_info(['a', 'b'], 1000., ['fnirs_od', 'misc'])
    assert_array_equal(np.arange(len(info['ch_names'])),
                       _picks_to_idx(info, 'all'))
    assert_array_equal([0], _picks_to_idx(info, 'data'))


def test_pick_channels_cov():
    """Test picking channels from a Covariance object."""
    info = create_info(['CH1', 'CH2', 'CH3'], 1., ch_types='eeg')
    cov = make_ad_hoc_cov(info)
    cov['data'] = np.array([1., 2., 3.])

    cov_copy = pick_channels_cov(cov, ['CH2', 'CH1'], ordered=False, copy=True)
    assert cov_copy.ch_names == ['CH1', 'CH2']
    assert_array_equal(cov_copy['data'], [1., 2.])

    # Test re-ordering channels
    cov_copy = pick_channels_cov(cov, ['CH2', 'CH1'], ordered=True, copy=True)
    assert cov_copy.ch_names == ['CH2', 'CH1']
    assert_array_equal(cov_copy['data'], [2., 1.])

    # Test picking in-place
    pick_channels_cov(cov, ['CH2', 'CH1'], copy=False)
    assert cov.ch_names == ['CH1', 'CH2']
    assert_array_equal(cov['data'], [1., 2.])

    # Test whether `method` and `loglik` are dropped when None
    cov['method'] = None
    cov['loglik'] = None
    cov_copy = pick_channels_cov(cov, ['CH1', 'CH2'], copy=True)
    assert 'method' not in cov_copy
    assert 'loglik' not in cov_copy


def test_pick_types_meg():
    """Test pick_types(meg=True)."""
    # info with MEG channels at indices 1, 2, and 4
    info1 = create_info(6, 256, ["eeg", "mag", "grad", "misc", "grad", "hbo"])

    assert list(pick_types(info1, meg=True)) == [1, 2, 4]
    assert list(pick_types(info1, meg=True, eeg=True)) == [0, 1, 2, 4]

    assert list(pick_types(info1, meg=True)) == [1, 2, 4]
    assert not list(pick_types(info1, meg=False))  # empty
    assert list(pick_types(info1, meg='planar1')) == [2]
    assert not list(pick_types(info1, meg='planar2'))  # empty

    # info without any MEG channels
    info2 = create_info(6, 256, ["eeg", "eeg", "eog", "misc", "stim", "hbo"])

    assert not list(pick_types(info2))  # empty
    assert list(pick_types(info2, eeg=True)) == [0, 1]


def test_pick_types_csd():
    """Test pick_types(csd=True)."""
    # info with laplacian/CSD channels at indices 1, 2
    names = ['F1', 'F2', 'C1', 'C2', 'A1', 'A2', 'misc1', 'CSD1']
    info1 = create_info(names, 256, ["eeg", "eeg", "eeg", "eeg", "mag",
                                     "mag", 'misc', 'csd'])
    raw = RawArray(np.zeros((8, 512)), info1)
    raw.set_montage(make_standard_montage('standard_1020'), verbose='error')
    raw_csd = compute_current_source_density(raw, verbose='error')

    assert_array_equal(pick_types(info1, csd=True), [7])

    # pick from the raw object
    assert raw_csd.copy().pick_types(csd=True).ch_names == [
        'F1', 'F2', 'C1', 'C2', 'CSD1']


@pytest.mark.parametrize('meg', [True, False, 'grad', 'mag'])
@pytest.mark.parametrize('eeg', [True, False])
@pytest.mark.parametrize('ordered', [True, False])
def test_get_channel_types_equiv(meg, eeg, ordered):
    """Test equivalence of get_channel_types."""
    raw = read_raw_fif(fif_fname)
    pick_types(raw.info, meg=meg, eeg=eeg)
    picks = pick_types(raw.info, meg=meg, eeg=eeg)
    if not ordered:
        picks = np.random.RandomState(0).permutation(picks)
    if not meg and not eeg:
        with pytest.raises(ValueError, match='No appropriate channels'):
            raw.get_channel_types(picks=picks)
        return
    types = np.array(raw.get_channel_types(picks=picks))
    types_iter = np.array([channel_type(raw.info, idx) for idx in picks])
    assert_array_equal(types, types_iter)
