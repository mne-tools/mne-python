# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os
from os import path as op
import shutil

import numpy as np
from nose.tools import assert_raises, assert_true
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from mne import pick_types
from mne.tests.common import assert_dig_allclose
from mne.transforms import apply_trans
from mne.io import Raw, read_raw_ctf
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import _TempDir, run_tests_if_main, slow_test
from mne.datasets import testing

ctf_dir = op.join(testing.data_path(download=False), 'CTF')
ctf_fname_continuous = 'testdata_ctf.ds'
ctf_fname_1_trial = 'testdata_ctf_short.ds'
ctf_fname_2_trials = 'testdata_ctf_pseudocontinuous.ds'
ctf_fname_discont = 'testdata_ctf_short_discontinuous.ds'
ctf_fname_somato = 'somMDYO-18av.ds'
ctf_fname_catch = 'catch-alp-good-f.ds'

block_sizes = {
    ctf_fname_continuous: 12000,
    ctf_fname_1_trial: 4801,
    ctf_fname_2_trials: 12000,
    ctf_fname_discont: 1201,
    ctf_fname_somato: 313,
    ctf_fname_catch: 2500,
}
single_trials = (
    ctf_fname_continuous,
    ctf_fname_1_trial,
)

ctf_fnames = tuple(sorted(block_sizes.keys()))


@slow_test
@testing.requires_testing_data
def test_read_ctf():
    """Test CTF reader"""
    temp_dir = _TempDir()
    out_fname = op.join(temp_dir, 'test_py_raw.fif')

    # Create a dummy .eeg file so we can test our reading/application of it
    os.mkdir(op.join(temp_dir, 'randpos'))
    ctf_eeg_fname = op.join(temp_dir, 'randpos', ctf_fname_catch)
    shutil.copytree(op.join(ctf_dir, ctf_fname_catch), ctf_eeg_fname)
    raw = _test_raw_reader(read_raw_ctf, directory=ctf_eeg_fname)
    picks = pick_types(raw.info, meg=False, eeg=True)
    pos = np.random.RandomState(42).randn(len(picks), 3)
    fake_eeg_fname = op.join(ctf_eeg_fname, 'catch-alp-good-f.eeg')
    # Create a bad file
    with open(fake_eeg_fname, 'wb') as fid:
        fid.write('foo\n'.encode('ascii'))
    assert_raises(RuntimeError, read_raw_ctf, ctf_eeg_fname)
    # Create a good file
    with open(fake_eeg_fname, 'wb') as fid:
        for ii, ch_num in enumerate(picks):
            args = (str(ch_num + 1), raw.ch_names[ch_num],) + tuple(
                '%0.5f' % x for x in 100 * pos[ii])  # convert to cm
            fid.write(('\t'.join(args) + '\n').encode('ascii'))
    pos_read_old = np.array([raw.info['chs'][p]['loc'][:3] for p in picks])
    raw = read_raw_ctf(ctf_eeg_fname)  # read modified data
    pos_read = np.array([raw.info['chs'][p]['loc'][:3] for p in picks])
    assert_allclose(apply_trans(raw.info['ctf_head_t'], pos), pos_read,
                    rtol=1e-5, atol=1e-5)
    assert_true((pos_read == pos_read_old).mean() < 0.1)
    shutil.copy(op.join(ctf_dir, 'catch-alp-good-f.ds_randpos_raw.fif'),
                op.join(temp_dir, 'randpos', 'catch-alp-good-f.ds_raw.fif'))

    # Create a version with no hc, starting out *with* EEG pos (error)
    os.mkdir(op.join(temp_dir, 'nohc'))
    ctf_no_hc_fname = op.join(temp_dir, 'no_hc', ctf_fname_catch)
    shutil.copytree(ctf_eeg_fname, ctf_no_hc_fname)
    remove_base = op.join(ctf_no_hc_fname, op.basename(ctf_fname_catch[:-3]))
    os.remove(remove_base + '.hc')
    assert_raises(RuntimeError, read_raw_ctf, ctf_no_hc_fname)  # no coord tr
    os.remove(remove_base + '.eeg')
    shutil.copy(op.join(ctf_dir, 'catch-alp-good-f.ds_nohc_raw.fif'),
                op.join(temp_dir, 'no_hc', 'catch-alp-good-f.ds_raw.fif'))

    # All our files
    use_fnames = [op.join(ctf_dir, c) for c in ctf_fnames]
    for fname in use_fnames:
        raw_c = Raw(fname + '_raw.fif', add_eeg_ref=False, preload=True)
        raw = read_raw_ctf(fname)

        # check info match
        assert_array_equal(raw.ch_names, raw_c.ch_names)
        assert_allclose(raw.times, raw_c.times)
        assert_allclose(raw._cals, raw_c._cals)
        for key in ('version', 'usecs'):
            assert_equal(raw.info['meas_id'][key], raw_c.info['meas_id'][key])
        py_time = raw.info['meas_id']['secs']
        c_time = raw_c.info['meas_id']['secs']
        max_offset = 24 * 60 * 60  # probably overkill but covers timezone
        assert_true(c_time - max_offset <= py_time <= c_time)
        for t in ('dev_head_t', 'dev_ctf_t', 'ctf_head_t'):
            assert_allclose(raw.info[t]['trans'], raw_c.info[t]['trans'],
                            rtol=1e-4, atol=1e-7)
        for key in ('acq_pars', 'acq_stim', 'bads',
                    'ch_names', 'custom_ref_applied', 'description',
                    'events', 'experimenter', 'highpass', 'line_freq',
                    'lowpass', 'nchan', 'proj_id', 'proj_name',
                    'projs', 'sfreq', 'subject_info'):
            assert_equal(raw.info[key], raw_c.info[key], key)
        if op.basename(fname) not in single_trials:
            # We don't force buffer size to be smaller like MNE-C
            assert_equal(raw.info['buffer_size_sec'],
                         raw_c.info['buffer_size_sec'])
        assert_equal(len(raw.info['comps']), len(raw_c.info['comps']))
        for c1, c2 in zip(raw.info['comps'], raw_c.info['comps']):
            for key in ('colcals', 'rowcals'):
                assert_allclose(c1[key], c2[key])
            assert_equal(c1['save_calibrated'], c2['save_calibrated'])
            for key in ('row_names', 'col_names', 'nrow', 'ncol'):
                assert_array_equal(c1['data'][key], c2['data'][key])
            assert_allclose(c1['data']['data'], c2['data']['data'], atol=1e-7,
                            rtol=1e-5)
        assert_allclose(raw.info['hpi_results'][0]['coord_trans']['trans'],
                        raw_c.info['hpi_results'][0]['coord_trans']['trans'],
                        rtol=1e-5, atol=1e-7)
        assert_equal(len(raw.info['chs']), len(raw_c.info['chs']))
        for ii, (c1, c2) in enumerate(zip(raw.info['chs'], raw_c.info['chs'])):
            for key in ('kind', 'scanno', 'unit', 'ch_name', 'unit_mul',
                        'range', 'coord_frame', 'coil_type', 'logno'):
                assert_equal(c1[key], c2[key])
            for key in ('loc', 'cal'):
                assert_allclose(c1[key], c2[key], atol=1e-6, rtol=1e-4,
                                err_msg='raw.info["chs"][%d][%s]' % (ii, key))
        assert_dig_allclose(raw.info, raw_c.info)

        # check data match
        raw_c.save(out_fname, overwrite=True, buffer_size_sec=1.)
        raw_read = Raw(out_fname, add_eeg_ref=False)

        # so let's check tricky cases based on sample boundaries
        rng = np.random.RandomState(0)
        pick_ch = rng.permutation(np.arange(len(raw.ch_names)))[:10]
        bnd = int(round(raw.info['sfreq'] * raw.info['buffer_size_sec']))
        assert_equal(bnd, raw._raw_extras[0]['block_size'])
        assert_equal(bnd, block_sizes[op.basename(fname)])
        slices = (slice(0, bnd), slice(bnd - 1, bnd), slice(3, bnd),
                  slice(3, 300), slice(None))
        if len(raw.times) >= 2 * bnd:  # at least two complete blocks
            slices = slices + (slice(bnd, 2 * bnd), slice(bnd, bnd + 1),
                               slice(0, bnd + 100))
        for sl_time in slices:
            assert_allclose(raw[pick_ch, sl_time][0],
                            raw_c[pick_ch, sl_time][0])
            assert_allclose(raw_read[pick_ch, sl_time][0],
                            raw_c[pick_ch, sl_time][0])
        # all data / preload
        raw = read_raw_ctf(fname, preload=True)
        assert_allclose(raw[:][0], raw_c[:][0])
    assert_raises(TypeError, read_raw_ctf, 1)
    assert_raises(ValueError, read_raw_ctf, ctf_fname_continuous + 'foo.ds')
    # test ignoring of system clock
    read_raw_ctf(op.join(ctf_dir, ctf_fname_continuous), 'ignore')
    assert_raises(ValueError, read_raw_ctf,
                  op.join(ctf_dir, ctf_fname_continuous), 'foo')

run_tests_if_main()
