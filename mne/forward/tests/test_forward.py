import os
import os.path as op
import warnings
import gc

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_array_equal, assert_allclose)

from mne.datasets import testing
from mne import (read_forward_solution, apply_forward, apply_forward_raw,
                 average_forward_solutions, write_forward_solution,
                 convert_forward_solution, SourceEstimate, pick_types_forward,
                 read_evokeds)
from mne.tests.common import assert_naming
from mne.label import read_label
from mne.utils import (requires_mne, run_subprocess, _TempDir,
                       run_tests_if_main, slow_test)
from mne.forward import (restrict_forward_to_stc, restrict_forward_to_label,
                         Forward)

data_path = testing.data_path(download=False)
fname_meeg = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_meeg_grad = op.join(data_path, 'MEG', 'sample',
                          'sample_audvis_trunc-meg-eeg-oct-2-grad-fwd.fif')

fname_raw = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                    'test_raw.fif')

fname_evoked = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                       'data', 'test-ave.fif')
fname_mri = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-trans.fif')
subjects_dir = os.path.join(data_path, 'subjects')
fname_src = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-4-src.fif')


def compare_forwards(f1, f2):
    """Helper to compare two potentially converted forward solutions"""
    assert_allclose(f1['sol']['data'], f2['sol']['data'])
    assert_equal(f1['sol']['ncol'], f2['sol']['ncol'])
    assert_allclose(f1['source_nn'], f2['source_nn'])
    if f1['sol_grad'] is not None:
        assert_true(f2['sol_grad'] is not None)
        assert_allclose(f1['sol_grad']['data'], f2['sol_grad']['data'])
        assert_equal(f1['sol_grad']['ncol'], f2['sol_grad']['ncol'])
    else:
        assert_true(f2['sol_grad'] is None)
    assert_equal(f1['source_ori'], f2['source_ori'])
    assert_equal(f1['surf_ori'], f2['surf_ori'])


@testing.requires_testing_data
def test_convert_forward():
    """Test converting forward solution between different representations
    """
    fwd = read_forward_solution(fname_meeg_grad)
    fwd_repr = repr(fwd)
    assert_true('306' in fwd_repr)
    assert_true('60' in fwd_repr)
    assert_true(fwd_repr)
    assert_true(isinstance(fwd, Forward))
    # look at surface orientation
    fwd_surf = convert_forward_solution(fwd, surf_ori=True)
    fwd_surf_io = read_forward_solution(fname_meeg_grad, surf_ori=True)
    compare_forwards(fwd_surf, fwd_surf_io)
    del fwd_surf_io
    gc.collect()
    # go back
    fwd_new = convert_forward_solution(fwd_surf, surf_ori=False)
    assert_true(repr(fwd_new))
    assert_true(isinstance(fwd_new, Forward))
    compare_forwards(fwd, fwd_new)
    # now go to fixed
    fwd_fixed = convert_forward_solution(fwd_surf, surf_ori=False,
                                         force_fixed=True)
    del fwd_surf
    gc.collect()
    assert_true(repr(fwd_fixed))
    assert_true(isinstance(fwd_fixed, Forward))
    fwd_fixed_io = read_forward_solution(fname_meeg_grad, surf_ori=False,
                                         force_fixed=True)
    compare_forwards(fwd_fixed, fwd_fixed_io)
    del fwd_fixed_io
    gc.collect()
    # now go back to cartesian (original condition)
    fwd_new = convert_forward_solution(fwd_fixed)
    assert_true(repr(fwd_new))
    assert_true(isinstance(fwd_new, Forward))
    compare_forwards(fwd, fwd_new)
    del fwd, fwd_new, fwd_fixed
    gc.collect()


@slow_test
@testing.requires_testing_data
def test_io_forward():
    """Test IO for forward solutions
    """
    temp_dir = _TempDir()
    # do extensive tests with MEEG + grad
    n_channels, n_src = 366, 108
    fwd = read_forward_solution(fname_meeg_grad)
    assert_true(isinstance(fwd, Forward))
    fwd = read_forward_solution(fname_meeg_grad, surf_ori=True)
    leadfield = fwd['sol']['data']
    assert_equal(leadfield.shape, (n_channels, n_src))
    assert_equal(len(fwd['sol']['row_names']), n_channels)
    fname_temp = op.join(temp_dir, 'test-fwd.fif')
    write_forward_solution(fname_temp, fwd, overwrite=True)

    fwd = read_forward_solution(fname_meeg_grad, surf_ori=True)
    fwd_read = read_forward_solution(fname_temp, surf_ori=True)
    leadfield = fwd_read['sol']['data']
    assert_equal(leadfield.shape, (n_channels, n_src))
    assert_equal(len(fwd_read['sol']['row_names']), n_channels)
    assert_equal(len(fwd_read['info']['chs']), n_channels)
    assert_true('dev_head_t' in fwd_read['info'])
    assert_true('mri_head_t' in fwd_read)
    assert_array_almost_equal(fwd['sol']['data'], fwd_read['sol']['data'])

    fwd = read_forward_solution(fname_meeg_grad, force_fixed=True)
    leadfield = fwd['sol']['data']
    assert_equal(leadfield.shape, (n_channels, n_src / 3))
    assert_equal(len(fwd['sol']['row_names']), n_channels)
    assert_equal(len(fwd['info']['chs']), n_channels)
    assert_true('dev_head_t' in fwd['info'])
    assert_true('mri_head_t' in fwd)
    assert_true(fwd['surf_ori'])

    # test warnings on bad filenames
    fwd = read_forward_solution(fname_meeg_grad)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fwd_badname = op.join(temp_dir, 'test-bad-name.fif.gz')
        write_forward_solution(fwd_badname, fwd)
        read_forward_solution(fwd_badname)
    assert_naming(w, 'test_forward.py', 2)

    fwd = read_forward_solution(fname_meeg)
    write_forward_solution(fname_temp, fwd, overwrite=True)
    fwd_read = read_forward_solution(fname_temp)
    compare_forwards(fwd, fwd_read)


@testing.requires_testing_data
def test_apply_forward():
    """Test projection of source space data to sensor space
    """
    start = 0
    stop = 5
    n_times = stop - start - 1
    sfreq = 10.0
    t_start = 0.123

    fwd = read_forward_solution(fname_meeg, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)
    assert_true(isinstance(fwd, Forward))

    vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc_data = np.ones((len(vertno[0]) + len(vertno[1]), n_times))
    stc = SourceEstimate(stc_data, vertno, tmin=t_start, tstep=1.0 / sfreq)

    gain_sum = np.sum(fwd['sol']['data'], axis=1)

    # Evoked
    with warnings.catch_warnings(record=True) as w:
        evoked = read_evokeds(fname_evoked, condition=0)
        evoked.pick_types(meg=True)
        evoked = apply_forward(fwd, stc, evoked.info, start=start, stop=stop)
        assert_equal(len(w), 2)
        data = evoked.data
        times = evoked.times

        # do some tests
        assert_array_almost_equal(evoked.info['sfreq'], sfreq)
        assert_array_almost_equal(np.sum(data, axis=1), n_times * gain_sum)
        assert_array_almost_equal(times[0], t_start)
        assert_array_almost_equal(times[-1], t_start + (n_times - 1) / sfreq)

        # Raw
        raw_proj = apply_forward_raw(fwd, stc, evoked.info, start=start,
                                     stop=stop)
        data, times = raw_proj[:, :]

        # do some tests
        assert_array_almost_equal(raw_proj.info['sfreq'], sfreq)
        assert_array_almost_equal(np.sum(data, axis=1), n_times * gain_sum)
        atol = 1. / sfreq
        assert_allclose(raw_proj.first_samp / sfreq, t_start, atol=atol)
        assert_allclose(raw_proj.last_samp / sfreq,
                        t_start + (n_times - 1) / sfreq, atol=atol)


@testing.requires_testing_data
def test_restrict_forward_to_stc():
    """Test restriction of source space to source SourceEstimate
    """
    start = 0
    stop = 5
    n_times = stop - start - 1
    sfreq = 10.0
    t_start = 0.123

    fwd = read_forward_solution(fname_meeg, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)

    vertno = [fwd['src'][0]['vertno'][0:15], fwd['src'][1]['vertno'][0:5]]
    stc_data = np.ones((len(vertno[0]) + len(vertno[1]), n_times))
    stc = SourceEstimate(stc_data, vertno, tmin=t_start, tstep=1.0 / sfreq)

    fwd_out = restrict_forward_to_stc(fwd, stc)
    assert_true(isinstance(fwd_out, Forward))

    assert_equal(fwd_out['sol']['ncol'], 20)
    assert_equal(fwd_out['src'][0]['nuse'], 15)
    assert_equal(fwd_out['src'][1]['nuse'], 5)
    assert_equal(fwd_out['src'][0]['vertno'], fwd['src'][0]['vertno'][0:15])
    assert_equal(fwd_out['src'][1]['vertno'], fwd['src'][1]['vertno'][0:5])

    fwd = read_forward_solution(fname_meeg, force_fixed=False)
    fwd = pick_types_forward(fwd, meg=True)

    vertno = [fwd['src'][0]['vertno'][0:15], fwd['src'][1]['vertno'][0:5]]
    stc_data = np.ones((len(vertno[0]) + len(vertno[1]), n_times))
    stc = SourceEstimate(stc_data, vertno, tmin=t_start, tstep=1.0 / sfreq)

    fwd_out = restrict_forward_to_stc(fwd, stc)

    assert_equal(fwd_out['sol']['ncol'], 60)
    assert_equal(fwd_out['src'][0]['nuse'], 15)
    assert_equal(fwd_out['src'][1]['nuse'], 5)
    assert_equal(fwd_out['src'][0]['vertno'], fwd['src'][0]['vertno'][0:15])
    assert_equal(fwd_out['src'][1]['vertno'], fwd['src'][1]['vertno'][0:5])


@testing.requires_testing_data
def test_restrict_forward_to_label():
    """Test restriction of source space to label
    """
    fwd = read_forward_solution(fname_meeg, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)

    label_path = op.join(data_path, 'MEG', 'sample', 'labels')
    labels = ['Aud-lh', 'Vis-rh']
    label_lh = read_label(op.join(label_path, labels[0] + '.label'))
    label_rh = read_label(op.join(label_path, labels[1] + '.label'))

    fwd_out = restrict_forward_to_label(fwd, [label_lh, label_rh])

    src_sel_lh = np.intersect1d(fwd['src'][0]['vertno'], label_lh.vertices)
    src_sel_lh = np.searchsorted(fwd['src'][0]['vertno'], src_sel_lh)
    vertno_lh = fwd['src'][0]['vertno'][src_sel_lh]

    nuse_lh = fwd['src'][0]['nuse']
    src_sel_rh = np.intersect1d(fwd['src'][1]['vertno'], label_rh.vertices)
    src_sel_rh = np.searchsorted(fwd['src'][1]['vertno'], src_sel_rh)
    vertno_rh = fwd['src'][1]['vertno'][src_sel_rh]
    src_sel_rh += nuse_lh

    assert_equal(fwd_out['sol']['ncol'], len(src_sel_lh) + len(src_sel_rh))
    assert_equal(fwd_out['src'][0]['nuse'], len(src_sel_lh))
    assert_equal(fwd_out['src'][1]['nuse'], len(src_sel_rh))
    assert_equal(fwd_out['src'][0]['vertno'], vertno_lh)
    assert_equal(fwd_out['src'][1]['vertno'], vertno_rh)

    fwd = read_forward_solution(fname_meeg, force_fixed=False)
    fwd = pick_types_forward(fwd, meg=True)

    label_path = op.join(data_path, 'MEG', 'sample', 'labels')
    labels = ['Aud-lh', 'Vis-rh']
    label_lh = read_label(op.join(label_path, labels[0] + '.label'))
    label_rh = read_label(op.join(label_path, labels[1] + '.label'))

    fwd_out = restrict_forward_to_label(fwd, [label_lh, label_rh])

    src_sel_lh = np.intersect1d(fwd['src'][0]['vertno'], label_lh.vertices)
    src_sel_lh = np.searchsorted(fwd['src'][0]['vertno'], src_sel_lh)
    vertno_lh = fwd['src'][0]['vertno'][src_sel_lh]

    nuse_lh = fwd['src'][0]['nuse']
    src_sel_rh = np.intersect1d(fwd['src'][1]['vertno'], label_rh.vertices)
    src_sel_rh = np.searchsorted(fwd['src'][1]['vertno'], src_sel_rh)
    vertno_rh = fwd['src'][1]['vertno'][src_sel_rh]
    src_sel_rh += nuse_lh

    assert_equal(fwd_out['sol']['ncol'],
                 3 * (len(src_sel_lh) + len(src_sel_rh)))
    assert_equal(fwd_out['src'][0]['nuse'], len(src_sel_lh))
    assert_equal(fwd_out['src'][1]['nuse'], len(src_sel_rh))
    assert_equal(fwd_out['src'][0]['vertno'], vertno_lh)
    assert_equal(fwd_out['src'][1]['vertno'], vertno_rh)


@testing.requires_testing_data
@requires_mne
def test_average_forward_solution():
    """Test averaging forward solutions
    """
    temp_dir = _TempDir()
    fwd = read_forward_solution(fname_meeg)
    # input not a list
    assert_raises(TypeError, average_forward_solutions, 1)
    # list is too short
    assert_raises(ValueError, average_forward_solutions, [])
    # negative weights
    assert_raises(ValueError, average_forward_solutions, [fwd, fwd], [-1, 0])
    # all zero weights
    assert_raises(ValueError, average_forward_solutions, [fwd, fwd], [0, 0])
    # weights not same length
    assert_raises(ValueError, average_forward_solutions, [fwd, fwd], [0, 0, 0])
    # list does not only have all dict()
    assert_raises(TypeError, average_forward_solutions, [1, fwd])

    # try an easy case
    fwd_copy = average_forward_solutions([fwd])
    assert_true(isinstance(fwd_copy, Forward))
    assert_array_equal(fwd['sol']['data'], fwd_copy['sol']['data'])

    # modify a fwd solution, save it, use MNE to average with old one
    fwd_copy['sol']['data'] *= 0.5
    fname_copy = op.join(temp_dir, 'copy-fwd.fif')
    write_forward_solution(fname_copy, fwd_copy, overwrite=True)
    cmd = ('mne_average_forward_solutions', '--fwd', fname_meeg, '--fwd',
           fname_copy, '--out', fname_copy)
    run_subprocess(cmd)

    # now let's actually do it, with one filename and one fwd
    fwd_ave = average_forward_solutions([fwd, fwd_copy])
    assert_array_equal(0.75 * fwd['sol']['data'], fwd_ave['sol']['data'])
    # fwd_ave_mne = read_forward_solution(fname_copy)
    # assert_array_equal(fwd_ave_mne['sol']['data'], fwd_ave['sol']['data'])

    # with gradient
    fwd = read_forward_solution(fname_meeg_grad)
    fwd_ave = average_forward_solutions([fwd, fwd])
    compare_forwards(fwd, fwd_ave)

run_tests_if_main()
