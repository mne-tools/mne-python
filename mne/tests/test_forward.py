import os
import os.path as op
from subprocess import CalledProcessError
import warnings

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal, \
                          assert_array_equal, assert_allclose

from mne.datasets import sample
from mne.fiff import Raw, Evoked, pick_types_forward
from mne import read_forward_solution, apply_forward, apply_forward_raw, \
                do_forward_solution, average_forward_solutions, \
                write_forward_solution
from mne import SourceEstimate, read_trans
from mne.label import read_label
from mne.utils import requires_mne, run_subprocess, _TempDir
from mne.forward import restrict_forward_to_stc, restrict_forward_to_label

data_path = sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-oct-6-fwd.fif')
fname_meeg = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis-meg-eeg-oct-6-fwd.fif')

fname_raw = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                    'test_raw.fif')

fname_evoked = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                       'test-ave.fif')
fname_mri = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
temp_dir = _TempDir()
# make a file that exists with some data in it
existing_file = op.join(temp_dir, 'test.fif')
with open(existing_file, 'wb') as fid:
    fid.write('aoeu')


def test_io_forward():
    """Test IO for forward solutions
    """
    # test M/EEG
    fwd_meeg = read_forward_solution(fname_meeg)
    leadfield = fwd_meeg['sol']['data']
    assert_equal(leadfield.shape, (366, 22494))
    assert_equal(len(fwd_meeg['sol']['row_names']), 366)
    fname_temp = op.join(temp_dir, 'fwd.fif')
    write_forward_solution(fname_temp, fwd_meeg, overwrite=True)

    fwd_meeg = read_forward_solution(fname_temp)
    assert_allclose(leadfield, fwd_meeg['sol']['data'])
    assert_equal(len(fwd_meeg['sol']['row_names']), 366)

    # now do extensive tests with MEG
    fwd = read_forward_solution(fname)
    fwd = read_forward_solution(fname, surf_ori=True)
    leadfield = fwd['sol']['data']
    assert_equal(leadfield.shape, (306, 22494))
    assert_equal(len(fwd['sol']['row_names']), 306)
    fname_temp = op.join(temp_dir, 'fwd.fif')
    write_forward_solution(fname_temp, fwd, overwrite=True)

    fwd = read_forward_solution(fname, surf_ori=True)
    fwd_read = read_forward_solution(fname_temp, surf_ori=True)
    leadfield = fwd_read['sol']['data']
    assert_equal(leadfield.shape, (306, 22494))
    assert_equal(len(fwd_read['sol']['row_names']), 306)
    assert_equal(len(fwd_read['info']['chs']), 306)
    assert_true('dev_head_t' in fwd_read['info'])
    assert_true('mri_head_t' in fwd_read)
    assert_array_almost_equal(fwd['sol']['data'], fwd_read['sol']['data'])

    fwd = read_forward_solution(fname, force_fixed=True)
    leadfield = fwd['sol']['data']
    assert_equal(leadfield.shape, (306, 22494 / 3))
    assert_equal(len(fwd['sol']['row_names']), 306)
    assert_equal(len(fwd['info']['chs']), 306)
    assert_true('dev_head_t' in fwd['info'])
    assert_true('mri_head_t' in fwd)


def test_apply_forward():
    """Test projection of source space data to sensor space
    """
    start = 0
    stop = 5
    n_times = stop - start - 1
    sfreq = 10.0
    t_start = 0.123

    fwd = read_forward_solution(fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)

    vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc_data = np.ones((len(vertno[0]) + len(vertno[1]), n_times))
    stc = SourceEstimate(stc_data, vertno, tmin=t_start, tstep=1.0 / sfreq)

    gain_sum = np.sum(fwd['sol']['data'], axis=1)

    # Evoked
    with warnings.catch_warnings(record=True) as w:
        evoked = Evoked(fname_evoked, setno=0)
        evoked = apply_forward(fwd, stc, evoked, start=start, stop=stop)
        assert_equal(len(w), 2)
        data = evoked.data
        times = evoked.times

        # do some tests
        assert_array_almost_equal(evoked.info['sfreq'], sfreq)
        assert_array_almost_equal(np.sum(data, axis=1), n_times * gain_sum)
        assert_array_almost_equal(times[0], t_start)
        assert_array_almost_equal(times[-1], t_start + (n_times - 1) / sfreq)

        # Raw
        raw = Raw(fname_raw)
        raw_proj = apply_forward_raw(fwd, stc, raw, start=start, stop=stop)
        data, times = raw_proj[:, :]

        # do some tests
        assert_array_almost_equal(raw_proj.info['sfreq'], sfreq)
        assert_array_almost_equal(np.sum(data, axis=1), n_times * gain_sum)
        assert_array_almost_equal(times[0], t_start)
        assert_array_almost_equal(times[-1], t_start + (n_times - 1) / sfreq)


def test_restrict_forward_to_stc():
    """Test restriction of source space to source SourceEstimate
    """
    start = 0
    stop = 5
    n_times = stop - start - 1
    sfreq = 10.0
    t_start = 0.123

    fwd = read_forward_solution(fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)

    vertno = [fwd['src'][0]['vertno'][0:15], fwd['src'][1]['vertno'][0:5]]
    stc_data = np.ones((len(vertno[0]) + len(vertno[1]), n_times))
    stc = SourceEstimate(stc_data, vertno, tmin=t_start, tstep=1.0 / sfreq)

    fwd_out = restrict_forward_to_stc(fwd, stc)

    assert_equal(fwd_out['sol']['ncol'], 20)
    assert_equal(fwd_out['src'][0]['nuse'], 15)
    assert_equal(fwd_out['src'][1]['nuse'], 5)
    assert_equal(fwd_out['src'][0]['vertno'], fwd['src'][0]['vertno'][0:15])
    assert_equal(fwd_out['src'][1]['vertno'], fwd['src'][1]['vertno'][0:5])

    fwd = read_forward_solution(fname, force_fixed=False)
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


def test_restrict_forward_to_label():
    """Test restriction of source space to label
    """
    fwd = read_forward_solution(fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)

    label_path = op.join(data_path, 'MEG', 'sample', 'labels')
    labels = ['Aud-lh', 'Vis-rh']
    label_lh = read_label(op.join(label_path, labels[0] + '.label'))
    label_rh = read_label(op.join(label_path, labels[1] + '.label'))

    fwd_out = restrict_forward_to_label(fwd, [label_lh, label_rh])

    src_sel_lh = np.intersect1d(fwd['src'][0]['vertno'], label_lh.vertices)
    src_sel_lh = np.searchsorted(fwd['src'][0]['vertno'], src_sel_lh)

    src_sel_rh = np.intersect1d(fwd['src'][1]['vertno'], label_rh.vertices)
    src_sel_rh = np.searchsorted(fwd['src'][1]['vertno'], src_sel_rh)\
                 + len(fwd['src'][0]['vertno'])

    assert_equal(fwd_out['sol']['ncol'], len(src_sel_lh) + len(src_sel_rh))
    assert_equal(fwd_out['src'][0]['nuse'], len(src_sel_lh))
    assert_equal(fwd_out['src'][1]['nuse'], len(src_sel_rh))
    assert_equal(fwd_out['src'][0]['vertno'], src_sel_lh)
    assert_equal(fwd_out['src'][1]['vertno'], src_sel_rh)

    fwd = read_forward_solution(fname, force_fixed=False)
    fwd = pick_types_forward(fwd, meg=True)

    label_path = op.join(data_path, 'MEG', 'sample', 'labels')
    labels = ['Aud-lh', 'Vis-rh']
    label_lh = read_label(op.join(label_path, labels[0] + '.label'))
    label_rh = read_label(op.join(label_path, labels[1] + '.label'))

    fwd_out = restrict_forward_to_label(fwd, [label_lh, label_rh])

    src_sel_lh = np.intersect1d(fwd['src'][0]['vertno'], label_lh.vertices)
    src_sel_lh = np.searchsorted(fwd['src'][0]['vertno'], src_sel_lh)

    src_sel_rh = np.intersect1d(fwd['src'][1]['vertno'], label_rh.vertices)
    src_sel_rh = np.searchsorted(fwd['src'][1]['vertno'], src_sel_rh)\
                 + len(fwd['src'][0]['vertno'])

    assert_equal(fwd_out['sol']['ncol'],
                 3 * (len(src_sel_lh) + len(src_sel_rh)))
    assert_equal(fwd_out['src'][0]['nuse'], len(src_sel_lh))
    assert_equal(fwd_out['src'][1]['nuse'], len(src_sel_rh))
    assert_equal(fwd_out['src'][0]['vertno'], src_sel_lh)
    assert_equal(fwd_out['src'][1]['vertno'], src_sel_rh)


@requires_mne
def test_average_forward_solution():
    """Test averaging forward solutions
    """
    fwd = read_forward_solution(fname)
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
    assert_array_equal(fwd['sol']['data'], fwd_copy['sol']['data'])

    # modify a fwd solution, save it, use MNE to average with old one
    fwd_copy['sol']['data'] *= 0.5
    fname_copy = op.join(temp_dir, 'fwd.fif')
    write_forward_solution(fname_copy, fwd_copy, overwrite=True)
    cmd = ('mne_average_forward_solutions', '--fwd', fname, '--fwd',
           fname_copy, '--out' , fname_copy)
    run_subprocess(cmd)

    # now let's actually do it, with one filename and one fwd
    fwd_ave = average_forward_solutions([fwd, fwd_copy])
    assert_array_equal(0.75 * fwd['sol']['data'], fwd_ave['sol']['data'])
    # fwd_ave_mne = read_forward_solution(fname_copy)
    # assert_array_equal(fwd_ave_mne['sol']['data'], fwd_ave['sol']['data'])


@requires_mne
def test_do_forward_solution():
    """Test making forward solution from python
    """
    subjects_dir = os.path.join(data_path, 'subjects')

    raw = Raw(fname_raw)
    mri = read_trans(fname_mri)
    fname_fake = op.join(temp_dir, 'no_have.fif')

    # ## Error checks
    # bad subject
    assert_raises(ValueError, do_forward_solution, 1, fname_raw,
                  subjects_dir=subjects_dir)
    # bad meas
    assert_raises(ValueError, do_forward_solution, 'sample', 1,
                  subjects_dir=subjects_dir)
    # meas doesn't exist
    assert_raises(IOError, do_forward_solution, 'sample', fname_fake,
                  subjects_dir=subjects_dir)
    # don't specify trans and meas
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  subjects_dir=subjects_dir)
    # specify both trans and meas
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  trans='me', mri='you', subjects_dir=subjects_dir)
    # specify non-existent trans
    assert_raises(IOError, do_forward_solution, 'sample', fname_raw,
                  trans=fname_fake, subjects_dir=subjects_dir)
    # specify non-existent mri
    assert_raises(IOError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_fake, subjects_dir=subjects_dir)
    # specify non-string mri
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=1, subjects_dir=subjects_dir)
    # specify non-string trans
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  trans=1, subjects_dir=subjects_dir)
    # test specifying an actual trans in python space -- this should work but
    # the transform I/O reduces our accuracy -- so we'll just hack a test here
    # by making it bomb with eeg=False and meg=False
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=mri, eeg=False, meg=False, subjects_dir=subjects_dir)
    # mindist as non-integer
    assert_raises(TypeError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_mri, mindist=dict(), subjects_dir=subjects_dir)
    # mindist as string but not 'all'
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_mri, eeg=False, mindist='yall',
                  subjects_dir=subjects_dir)
    # src, spacing, and bem as non-str
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_mri, src=1, subjects_dir=subjects_dir)
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_mri, spacing=1, subjects_dir=subjects_dir)
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_mri, bem=1, subjects_dir=subjects_dir)
    # no overwrite flag
    assert_raises(IOError, do_forward_solution, 'sample', fname_raw,
                  existing_file, mri=fname_mri, subjects_dir=subjects_dir)
    # let's catch an MNE error, this time about trans being wrong
    assert_raises(CalledProcessError, do_forward_solution, 'sample', fname_raw,
                  existing_file, trans=fname_mri, overwrite=True,
                  spacing='oct-6', subjects_dir=subjects_dir)

    # ## Actually calculate one and check
    # make a meas from raw (tests all steps in creating evoked),
    # don't do EEG or 5120-5120-5120 BEM because they're ~3x slower
    fwd_py = do_forward_solution('sample', raw, mindist=5, spacing='oct-6',
                                 bem='sample-5120', mri=fname_mri, eeg=False,
                                 subjects_dir=subjects_dir)
    fwd = read_forward_solution(fname)
    assert_allclose(fwd['sol']['data'], fwd_py['sol']['data'],
                    rtol=1e-5, atol=1e-8)
    assert_equal(fwd_py['sol']['data'].shape, (306, 22494))
    assert_equal(len(fwd['sol']['row_names']), 306)
