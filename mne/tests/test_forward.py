import os.path as op
import warnings

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal, \
                          assert_array_equal
import shutil

from mne.datasets import sample
from mne.fiff import Raw, Evoked, pick_types_forward
from mne import read_forward_solution, apply_forward, apply_forward_raw, \
                do_forward_solution, average_forward_solutions
from mne import SourceEstimate, read_trans
from mne.label import read_label
from mne.utils import requires_mne, _TempDir
from mne.forward import restrict_forward_to_stc, restrict_forward_to_label

data_path = sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-oct-6-fwd.fif')
fname_mri = op.join(data_path, 'MEG', 'sample',
                    'all-trans.fif')

fname_raw = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                    'test_raw.fif')

fname_evoked = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                       'test-ave.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
temp_dir = _TempDir()
# make a file that exists with some data in it
existing_file = op.join(temp_dir, 'test.fif')
with open(existing_file, 'wb') as fid:
    fid.write('aoeu')


def test_io_forward():
    """Test IO for forward solutions
    """
    fwd = read_forward_solution(fname)
    fwd = read_forward_solution(fname, surf_ori=True)
    leadfield = fwd['sol']['data']
    assert_equal(leadfield.shape, (306, 22494))
    assert_equal(len(fwd['sol']['row_names']), 306)

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
    """Test restriction of source space to source SourceEstimate
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
    fname_2 = op.join(temp_dir, 'copy-fwd.fif')
    shutil.copy(fname, fname_2)
    fnames = [fname, fname_2]
    # file exists without overwrite
    assert_raises(IOError, average_forward_solutions, existing_file, fnames)
    # not a string
    assert_raises(TypeError, average_forward_solutions, 1, fnames)
    # not a list
    assert_raises(TypeError, average_forward_solutions, 'whatever', 1)
    # not a list of str
    assert_raises(TypeError, average_forward_solutions, 'whatever', ['m', 1])
    # not correct arguments passed
    assert_raises(RuntimeError, average_forward_solutions, '', fnames)

    # now let's actually do it
    average_forward_solutions(existing_file, fnames, overwrite=True)
    # XXX These lines should be commented out once forward reading bug
    # is fixed
    #fwd_py = read_forward_solution(existing_file)
    #fwd = read_forward_solution(fname)
    #assert_array_equal(fwd['sol']['data'], fwd_py['sol']['data'])


@requires_mne
def test_do_forward_solution():
    """Test making forward solution from python
    """
    fwd = read_forward_solution(fname)
    raw = Raw(fname_raw)
    mri = read_trans(fname_trans)
    fname_out = op.join(temp_dir, 'meg-fwd.fif')
    fname_fake = op.join(temp_dir, 'no_have.fif')

    ### Error checks
    # bad subject
    assert_raises(ValueError, do_forward_solution, fname_out, 1, fname_raw)
    # bad meas
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample', 1)
    # meas doesn't exist
    assert_raises(IOError, do_forward_solution, fname_out, 'sample',
                  fname_fake)
    # don't specify trans and meas
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample',
                  fname_raw)
    # specify both trans and meas
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample',
                  fname_raw, trans='me', mri='you')
    # specify non-existent trans
    assert_raises(IOError, do_forward_solution, fname_out, 'sample',
                  fname_raw, trans=fname_fake)
    # specify non-existent mri
    assert_raises(IOError, do_forward_solution, fname_out, 'sample',
                  fname_raw, mri=fname_fake)
    # specify non-string mri
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample',
                  fname_raw, mri=1)
    # specify non-string trans
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample',
                  fname_raw, trans=1)
    # test specifying an actual trans in python space -- this should work but
    # the transform I/O reduces our accuracy -- so we'll just hack a test here
    # by making it bomb with eeg=False and meg=False
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample',
                  fname_raw, mri=mri, eeg=False, meg=False)
    # mindist as non-integer
    assert_raises(TypeError, do_forward_solution, fname_out, 'sample',
                  fname_raw, mri=fname_trans, mindist=dict())
    # mindist as string but not 'all'
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample',
                  fname_raw, mri=fname_trans, eeg=False, mindist='yall')
    # src, spacing, and bem as non-str
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample',
                  fname_raw, mri=fname_trans, src=1)
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample',
                  fname_raw, mri=fname_trans, spacing=1)
    assert_raises(ValueError, do_forward_solution, fname_out, 'sample',
                  fname_raw, mri=fname_trans, bem=1)
    # no overwrite flag
    assert_raises(IOError, do_forward_solution, existing_file, 'sample',
                  fname_raw, mri=fname_trans)
    # let's catch an MNE error, this time about trans being wrong
    assert_raises(RuntimeError, do_forward_solution, fname_out, 'sample',
                  fname_raw, trans=fname_trans)

    ### Actually calculate one and check
    # make a meas from raw (tests all steps in creating evoked),
    # don't do EEG or 5120-5120-5120 BEM because they're ~3x slower
    do_forward_solution(existing_file, 'sample', raw, mindist=5,
                        spacing='oct-6', bem='sample-5120',
                        mri=fname_trans, eeg=False, overwrite=True)
    fwd_py = read_forward_solution(existing_file)
    assert_array_equal(fwd['sol']['data'], fwd_py['sol']['data'])
    assert_equal(fwd_py['sol']['data'].shape, (306, 22494))
    assert_equal(len(fwd['sol']['row_names']), 306)

