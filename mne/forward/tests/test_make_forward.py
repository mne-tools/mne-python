import os
import os.path as op
from subprocess import CalledProcessError

from nose.tools import assert_raises
from numpy.testing import (assert_equal, assert_allclose)

from mne.datasets import sample
from mne.fiff import Raw
from mne import (read_forward_solution, make_forward_solution,
                 do_forward_solution, setup_source_space, read_trans,
                 convert_forward_solution)
from mne.utils import requires_mne, _TempDir
from mne.tests.test_source_space import _compare_source_spaces

data_path = sample.data_path(download=False)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-oct-6-fwd.fif')
fname_meeg = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis-meg-eeg-oct-6-fwd.fif')

fname_raw = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data',
                    'test_raw.fif')

fname_evoked = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                       'data', 'test-ave.fif')
fname_mri = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
subjects_dir = os.path.join(data_path, 'subjects')
fname_src = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')
temp_dir = _TempDir()
# make a file that exists with some data in it
existing_file = op.join(temp_dir, 'test.fif')
with open(existing_file, 'wb') as fid:
    fid.write('aoeu')


def _compare_forwards(fwd, fwd_py, n_sensors, n_src):
    """Helper to test forwards"""
    # check source spaces
    assert_equal(len(fwd['src']), len(fwd_py['src']))
    _compare_source_spaces(fwd['src'], fwd_py['src'], mode='approx')
    for surf_ori in [False, True]:
        if surf_ori:
            fwd = convert_forward_solution(fwd, surf_ori, copy=False)
            fwd_py = convert_forward_solution(fwd, surf_ori, copy=False)

        for key in ['nchan', 'source_nn', 'source_rr', 'source_ori',
                    'surf_ori', 'coord_frame', 'nsource']:
            print key
            assert_allclose(fwd_py[key], fwd[key], rtol=1e-4, atol=1e-7)
        assert_allclose(fwd_py['mri_head_t']['trans'],
                        fwd['mri_head_t']['trans'], rtol=1e-5, atol=1e-8)

        # check MEG
        assert_allclose(fwd['sol']['data'][:306],
                        fwd_py['sol']['data'][:306],
                        rtol=1e-4, atol=1e-9)
        # check EEG
        if fwd['sol']['data'].shape[0] > 306:
            assert_allclose(fwd['sol']['data'][306:],
                            fwd_py['sol']['data'][306:],
                            rtol=1e-3, atol=1e-3)
        assert_equal(fwd_py['sol']['data'].shape, (n_sensors, n_src))
        assert_equal(len(fwd['sol']['row_names']), n_sensors)
        assert_equal(len(fwd_py['sol']['row_names']), n_sensors)


@sample.requires_sample_data
@requires_mne
def test_make_forward_solution_compensation():
    """Test making forward solution from python with compensation
    """
    fname_ctf_raw = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                            'data', 'test_ctf_comp_raw.fif')
    fname_bem = op.join(subjects_dir, 'sample', 'bem',
                        'sample-5120-bem-sol.fif')
    fname_src = op.join(temp_dir, 'oct2-src.fif')
    src = setup_source_space('sample', fname_src, 'oct2',
                             subjects_dir=subjects_dir)
    fwd_py = make_forward_solution(fname_ctf_raw, mindist=0.0,
                                   src=src, eeg=False, meg=True,
                                   bem=fname_bem, mri=fname_mri)

    fwd = do_forward_solution('sample', fname_ctf_raw, src=fname_src,
                              mindist=0.0, bem=fname_bem, mri=fname_mri,
                              eeg=False, meg=True, subjects_dir=subjects_dir)
    _compare_forwards(fwd, fwd_py, 274, 108)


@sample.requires_sample_data
def test_make_forward_solution():
    """Test making M-EEG forward solution from python
    """
    fname_bem = op.join(subjects_dir, 'sample', 'bem',
                        'sample-5120-5120-5120-bem-sol.fif')
    fwd_py = make_forward_solution(fname_raw, mindist=5.0,
                                   src=fname_src, eeg=True, meg=True,
                                   bem=fname_bem, mri=fname_mri)
    fwd = read_forward_solution(fname_meeg)
    _compare_forwards(fwd, fwd_py, 366, 22494)


@sample.requires_sample_data
@requires_mne
def test_do_forward_solution():
    """Test wrapping forward solution from python
    """
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
    assert_raises(CalledProcessError, do_forward_solution, 'sample',
                  fname_raw, existing_file, trans=fname_mri, overwrite=True,
                  spacing='oct6', subjects_dir=subjects_dir)

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
