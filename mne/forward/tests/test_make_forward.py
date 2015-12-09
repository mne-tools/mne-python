from __future__ import print_function

import os
import os.path as op
from subprocess import CalledProcessError
import warnings

from nose.tools import assert_raises, assert_true
import numpy as np
from numpy.testing import (assert_equal, assert_allclose)

from mne.datasets import testing
from mne.io import Raw, read_raw_kit, read_raw_bti, read_info
from mne.io.constants import FIFF
from mne import (read_forward_solution, make_forward_solution,
                 do_forward_solution, read_trans,
                 convert_forward_solution, setup_volume_source_space,
                 read_source_spaces, make_sphere_model,
                 pick_types_forward, pick_info, pick_types, Transform)
from mne.utils import (requires_mne, requires_nibabel, _TempDir,
                       run_tests_if_main, slow_test, run_subprocess)
from mne.forward._make_forward import _create_meg_coils
from mne.forward._compute_forward import _magnetic_dipole_field_vec
from mne.forward import Forward
from mne.source_space import (get_volume_labels_from_aseg,
                              _compare_source_spaces, setup_source_space)

data_path = testing.data_path(download=False)
fname_meeg = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_raw = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                    'test_raw.fif')
fname_evoked = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                       'data', 'test-ave.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
subjects_dir = os.path.join(data_path, 'subjects')
fname_src = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-4-src.fif')
fname_bem = op.join(subjects_dir, 'sample', 'bem',
                    'sample-1280-1280-1280-bem-sol.fif')
fname_aseg = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
fname_bem_meg = op.join(subjects_dir, 'sample', 'bem',
                        'sample-1280-bem-sol.fif')


def _compare_forwards(fwd, fwd_py, n_sensors, n_src,
                      meg_rtol=1e-4, meg_atol=1e-9,
                      eeg_rtol=1e-3, eeg_atol=1e-3):
    """Helper to test forwards"""
    # check source spaces
    assert_equal(len(fwd['src']), len(fwd_py['src']))
    _compare_source_spaces(fwd['src'], fwd_py['src'], mode='approx')
    for surf_ori in [False, True]:
        if surf_ori:
            # use copy here to leave our originals unmodified
            fwd = convert_forward_solution(fwd, surf_ori, copy=True)
            fwd_py = convert_forward_solution(fwd, surf_ori, copy=True)

        for key in ['nchan', 'source_nn', 'source_rr', 'source_ori',
                    'surf_ori', 'coord_frame', 'nsource']:
            print(key)
            assert_allclose(fwd_py[key], fwd[key], rtol=1e-4, atol=1e-7)
        assert_allclose(fwd_py['mri_head_t']['trans'],
                        fwd['mri_head_t']['trans'], rtol=1e-5, atol=1e-8)

        assert_equal(fwd_py['sol']['data'].shape, (n_sensors, n_src))
        assert_equal(len(fwd['sol']['row_names']), n_sensors)
        assert_equal(len(fwd_py['sol']['row_names']), n_sensors)

        # check MEG
        assert_allclose(fwd['sol']['data'][:306],
                        fwd_py['sol']['data'][:306],
                        rtol=meg_rtol, atol=meg_atol,
                        err_msg='MEG mismatch')
        # check EEG
        if fwd['sol']['data'].shape[0] > 306:
            assert_allclose(fwd['sol']['data'][306:],
                            fwd_py['sol']['data'][306:],
                            rtol=eeg_rtol, atol=eeg_atol,
                            err_msg='EEG mismatch')


def test_magnetic_dipole():
    """Test basic magnetic dipole forward calculation
    """
    trans = Transform('mri', 'head', np.eye(4))
    info = read_info(fname_raw)
    picks = pick_types(info, meg=True, eeg=False, exclude=[])
    info = pick_info(info, picks[:12])
    coils = _create_meg_coils(info['chs'], 'normal', trans)
    # magnetic dipole at device origin
    r0 = np.array([0., 13., -6.])
    for ch, coil in zip(info['chs'], coils):
        rr = (ch['loc'][:3] + r0) / 2.
        far_fwd = _magnetic_dipole_field_vec(r0[np.newaxis, :], [coil])
        near_fwd = _magnetic_dipole_field_vec(rr[np.newaxis, :], [coil])
        ratio = 8. if ch['ch_name'][-1] == '1' else 16.  # grad vs mag
        assert_allclose(np.median(near_fwd / far_fwd), ratio, atol=1e-1)


@testing.requires_testing_data
@requires_mne
def test_make_forward_solution_kit():
    """Test making fwd using KIT, BTI, and CTF (compensated) files
    """
    kit_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'kit',
                      'tests', 'data')
    sqd_path = op.join(kit_dir, 'test.sqd')
    mrk_path = op.join(kit_dir, 'test_mrk.sqd')
    elp_path = op.join(kit_dir, 'test_elp.txt')
    hsp_path = op.join(kit_dir, 'test_hsp.txt')
    trans_path = op.join(kit_dir, 'trans-sample.fif')
    fname_kit_raw = op.join(kit_dir, 'test_bin_raw.fif')

    bti_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'bti',
                      'tests', 'data')
    bti_pdf = op.join(bti_dir, 'test_pdf_linux')
    bti_config = op.join(bti_dir, 'test_config_linux')
    bti_hs = op.join(bti_dir, 'test_hs_linux')
    fname_bti_raw = op.join(bti_dir, 'exported4D_linux_raw.fif')

    fname_ctf_raw = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                            'data', 'test_ctf_comp_raw.fif')

    # first set up a small testing source space
    temp_dir = _TempDir()
    fname_src_small = op.join(temp_dir, 'sample-oct-2-src.fif')
    src = setup_source_space('sample', fname_src_small, 'oct2',
                             subjects_dir=subjects_dir, add_dist=False)
    n_src = 108  # this is the resulting # of verts in fwd

    # first use mne-C: convert file, make forward solution
    fwd = do_forward_solution('sample', fname_kit_raw, src=fname_src_small,
                              bem=fname_bem_meg, mri=trans_path,
                              eeg=False, meg=True, subjects_dir=subjects_dir)
    assert_true(isinstance(fwd, Forward))

    # now let's use python with the same raw file
    fwd_py = make_forward_solution(fname_kit_raw, trans_path, src,
                                   fname_bem_meg, eeg=False, meg=True)
    _compare_forwards(fwd, fwd_py, 157, n_src)
    assert_true(isinstance(fwd_py, Forward))

    # now let's use mne-python all the way
    raw_py = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path)
    # without ignore_ref=True, this should throw an error:
    assert_raises(NotImplementedError, make_forward_solution, raw_py.info,
                  src=src, eeg=False, meg=True,
                  bem=fname_bem_meg, trans=trans_path)

    # check that asking for eeg channels (even if they don't exist) is handled
    meg_only_info = pick_info(raw_py.info, pick_types(raw_py.info, meg=True,
                                                      eeg=False))
    fwd_py = make_forward_solution(meg_only_info, src=src, meg=True, eeg=True,
                                   bem=fname_bem_meg, trans=trans_path,
                                   ignore_ref=True)
    _compare_forwards(fwd, fwd_py, 157, n_src,
                      meg_rtol=1e-3, meg_atol=1e-7)

    # BTI python end-to-end versus C
    fwd = do_forward_solution('sample', fname_bti_raw, src=fname_src_small,
                              bem=fname_bem_meg, mri=trans_path,
                              eeg=False, meg=True, subjects_dir=subjects_dir)
    raw_py = read_raw_bti(bti_pdf, bti_config, bti_hs, preload=False)
    fwd_py = make_forward_solution(raw_py.info, src=src, eeg=False, meg=True,
                                   bem=fname_bem_meg, trans=trans_path)
    _compare_forwards(fwd, fwd_py, 248, n_src)

    # now let's test CTF w/compensation
    fwd_py = make_forward_solution(fname_ctf_raw, fname_trans, src,
                                   fname_bem_meg, eeg=False, meg=True)

    fwd = do_forward_solution('sample', fname_ctf_raw, mri=fname_trans,
                              src=fname_src_small, bem=fname_bem_meg,
                              eeg=False, meg=True, subjects_dir=subjects_dir)
    _compare_forwards(fwd, fwd_py, 274, n_src)

    # CTF with compensation changed in python
    ctf_raw = Raw(fname_ctf_raw, compensation=2)

    fwd_py = make_forward_solution(ctf_raw.info, fname_trans, src,
                                   fname_bem_meg, eeg=False, meg=True)
    with warnings.catch_warnings(record=True):
        fwd = do_forward_solution('sample', ctf_raw, mri=fname_trans,
                                  src=fname_src_small, bem=fname_bem_meg,
                                  eeg=False, meg=True,
                                  subjects_dir=subjects_dir)
    _compare_forwards(fwd, fwd_py, 274, n_src)


@slow_test
@testing.requires_testing_data
def test_make_forward_solution():
    """Test making M-EEG forward solution from python
    """
    fwd_py = make_forward_solution(fname_raw, fname_trans, fname_src,
                                   fname_bem, mindist=5.0, eeg=True, meg=True)
    assert_true(isinstance(fwd_py, Forward))
    fwd = read_forward_solution(fname_meeg)
    assert_true(isinstance(fwd, Forward))
    _compare_forwards(fwd, fwd_py, 366, 1494, meg_rtol=1e-3)


@testing.requires_testing_data
@requires_mne
def test_make_forward_solution_sphere():
    """Test making a forward solution with a sphere model"""
    temp_dir = _TempDir()
    fname_src_small = op.join(temp_dir, 'sample-oct-2-src.fif')
    src = setup_source_space('sample', fname_src_small, 'oct2',
                             subjects_dir=subjects_dir, add_dist=False)
    out_name = op.join(temp_dir, 'tmp-fwd.fif')
    run_subprocess(['mne_forward_solution', '--meg', '--eeg',
                    '--meas', fname_raw, '--src', fname_src_small,
                    '--mri', fname_trans, '--fwd', out_name])
    fwd = read_forward_solution(out_name)
    sphere = make_sphere_model(verbose=True)
    fwd_py = make_forward_solution(fname_raw, fname_trans, src, sphere,
                                   meg=True, eeg=True, verbose=True)
    _compare_forwards(fwd, fwd_py, 366, 108,
                      meg_rtol=5e-1, meg_atol=1e-6,
                      eeg_rtol=5e-1, eeg_atol=5e-1)
    # Since the above is pretty lax, let's check a different way
    for meg, eeg in zip([True, False], [False, True]):
        fwd_ = pick_types_forward(fwd, meg=meg, eeg=eeg)
        fwd_py_ = pick_types_forward(fwd, meg=meg, eeg=eeg)
        assert_allclose(np.corrcoef(fwd_['sol']['data'].ravel(),
                                    fwd_py_['sol']['data'].ravel())[0, 1],
                        1.0, rtol=1e-3)


@testing.requires_testing_data
@requires_mne
def test_do_forward_solution():
    """Test wrapping forward solution from python
    """
    temp_dir = _TempDir()
    existing_file = op.join(temp_dir, 'test.fif')
    with open(existing_file, 'w') as fid:
        fid.write('aoeu')

    mri = read_trans(fname_trans)
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
                  mri=fname_trans, mindist=dict(), subjects_dir=subjects_dir)
    # mindist as string but not 'all'
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_trans, eeg=False, mindist='yall',
                  subjects_dir=subjects_dir)
    # src, spacing, and bem as non-str
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_trans, src=1, subjects_dir=subjects_dir)
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_trans, spacing=1, subjects_dir=subjects_dir)
    assert_raises(ValueError, do_forward_solution, 'sample', fname_raw,
                  mri=fname_trans, bem=1, subjects_dir=subjects_dir)
    # no overwrite flag
    assert_raises(IOError, do_forward_solution, 'sample', fname_raw,
                  existing_file, mri=fname_trans, subjects_dir=subjects_dir)
    # let's catch an MNE error, this time about trans being wrong
    assert_raises(CalledProcessError, do_forward_solution, 'sample',
                  fname_raw, existing_file, trans=fname_trans, overwrite=True,
                  spacing='oct6', subjects_dir=subjects_dir)

    # No need to actually calculate and check here, since it's effectively
    # done in previous tests.


@slow_test
@testing.requires_testing_data
@requires_nibabel(False)
def test_forward_mixed_source_space():
    """Test making the forward solution for a mixed source space
    """
    temp_dir = _TempDir()
    # get the surface source space
    surf = read_source_spaces(fname_src)

    # setup two volume source spaces
    label_names = get_volume_labels_from_aseg(fname_aseg)
    vol_labels = [label_names[int(np.random.rand() * len(label_names))]
                  for _ in range(2)]
    vol1 = setup_volume_source_space('sample', fname=None, pos=20.,
                                     mri=fname_aseg,
                                     volume_label=vol_labels[0],
                                     add_interpolator=False)
    vol2 = setup_volume_source_space('sample', fname=None, pos=20.,
                                     mri=fname_aseg,
                                     volume_label=vol_labels[1],
                                     add_interpolator=False)

    # merge surfaces and volume
    src = surf + vol1 + vol2

    # calculate forward solution
    fwd = make_forward_solution(fname_raw, fname_trans, src, fname_bem, None)
    assert_true(repr(fwd))

    # extract source spaces
    src_from_fwd = fwd['src']

    # get the coordinate frame of each source space
    coord_frames = np.array([s['coord_frame'] for s in src_from_fwd])

    # assert that all source spaces are in head coordinates
    assert_true((coord_frames == FIFF.FIFFV_COORD_HEAD).all())

    # run tests for SourceSpaces.export_volume
    fname_img = op.join(temp_dir, 'temp-image.mgz')

    # head coordinates and mri_resolution, but trans file
    assert_raises(ValueError, src_from_fwd.export_volume, fname_img,
                  mri_resolution=True, trans=None)

    # head coordinates and mri_resolution, but wrong trans file
    vox_mri_t = vol1[0]['vox_mri_t']
    assert_raises(ValueError, src_from_fwd.export_volume, fname_img,
                  mri_resolution=True, trans=vox_mri_t)


run_tests_if_main()
