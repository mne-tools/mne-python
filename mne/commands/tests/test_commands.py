# -*- coding: utf-8 -*-
import os
from os import path as op
import shutil
import glob
import warnings
from nose.tools import assert_true, assert_raises
from numpy.testing import assert_equal, assert_allclose

from mne import concatenate_raws, read_bem_surfaces
from mne.commands import (mne_browse_raw, mne_bti2fiff, mne_clean_eog_ecg,
                          mne_compute_proj_ecg, mne_compute_proj_eog,
                          mne_coreg, mne_kit2fiff,
                          mne_make_scalp_surfaces, mne_maxfilter,
                          mne_report, mne_surf2bem, mne_watershed_bem,
                          mne_compare_fiff, mne_flash_bem, mne_show_fiff,
                          mne_show_info)
from mne.datasets import testing, sample
from mne.io import read_raw_fif
from mne.utils import (run_tests_if_main, _TempDir, requires_mne, requires_PIL,
                       requires_mayavi, requires_tvtk, requires_freesurfer,
                       ArgvSetter, slow_test, ultra_slow_test)


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')

subjects_dir = op.join(testing.data_path(download=False), 'subjects')

warnings.simplefilter('always')


def check_usage(module, force_help=False):
    """Helper to ensure we print usage"""
    args = ('--help',) if force_help else ()
    with ArgvSetter(args) as out:
        try:
            module.run()
        except SystemExit:
            pass
        assert_true('Usage: ' in out.stdout.getvalue())


@slow_test
def test_browse_raw():
    """Test mne browse_raw."""
    check_usage(mne_browse_raw)


def test_bti2fiff():
    """Test mne bti2fiff."""
    check_usage(mne_bti2fiff)


def test_compare_fiff():
    """Test mne compare_fiff."""
    check_usage(mne_compare_fiff)


def test_show_fiff():
    """Test mne compare_fiff."""
    check_usage(mne_show_fiff)
    with ArgvSetter((raw_fname,)):
        mne_show_fiff.run()


@requires_mne
def test_clean_eog_ecg():
    """Test mne clean_eog_ecg."""
    check_usage(mne_clean_eog_ecg)
    tempdir = _TempDir()
    raw = concatenate_raws([read_raw_fif(f)
                            for f in [raw_fname, raw_fname, raw_fname]])
    raw.info['bads'] = ['MEG 2443']
    use_fname = op.join(tempdir, op.basename(raw_fname))
    raw.save(use_fname)
    with ArgvSetter(('-i', use_fname, '--quiet')):
        mne_clean_eog_ecg.run()
    fnames = glob.glob(op.join(tempdir, '*proj.fif'))
    assert_true(len(fnames) == 2)  # two projs
    fnames = glob.glob(op.join(tempdir, '*-eve.fif'))
    assert_true(len(fnames) == 3)  # raw plus two projs


@slow_test
def test_compute_proj_ecg_eog():
    """Test mne compute_proj_ecg/eog."""
    for fun in (mne_compute_proj_ecg, mne_compute_proj_eog):
        check_usage(fun)
        tempdir = _TempDir()
        use_fname = op.join(tempdir, op.basename(raw_fname))
        bad_fname = op.join(tempdir, 'bads.txt')
        with open(bad_fname, 'w') as fid:
            fid.write('MEG 2443\n')
        shutil.copyfile(raw_fname, use_fname)
        with ArgvSetter(('-i', use_fname, '--bad=' + bad_fname,
                         '--rej-eeg', '150')):
            fun.run()
        fnames = glob.glob(op.join(tempdir, '*proj.fif'))
        assert_true(len(fnames) == 1)
        fnames = glob.glob(op.join(tempdir, '*-eve.fif'))
        assert_true(len(fnames) == 1)


def test_coreg():
    """Test mne coreg."""
    assert_true(hasattr(mne_coreg, 'run'))


def test_kit2fiff():
    """Test mne kit2fiff."""
    # Can't check
    check_usage(mne_kit2fiff, force_help=True)


@requires_tvtk
@testing.requires_testing_data
def test_make_scalp_surfaces():
    """Test mne make_scalp_surfaces."""
    check_usage(mne_make_scalp_surfaces)
    # Copy necessary files to avoid FreeSurfer call
    tempdir = _TempDir()
    surf_path = op.join(subjects_dir, 'sample', 'surf')
    surf_path_new = op.join(tempdir, 'sample', 'surf')
    os.mkdir(op.join(tempdir, 'sample'))
    os.mkdir(surf_path_new)
    subj_dir = op.join(tempdir, 'sample', 'bem')
    os.mkdir(subj_dir)
    shutil.copy(op.join(surf_path, 'lh.seghead'), surf_path_new)

    orig_fs = os.getenv('FREESURFER_HOME', None)
    if orig_fs is not None:
        del os.environ['FREESURFER_HOME']
    cmd = ('-s', 'sample', '--subjects-dir', tempdir)
    os.environ['_MNE_TESTING_SCALP'] = 'true'
    dense_fname = op.join(subj_dir, 'sample-head-dense.fif')
    medium_fname = op.join(subj_dir, 'sample-head-medium.fif')
    try:
        with ArgvSetter(cmd, disable_stdout=False, disable_stderr=False):
            assert_raises(RuntimeError, mne_make_scalp_surfaces.run)
            os.environ['FREESURFER_HOME'] = tempdir  # don't actually use it
            mne_make_scalp_surfaces.run()
            assert_true(op.isfile(dense_fname))
            assert_true(op.isfile(medium_fname))
            assert_raises(IOError, mne_make_scalp_surfaces.run)  # no overwrite
    finally:
        if orig_fs is not None:
            os.environ['FREESURFER_HOME'] = orig_fs
        else:
            del os.environ['FREESURFER_HOME']
        del os.environ['_MNE_TESTING_SCALP']
    # actually check the outputs
    head_py = read_bem_surfaces(dense_fname)
    assert_equal(len(head_py), 1)
    head_py = head_py[0]
    head_c = read_bem_surfaces(op.join(subjects_dir, 'sample', 'bem',
                                       'sample-head-dense.fif'))[0]
    assert_allclose(head_py['rr'], head_c['rr'])


def test_maxfilter():
    """Test mne maxfilter."""
    check_usage(mne_maxfilter)
    with ArgvSetter(('-i', raw_fname, '--st', '--movecomp', '--linefreq', '60',
                     '--trans', raw_fname)) as out:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            os.environ['_MNE_MAXFILTER_TEST'] = 'true'
            try:
                mne_maxfilter.run()
            finally:
                del os.environ['_MNE_MAXFILTER_TEST']
        assert_true(len(w) == 1)
        for check in ('maxfilter', '-trans', '-movecomp'):
            assert_true(check in out.stdout.getvalue(), check)


@slow_test
@requires_mayavi
@requires_PIL
@testing.requires_testing_data
def test_report():
    """Test mne report."""
    check_usage(mne_report)
    tempdir = _TempDir()
    use_fname = op.join(tempdir, op.basename(raw_fname))
    shutil.copyfile(raw_fname, use_fname)
    with ArgvSetter(('-p', tempdir, '-i', use_fname, '-d', subjects_dir,
                     '-s', 'sample', '--no-browser', '-m', '30')):
        mne_report.run()
    fnames = glob.glob(op.join(tempdir, '*.html'))
    assert_true(len(fnames) == 1)


def test_surf2bem():
    """Test mne surf2bem."""
    check_usage(mne_surf2bem)


@ultra_slow_test
@requires_freesurfer
@testing.requires_testing_data
def test_watershed_bem():
    """Test mne watershed bem."""
    check_usage(mne_watershed_bem)
    # Copy necessary files to tempdir
    tempdir = _TempDir()
    mridata_path = op.join(subjects_dir, 'sample', 'mri')
    mridata_path_new = op.join(tempdir, 'sample', 'mri')
    os.mkdir(op.join(tempdir, 'sample'))
    os.mkdir(mridata_path_new)
    if op.exists(op.join(mridata_path, 'T1')):
        shutil.copytree(op.join(mridata_path, 'T1'), op.join(mridata_path_new,
                        'T1'))
    if op.exists(op.join(mridata_path, 'T1.mgz')):
        shutil.copyfile(op.join(mridata_path, 'T1.mgz'),
                        op.join(mridata_path_new, 'T1.mgz'))

    with ArgvSetter(('-d', tempdir, '-s', 'sample', '-o'),
                    disable_stdout=False, disable_stderr=False):
        mne_watershed_bem.run()


@ultra_slow_test
@requires_freesurfer
@sample.requires_sample_data
def test_flash_bem():
    """Test mne flash_bem."""
    check_usage(mne_flash_bem, force_help=True)
    # Using the sample dataset
    subjects_dir = op.join(sample.data_path(download=False), 'subjects')
    # Copy necessary files to tempdir
    tempdir = _TempDir()
    mridata_path = op.join(subjects_dir, 'sample', 'mri')
    mridata_path_new = op.join(tempdir, 'sample', 'mri')
    os.makedirs(op.join(mridata_path_new, 'flash'))
    os.makedirs(op.join(tempdir, 'sample', 'bem'))
    shutil.copyfile(op.join(mridata_path, 'T1.mgz'),
                    op.join(mridata_path_new, 'T1.mgz'))
    shutil.copyfile(op.join(mridata_path, 'brain.mgz'),
                    op.join(mridata_path_new, 'brain.mgz'))
    # Copy the available mri/flash/mef*.mgz files from the dataset
    files = glob.glob(op.join(mridata_path, 'flash', 'mef*.mgz'))
    for infile in files:
        shutil.copyfile(infile, op.join(mridata_path_new, 'flash',
                                        op.basename(infile)))
    # Test mne flash_bem with --noconvert option
    # (since there are no DICOM Flash images in dataset)
    currdir = os.getcwd()
    with ArgvSetter(('-d', tempdir, '-s', 'sample', '-n'),
                    disable_stdout=False, disable_stderr=False):
        mne_flash_bem.run()
    os.chdir(currdir)


def test_show_info():
    """Test mne show_info."""
    check_usage(mne_show_info)
    with ArgvSetter((raw_fname,)):
        mne_show_info.run()


run_tests_if_main()
