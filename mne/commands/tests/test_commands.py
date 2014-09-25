# -*- coding: utf-8 -*-
import os
from os import path as op
import shutil
import glob
import warnings
from nose.tools import assert_true

from mne.commands import (mne_browse_raw, mne_bti2fiff, mne_clean_eog_ecg,
                          mne_compute_proj_ecg, mne_compute_proj_eog,
                          mne_coreg, mne_flash_bem_model, mne_kit2fiff,
                          mne_make_scalp_surfaces, mne_maxfilter,
                          mne_report, mne_surf2bem)
from mne.utils import (run_tests_if_main, _TempDir, requires_mne, requires_PIL,
                       requires_mayavi, ArgvSetter)
from mne.io import Raw
from mne.datasets import testing


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


def test_browse_raw():
    """Test mne browse_raw"""
    check_usage(mne_browse_raw)


def test_bti2fiff():
    """Test mne bti2fiff"""
    check_usage(mne_bti2fiff)


@requires_mne
def test_clean_eog_ecg():
    """Test mne clean_eog_ecg"""
    check_usage(mne_clean_eog_ecg)
    tempdir = _TempDir()
    raw = Raw([raw_fname, raw_fname, raw_fname])
    raw.info['bads'] = ['MEG 2443']
    use_fname = op.join(tempdir, op.basename(raw_fname))
    raw.save(use_fname)
    with ArgvSetter(('-i', use_fname, '--quiet')):
        mne_clean_eog_ecg.run()
    fnames = glob.glob(op.join(tempdir, '*proj.fif'))
    assert_true(len(fnames) == 2)  # two projs
    fnames = glob.glob(op.join(tempdir, '*-eve.fif'))
    assert_true(len(fnames) == 3)  # raw plus two projs


def test_compute_proj_ecg_eog():
    """Test mne compute_proj_ecg/eog"""
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
    """Test mne coreg"""
    assert_true(hasattr(mne_coreg, 'run'))


def test_flash_bem_model():
    """Test mne flash_bem_model"""
    assert_true(hasattr(mne_flash_bem_model, 'run'))
    check_usage(mne_flash_bem_model)


def test_kit2fiff():
    """Test mne kit2fiff"""
    # Can't check
    check_usage(mne_kit2fiff, force_help=True)


def test_make_scalp_surfaces():
    """Test mne make_scalp_surfaces"""
    check_usage(mne_make_scalp_surfaces)


def test_maxfilter():
    """Test mne maxfilter"""
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


@requires_mayavi()
@requires_PIL()
@testing.requires_testing_data
def test_report():
    """Test mne report"""
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
    """Test mne surf2bem"""
    check_usage(mne_surf2bem)


run_tests_if_main()
