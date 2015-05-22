# -*- coding: utf-8 -*-
import os
from os import path as op
import shutil
import warnings
from nose.tools import assert_true

from mne.commands import mne_watershed_bem
from mne.utils import (run_tests_if_main, _TempDir, ArgvSetter)
from mne.datasets import sample


subjects_dir = op.join(sample.data_path(download=False), 'subjects')

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


def test_watershed_bem():
    """Test mne watershed bem"""
    check_usage(mne_watershed_bem, force_help=True)
    orig_subject_dir = os.getenv('SUBJECTS_DIR', None)
    orig_subject = os.getenv('SUBJECT', None)
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

    with ArgvSetter(('-s', 'sample', '-o', '-d', tempdir),
                     disable_stdout=False, disable_stderr=False):
        mne_watershed_bem.run()

    os.environ['SUBJECTS_DIR'] = orig_subject_dir
    os.environ['SUBJECT'] = orig_subject

run_tests_if_main()
