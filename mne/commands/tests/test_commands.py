# -*- coding: utf-8 -*-
import sys

from nose.tools import assert_true
from mne.commands import (mne_browse_raw, mne_bti2fiff, mne_clean_eog_ecg,
                          mne_compute_proj_ecg, mne_compute_proj_eog,
                          mne_coreg, mne_flash_bem_model, mne_kit2fiff,
                          mne_make_scalp_surfaces, mne_maxfilter,
                          mne_report, mne_surf2bem)
from mne.utils import run_tests_if_main
from mne.externals.six.moves import StringIO


class argvr(object):
    def __init__(self, *args):
        self.argv = list(('python',) + args)
        self.stdout = StringIO()
        self.stderr = StringIO()

    def __enter__(self):
        self.orig_argv = sys.argv
        sys.argv = self.argv
        self.orig_stdout = sys.stdout
        sys.stdout = self.stdout
        self.orig_stderr = sys.stderr
        sys.stderr = self.stderr
        return self

    def __exit__(self, *args):
        sys.argv = self.orig_argv
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr


def check_usage(module):
    """Helper to ensure we print usage"""
    with argvr() as a:
        module.run()
        assert_true('Usage: ' in a.stdout.getvalue())


def test_browse_raw():
    """Test mne browse_raw"""
    check_usage(mne_browse_raw)


def test_bti2fiff():
    """Test mne bti2fiff"""
    check_usage(mne_bti2fiff)


def test_clean_eog_ecg():
    """Test mne cleane_eog_ecg"""
    check_usage(mne_clean_eog_ecg)


def test_compute_proj_ecg():
    """Test mne compute_proj_ecg"""
    check_usage(mne_compute_proj_ecg)


def test_compute_proj_eog():
    """Test mne compute_proj_eog"""
    check_usage(mne_compute_proj_eog)


def test_coreg():
    """Test mne coreg"""
    assert_true(hasattr(mne_coreg, 'run'))


def test_flash_bem_model():
    """Test mne flash_bem_model"""
    assert_true(hasattr(mne_flash_bem_model, 'run'))


def test_kit2fiff():
    """Test mne kit2fiff"""
    assert_true(hasattr(mne_kit2fiff, 'run'))


def test_make_scalp_surfaces():
    """Test mne make_scalp_surfaces"""
    check_usage(mne_make_scalp_surfaces)


def test_maxfilter():
    """Test mne maxfilter"""
    check_usage(mne_maxfilter)


def test_report():
    """Test mne report"""
    check_usage(mne_report)


def test_surf2bem():
    """Test mne surf2bem"""
    check_usage(mne_surf2bem)


run_tests_if_main()
