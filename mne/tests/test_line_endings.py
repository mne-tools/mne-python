# Adapted from vispy
#
# License: BSD (3-clause)

import os
from nose.tools import assert_raises
from nose.plugins.skip import SkipTest
from os import path as op
import sys

import mne
from mne.utils import run_tests_if_main, _TempDir


skip_files = (
    # known crlf
    'FreeSurferColorLUT.txt',
    'test_edf_stim_channel.txt',
    'FieldTrip.py',
    # binaries
    'test_hs_linux',
    'test_config_solaris',
    'test_config_linux',
    'test_hs_solaris',
    'test_pdf_linux',
    'test_pdf_solaris',
)


def _assert_line_endings(dir_):
    """Check line endings for a directory"""
    if sys.platform == 'win32':
        raise SkipTest('Skipping line endings check on Windows')
    report = list()
    for dirpath, dirnames, filenames in os.walk(dir_):
        for fname in filenames:
            if op.splitext(fname)[1] in ('.pyc', '.pyo', '.gz', '.mat',
                                         '.gif', '.fif', '.stc', '.data',
                                         '.eeg', '.edf', '.w', '.sqd',
                                         '.bdf', '.raw') or \
                    fname in skip_files:
                continue
            filename = op.join(dirpath, fname)
            relfilename = op.relpath(filename, dir_)
            with open(filename, 'rb') as fid:
                text = fid.read().decode('utf-8')
            crcount = text.count('\r')
            if crcount:
                report.append('In %s found %i/%i CR/LF' %
                              (relfilename, crcount, text.count('\n')))
    if len(report) > 0:
        raise AssertionError('Found %s files with incorrect endings:\n%s'
                             % (len(report), '\n'.join(report)))


def test_line_endings():
    """Test line endings of mne-python
    """
    tempdir = _TempDir()
    with open(op.join(tempdir, 'foo.py'), 'wb') as fid:
        fid.write('bad\r\ngood\n')
    assert_raises(AssertionError, _assert_line_endings, tempdir)
    # now check mne
    _assert_line_endings(mne.__path__[0])

run_tests_if_main()
