# Adapted from vispy
#
# License: BSD (3-clause)

import os
from nose.plugins.skip import SkipTest
from os import path as op
import sys

import mne
from mne.utils import run_tests_if_main


known_crlf = (
    'FreeSurferColorLUT.txt',
    'test_edf_stim_channel.txt',
    'FieldTrip.py',
)


def test_line_endings():
    """Test files in the repository for CR characters
    """
    if sys.platform == 'win32':
        raise SkipTest('Skipping line endings check on Windows')
    sys.stdout.flush()
    report = []
    import_dir = mne.__path__[0]
    for dirpath, dirnames, filenames in os.walk(import_dir):
        for fname in filenames:
            if op.splitext(fname)[1] in ('.pyc', '.pyo'):
                continue
            # Get filename
            filename = op.join(dirpath, fname)
            relfilename = op.relpath(filename, import_dir)
            # Open and check
            try:
                with open(filename, 'rb') as fid:
                    text = fid.read().decode('utf-8')
            except UnicodeDecodeError:
                continue  # Probably a binary file
            crcount = text.count('\r')
            if crcount and op.basename(fname) not in known_crlf:
                lfcount = text.count('\n')
                report.append('In %s found %i/%i CR/LF' %
                              (relfilename, crcount, lfcount))

    # Process result
    if len(report) > 0:
        raise AssertionError('Found %s files with incorrect endings:\n%s'
                             % (len(report), '\n'.join(report)))

run_tests_if_main()
