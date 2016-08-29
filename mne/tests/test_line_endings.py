# Author: Eric Larson <larson.eric.d@gmail.com>
#         Adapted from vispy
#
# License: BSD (3-clause)

import os
from nose.tools import assert_raises
from nose.plugins.skip import SkipTest
from os import path as op
import sys

from mne.utils import run_tests_if_main, _TempDir, _get_root_dir


skip_files = (
    # known crlf
    'FreeSurferColorLUT.txt',
    'test_edf_stim_channel.txt',
    'FieldTrip.py',
    'license.txt',
    # part of testing compatibility with older BV formats is testing
    # the line endings and coding schemes used there
    'test_old_layout_latin1_software_filter.vhdr',
    'test_old_layout_latin1_software_filter.vmrk'
)


def _assert_line_endings(dir_):
    """Check line endings for a directory"""
    if sys.platform == 'win32':
        raise SkipTest('Skipping line endings check on Windows')
    report = list()
    good_exts = ('.py', '.dat', '.sel', '.lout', '.css', '.js', '.lay', '.txt',
                 '.elc', '.csd', '.sfp', '.json', '.hpts', '.vmrk', '.vhdr',
                 '.head', '.eve', '.ave', '.cov', '.label')
    for dirpath, dirnames, filenames in os.walk(dir_):
        for fname in filenames:
            if op.splitext(fname)[1] not in good_exts or fname in skip_files:
                continue
            filename = op.join(dirpath, fname)
            relfilename = op.relpath(filename, dir_)
            try:
                with open(filename, 'rb') as fid:
                    text = fid.read().decode('utf-8')
            except UnicodeDecodeError:
                report.append('In %s found non-decodable bytes' % relfilename)
            else:
                crcount = text.count('\r')
                if crcount:
                    report.append('In %s found %i/%i CR/LF' %
                                  (relfilename, crcount, text.count('\n')))
    if len(report) > 0:
        raise AssertionError('Found %s files with incorrect endings:\n%s'
                             % (len(report), '\n'.join(report)))


def test_line_endings():
    """Test line endings of mne-python"""
    tempdir = _TempDir()
    with open(op.join(tempdir, 'foo'), 'wb') as fid:
        fid.write('bad\r\ngood\n'.encode('ascii'))
    _assert_line_endings(tempdir)
    with open(op.join(tempdir, 'bad.py'), 'wb') as fid:
        fid.write(b'\x97')
    assert_raises(AssertionError, _assert_line_endings, tempdir)
    with open(op.join(tempdir, 'bad.py'), 'wb') as fid:
        fid.write('bad\r\ngood\n'.encode('ascii'))
    assert_raises(AssertionError, _assert_line_endings, tempdir)
    # now check mne
    _assert_line_endings(_get_root_dir())

run_tests_if_main()
