# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import os.path as op
import glob
import warnings

from nose.tools import assert_true, assert_equal

from mne import read_evokeds
from mne.report import Report
from mne.io import Raw
from mne.utils import _TempDir

base_dir = op.realpath(op.join(op.dirname(__file__), '..', 'io', 'tests',
                               'data'))
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

warnings.simplefilter('always')  # enable b/c these tests throw warnings

tempdir = _TempDir()


def test_parse_folder():
    """Test parsing of folders for MNE-Report.
    """

    report = Report(info_fname=raw_fname)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        report.parse_folder(data_path=base_dir)

    # Check correct paths and filenames
    assert_true(raw_fname in report.fnames)
    assert_true(event_name in report.fnames)
    assert_true(report.data_path == base_dir)

    # Check if raw repr is printed correctly
    raw = Raw(raw_fname)
    raw_idx = [ii for ii, fname in enumerate(report.fnames)
               if fname == raw_fname][0]
    raw_html = report.html[raw_idx]
    assert_true(raw_html.find(repr(raw)[1:-1]) != -1)
    assert_true(raw_html.find(str(raw.info['sfreq'])) != -1)
    assert_true(raw_html.find('class="raw"') != -1)
    assert_true(raw_html.find(raw_fname) != -1)

    c_evoked = read_evokeds(evoked_nf_name)
    evoked_idx = [ii for ii, fname in enumerate(report.fnames)
                  if fname == evoked_nf_name][0]
    assert_true(report.html[evoked_idx].find(evoked_nf_name) != -1)

    # Check saving functionality
    report.data_path = tempdir
    report.save(fname=op.join(tempdir, 'report.html'), open_browser=False)
    assert_true(op.isfile(op.join(tempdir, 'report.html')))

    # Check if all files were rendered
    fnames = glob.glob(op.join(base_dir, '*.fif'))
    fnames = [fname for fname in fnames if
              fname.endswith(('-eve.fif', '-ave.fif', '-cov.fif',
                              '-sol.fif', '-fwd.fif', '-inv.fif',
                              '-src.fif', '-trans.fif', 'raw.fif',
                              'sss.fif', '-epo.fif'))]

    # Check if TOC contains names of all files
    for fname in fnames:
        assert_true(report.html[1].find(fname) != -1)

    assert_equal(len(report.fnames), len(fnames))
    # different evoked conditions have different ids
    assert_equal(report.initial_id, len(report.html) + len(c_evoked) - 1)

    # Check saving same report to new filename
    report.save(fname=op.join(tempdir, 'report2.html'), open_browser=False)
    assert_true(op.isfile(op.join(tempdir, 'report2.html')))

    # Check overwriting file
    report.save(fname=op.join(tempdir, 'report.html'), open_browser=False,
                overwrite=True)
    assert_true(op.isfile(op.join(tempdir, 'report2.html')))
