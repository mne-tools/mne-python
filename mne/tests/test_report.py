# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import os
import os.path as op
import glob
import warnings

from nose.tools import assert_true, assert_equal, assert_raises

from mne import read_evokeds
from mne.datasets import sample
from mne.report import Report
from mne.io import Raw
from mne.utils import _TempDir

data_dir = sample.data_path(download=False)
base_dir = op.realpath(op.join(op.dirname(__file__), '..', 'io', 'tests',
                               'data'))
subjects_dir = op.join(data_dir, 'subjects')

raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked1_fname = op.join(base_dir, 'test-nf-ave.fif')
evoked2_fname = op.join(base_dir, 'test-ave.fif')

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

os.environ['MNE_REPORT_TESTING'] = 'True'
warnings.simplefilter('always')  # enable b/c these tests throw warnings

tempdir = _TempDir()


def test_render_report():
    """Test rendering -*.fif files for mne report.
    """

    report = Report(info_fname=raw_fname)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        report.parse_folder(data_path=base_dir)
    assert_true(len(w) == 1)

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

    # Check if all files were rendered in the report
    fnames = glob.glob(op.join(base_dir, '*.fif'))
    bad_name = 'test_ctf_comp_raw-eve.fif'
    decrement = any(fname.endswith(bad_name) for fname in fnames)
    fnames = [fname for fname in fnames if
              fname.endswith(('-eve.fif', '-ave.fif', '-cov.fif',
                              '-sol.fif', '-fwd.fif', '-inv.fif',
                              '-src.fif', '-trans.fif', 'raw.fif',
                              'sss.fif', '-epo.fif')) and
              not fname.endswith(bad_name)]
    # last file above gets created by another test, and it shouldn't be there

    for fname in fnames:
        assert_true(''.join(report.html).find(op.basename(fname)) != -1)

    assert_equal(len(report.fnames), len(fnames))
    assert_equal(len(report.html), len(report.fnames))

    evoked1 = read_evokeds(evoked1_fname)
    evoked2 = read_evokeds(evoked2_fname)
    assert_equal(len(report.fnames) + len(evoked1) + len(evoked2) - 2,
                 report.initial_id - decrement)

    # Check saving functionality
    report.data_path = tempdir
    report.save(fname=op.join(tempdir, 'report.html'), open_browser=False)
    assert_true(op.isfile(op.join(tempdir, 'report.html')))

    # Check add_section functionality
    fig = evoked1[0].plot(show=False)
    report.add_section(figs=fig,  # test non-list input
                       captions=['evoked response'])
    assert_equal(len(report.html), len(fnames) + 1)
    assert_equal(len(report.html), len(report.fnames))
    assert_raises(ValueError, report.add_section, figs=[fig, fig],
                  captions='H')

    # Check saving same report to new filename
    report.save(fname=op.join(tempdir, 'report2.html'), open_browser=False)
    assert_true(op.isfile(op.join(tempdir, 'report2.html')))

    # Check overwriting file
    report.save(fname=op.join(tempdir, 'report.html'), open_browser=False,
                overwrite=True)
    assert_true(op.isfile(op.join(tempdir, 'report.html')))


@sample.requires_sample_data
def test_render_mri():
    """Test rendering MRI for mne report.
    """
    report = Report(info_fname=raw_fname,
                    subject='sample', subjects_dir=subjects_dir)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        report.parse_folder(data_path=data_dir,
                            pattern='*sample_audvis_raw-trans.fif')
