# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import os
import os.path as op
import glob
import warnings

from nose.tools import assert_true, assert_equal, assert_raises

from mne.datasets import testing
from mne.report import Report
from mne.utils import (_TempDir, requires_mayavi, requires_nibabel,
                       requires_PIL, run_tests_if_main)

data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
base_dir = op.join(data_dir, 'MEG', 'sample')

raw_fname = op.join(base_dir, 'sample_audvis_trunc_raw.fif')
event_name = op.join(base_dir, 'sample_audvis_trunc_raw-eve.fif')
evoked_fname = op.join(base_dir, 'sample_audvis_trunc-ave.fif')

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

os.environ['MNE_REPORT_TESTING'] = 'True'
warnings.simplefilter('always')  # enable b/c these tests throw warnings


@testing.requires_testing_data
@requires_PIL()
def test_render_report():
    """Test rendering -*.fif files for mne report.
    """
    tempdir = _TempDir()

    report = Report(info_fname=raw_fname, subjects_dir=subjects_dir)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        report.parse_folder(data_path=base_dir)
    assert_true(len(w) == 1)

    # Check correct paths and filenames
    assert_true(raw_fname in report.fnames)
    assert_true(event_name in report.fnames)
    assert_true(report.data_path == base_dir)

    # Check if all files were rendered in the report
    fnames = glob.glob(op.join(base_dir, '*.fif'))
    fnames = [fname for fname in fnames if
              fname.endswith(('-eve.fif', '-ave.fif', '-cov.fif',
                              '-sol.fif', '-fwd.fif', '-inv.fif',
                              '-src.fif', 'raw.fif',
                              'sss.fif', '-epo.fif'))]
    # last file above gets created by another test, and it shouldn't be there

    for fname in fnames:
        assert_true(''.join(report.html).find(op.basename(fname)) != -1)

    assert_equal(len(report.fnames), len(fnames))
    assert_equal(len(report.html), len(report.fnames))
    assert_equal(len(report.fnames), report.initial_id - 6)

    # Check saving functionality
    report.data_path = tempdir
    report.save(fname=op.join(tempdir, 'report.html'), open_browser=False)
    assert_true(op.isfile(op.join(tempdir, 'report.html')))

    assert_equal(len(report.html), len(fnames))
    assert_equal(len(report.html), len(report.fnames))

    # Check saving same report to new filename
    report.save(fname=op.join(tempdir, 'report2.html'), open_browser=False)
    assert_true(op.isfile(op.join(tempdir, 'report2.html')))

    # Check overwriting file
    report.save(fname=op.join(tempdir, 'report.html'), open_browser=False,
                overwrite=True)
    assert_true(op.isfile(op.join(tempdir, 'report.html')))


@requires_mayavi()
@requires_PIL()
def test_render_add_sections():
    """Test adding figures/images to section.
    """
    tempdir = _TempDir()
    import matplotlib.pyplot as plt

    report = Report(subjects_dir=subjects_dir)
    # Check add_figs_to_section functionality
    fig = plt.plot([1, 2], [1, 2])[0].figure
    report.add_figs_to_section(figs=fig,  # test non-list input
                               captions=['evoked response'])
    assert_raises(ValueError, report.add_figs_to_section, figs=[fig, fig],
                  captions='H')

    # Check add_images_to_section
    img_fname = op.join(tempdir, 'testimage.png')
    fig.savefig(img_fname)
    report.add_images_to_section(fnames=[img_fname],
                                 captions=['evoked response'])
    assert_raises(ValueError, report.add_images_to_section,
                  fnames=[img_fname, img_fname], captions='H')

    # Check deprecation of add_section
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        report.add_section(figs=fig,
                           captions=['evoked response'])
        assert_true(w[0].category == DeprecationWarning)


@testing.requires_testing_data
@requires_nibabel()
def test_render_mri():
    """Test rendering MRI for mne report.
    """
    report = Report(info_fname=raw_fname,
                    subject='sample', subjects_dir=subjects_dir)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        report.parse_folder(data_path=data_dir,
                            pattern='*sample_audvis_raw-trans.fif')


run_tests_if_main()
