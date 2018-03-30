# -*- coding: utf-8 -*-
# Authors: Mainak Jas <mainak@neuro.hut.fi>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import glob
import os
import os.path as op
import shutil
import warnings

import numpy as np
from nose.tools import assert_true, assert_equal, assert_raises
import pytest

from mne import Epochs, read_events, read_evokeds
from mne.io import read_raw_fif
from mne.datasets import testing
from mne.report import Report
from mne.utils import (_TempDir, requires_mayavi, requires_nibabel,
                       run_tests_if_main, traits_test)
from mne.viz import plot_alignment

import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
report_dir = op.join(data_dir, 'MEG', 'sample')
raw_fname = op.join(report_dir, 'sample_audvis_trunc_raw.fif')
ms_fname = op.join(data_dir, 'SSS', 'test_move_anon_raw.fif')
event_fname = op.join(report_dir, 'sample_audvis_trunc_raw-eve.fif')
cov_fname = op.join(report_dir, 'sample_audvis_trunc-cov.fif')
fwd_fname = op.join(report_dir, 'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
trans_fname = op.join(report_dir, 'sample_audvis_trunc-trans.fif')
inv_fname = op.join(report_dir,
                    'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
mri_fname = op.join(subjects_dir, 'sample', 'mri', 'T1.mgz')

base_dir = op.realpath(op.join(op.dirname(__file__), '..', 'io', 'tests',
                               'data'))
evoked_fname = op.join(base_dir, 'test-ave.fif')

# Set our plotters to test mode

warnings.simplefilter('always')  # enable b/c these tests throw warnings


@pytest.mark.slowtest
@testing.requires_testing_data
def test_render_report():
    """Test rendering -*.fif files for mne report."""
    tempdir = _TempDir()
    raw_fname_new = op.join(tempdir, 'temp_raw.fif')
    ms_fname_new = op.join(tempdir, 'temp_ms_raw.fif')
    event_fname_new = op.join(tempdir, 'temp_raw-eve.fif')
    cov_fname_new = op.join(tempdir, 'temp_raw-cov.fif')
    fwd_fname_new = op.join(tempdir, 'temp_raw-fwd.fif')
    inv_fname_new = op.join(tempdir, 'temp_raw-inv.fif')
    for a, b in [[raw_fname, raw_fname_new],
                 [ms_fname, ms_fname_new],
                 [event_fname, event_fname_new],
                 [cov_fname, cov_fname_new],
                 [fwd_fname, fwd_fname_new],
                 [inv_fname, inv_fname_new]]:
        shutil.copyfile(a, b)

    # create and add -epo.fif and -ave.fif files
    epochs_fname = op.join(tempdir, 'temp-epo.fif')
    evoked_fname = op.join(tempdir, 'temp-ave.fif')
    # Speed it up by picking channels
    raw = read_raw_fif(raw_fname_new, preload=True)
    raw.pick_channels(['MEG 0111', 'MEG 0121'])
    raw.del_proj()
    epochs = Epochs(raw, read_events(event_fname), 1, -0.2, 0.2)
    epochs.save(epochs_fname)
    # This can take forever (stall Travis), so let's make it fast
    # Also, make sure crop range is wide enough to avoid rendering bug
    epochs.average().crop(0.1, 0.2).save(evoked_fname)

    report = Report(info_fname=raw_fname_new, subjects_dir=subjects_dir)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        report.parse_folder(data_path=tempdir, on_error='raise')
    assert_true(len(w) >= 1)
    assert_true(repr(report))

    # Check correct paths and filenames
    fnames = glob.glob(op.join(tempdir, '*.fif'))
    for fname in fnames:
        assert_true(op.basename(fname) in
                    [op.basename(x) for x in report.fnames])
        assert_true(''.join(report.html).find(op.basename(fname)) != -1)

    assert_equal(len(report.fnames), len(fnames))
    assert_equal(len(report.html), len(report.fnames))
    assert_equal(len(report.fnames), len(report))

    # Check saving functionality
    report.data_path = tempdir
    fname = op.join(tempdir, 'report.html')
    report.save(fname=fname, open_browser=False)
    assert_true(op.isfile(fname))
    with open(fname, 'rb') as fid:
        html = fid.read().decode('utf-8')
    assert '(MaxShield on)' in html

    assert_equal(len(report.html), len(fnames))
    assert_equal(len(report.html), len(report.fnames))

    # Check saving same report to new filename
    report.save(fname=op.join(tempdir, 'report2.html'), open_browser=False)
    assert_true(op.isfile(op.join(tempdir, 'report2.html')))

    # Check overwriting file
    report.save(fname=op.join(tempdir, 'report.html'), open_browser=False,
                overwrite=True)
    assert_true(op.isfile(op.join(tempdir, 'report.html')))

    # Check pattern matching with multiple patterns
    pattern = ['*raw.fif', '*eve.fif']
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        report.parse_folder(data_path=tempdir, pattern=pattern)
    assert_true(len(w) >= 1)
    assert_true(repr(report))

    fnames = glob.glob(op.join(tempdir, '*.raw')) + \
        glob.glob(op.join(tempdir, '*.raw'))
    for fname in fnames:
        assert_true(op.basename(fname) in
                    [op.basename(x) for x in report.fnames])
        assert_true(''.join(report.html).find(op.basename(fname)) != -1)

    assert_raises(ValueError, Report, image_format='foo')
    assert_raises(ValueError, Report, image_format=None)

    # SVG rendering
    report = Report(info_fname=raw_fname_new, subjects_dir=subjects_dir,
                    image_format='svg')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        report.parse_folder(data_path=tempdir, on_error='raise')

    # ndarray support smoke test
    report.add_figs_to_section(np.zeros((2, 3, 3)), 'caption', 'section')


@testing.requires_testing_data
@requires_mayavi
@traits_test
def test_render_add_sections():
    """Test adding figures/images to section."""
    tempdir = _TempDir()
    import matplotlib.pyplot as plt
    report = Report(subjects_dir=subjects_dir)
    # Check add_figs_to_section functionality
    fig = plt.plot([1, 2], [1, 2])[0].figure
    report.add_figs_to_section(figs=fig,  # test non-list input
                               captions=['evoked response'], scale=1.2,
                               image_format='svg')
    assert_raises(ValueError, report.add_figs_to_section, figs=[fig, fig],
                  captions='H')
    assert_raises(ValueError, report.add_figs_to_section, figs=fig,
                  captions=['foo'], scale=0, image_format='svg')
    assert_raises(ValueError, report.add_figs_to_section, figs=fig,
                  captions=['foo'], scale=1e-10, image_format='svg')
    # need to recreate because calls above change size
    fig = plt.plot([1, 2], [1, 2])[0].figure

    # Check add_images_to_section with png
    img_fname = op.join(tempdir, 'testimage.png')
    fig.savefig(img_fname)
    report.add_images_to_section(fnames=[img_fname],
                                 captions=['evoked response'])

    report.add_images_to_section(fnames=[img_fname],
                                 captions=['evoked response'])

    assert_raises(ValueError, report.add_images_to_section,
                  fnames=[img_fname, img_fname], captions='H')

    assert_raises(ValueError, report.add_images_to_section,
                  fnames=['foobar.xxx'], captions='H')

    evoked = read_evokeds(evoked_fname, condition='Left Auditory',
                          baseline=(-0.2, 0.0))
    fig = plot_alignment(evoked.info, trans_fname, subject='sample',
                         subjects_dir=subjects_dir)

    report.add_figs_to_section(figs=fig,  # test non-list input
                               captions='random image', scale=1.2)
    assert_true(repr(report))


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_mayavi
@traits_test
@requires_nibabel()
def test_render_mri():
    """Test rendering MRI for mne report."""
    tempdir = _TempDir()
    trans_fname_new = op.join(tempdir, 'temp-trans.fif')
    for a, b in [[trans_fname, trans_fname_new]]:
        shutil.copyfile(a, b)
    report = Report(info_fname=raw_fname,
                    subject='sample', subjects_dir=subjects_dir)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        report.parse_folder(data_path=tempdir, mri_decim=30, pattern='*',
                            n_jobs=2)
    report.save(op.join(tempdir, 'report.html'), open_browser=False)
    assert_true(repr(report))


@testing.requires_testing_data
@requires_nibabel()
def test_render_mri_without_bem():
    """Test rendering MRI without BEM for mne report."""
    tempdir = _TempDir()
    os.mkdir(op.join(tempdir, 'sample'))
    os.mkdir(op.join(tempdir, 'sample', 'mri'))
    shutil.copyfile(mri_fname, op.join(tempdir, 'sample', 'mri', 'T1.mgz'))
    report = Report(info_fname=raw_fname,
                    subject='sample', subjects_dir=tempdir)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        report.parse_folder(tempdir)
    assert_true(len(w) >= 1)
    report.save(op.join(tempdir, 'report.html'), open_browser=False)


@testing.requires_testing_data
@requires_nibabel()
def test_add_htmls_to_section():
    """Test adding html str to mne report."""
    report = Report(info_fname=raw_fname,
                    subject='sample', subjects_dir=subjects_dir)
    html = '<b>MNE-Python is AWESOME</b>'
    caption, section = 'html', 'html_section'
    report.add_htmls_to_section(html, caption, section)
    idx = report._sectionlabels.index('report_' + section)
    html_compare = report.html[idx]
    assert_true(html in html_compare)
    assert_true(repr(report))


def test_add_slider_to_section():
    """Test adding a slider with a series of images to mne report."""
    tempdir = _TempDir()
    from matplotlib import pyplot as plt
    report = Report(info_fname=raw_fname,
                    subject='sample', subjects_dir=subjects_dir)
    section = 'slider_section'
    figs = list()
    figs.append(plt.figure())
    plt.plot([1, 2, 3])
    plt.close('all')
    figs.append(plt.figure())
    plt.plot([3, 2, 1])
    plt.close('all')
    report.add_slider_to_section(figs, section=section)
    report.save(op.join(tempdir, 'report.html'), open_browser=False)

    assert_raises(NotImplementedError, report.add_slider_to_section,
                  [figs, figs])
    assert_raises(ValueError, report.add_slider_to_section, figs, ['wug'])
    assert_raises(TypeError, report.add_slider_to_section, figs, 'wug')
    # need at least 2
    assert_raises(ValueError, report.add_slider_to_section, figs[:1], 'wug')

    # Smoke test that SVG w/unicode can be added
    report = Report()
    fig, ax = plt.subplots()
    ax.set_xlabel(u'μ')
    report.add_slider_to_section([fig] * 2, image_format='svg')


def test_validate_input():
    """Test Report input validation."""
    report = Report()
    items = ['a', 'b', 'c']
    captions = ['Letter A', 'Letter B', 'Letter C']
    section = 'ABCs'
    comments = ['First letter of the alphabet.',
                'Second letter of the alphabet',
                'Third letter of the alphabet']
    assert_raises(ValueError, report._validate_input, items, captions[:-1],
                  section, comments=None)
    assert_raises(ValueError, report._validate_input, items, captions, section,
                  comments=comments[:-1])
    values = report._validate_input(items, captions, section, comments=None)
    items_new, captions_new, comments_new = values
    assert_equal(len(comments_new), len(items))


run_tests_if_main()
