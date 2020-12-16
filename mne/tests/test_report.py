# -*- coding: utf-8 -*-
# Authors: Mainak Jas <mainak@neuro.hut.fi>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import copy
import glob
import os
import os.path as op
import shutil
import pathlib

import numpy as np
from numpy.testing import assert_equal
import pytest
from matplotlib import pyplot as plt

from mne import Epochs, read_events, read_evokeds
from mne.io import read_raw_fif
from mne.datasets import testing
from mne.report import Report, open_report, _ReportScraper
from mne.utils import (requires_nibabel, Bunch,
                       run_tests_if_main, requires_h5py)
from mne.viz import plot_alignment
from mne.io.write import DATE_NONE

data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
report_dir = op.join(data_dir, 'MEG', 'sample')
raw_fname = op.join(report_dir, 'sample_audvis_trunc_raw.fif')
ms_fname = op.join(data_dir, 'SSS', 'test_move_anon_raw.fif')
event_fname = op.join(report_dir, 'sample_audvis_trunc_raw-eve.fif')
cov_fname = op.join(report_dir, 'sample_audvis_trunc-cov.fif')
proj_fname = op.join(report_dir, 'sample_audvis_ecg-proj.fif')
fwd_fname = op.join(report_dir, 'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
trans_fname = op.join(report_dir, 'sample_audvis_trunc-trans.fif')
inv_fname = op.join(report_dir,
                    'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
mri_fname = op.join(subjects_dir, 'sample', 'mri', 'T1.mgz')

base_dir = op.realpath(op.join(op.dirname(__file__), '..', 'io', 'tests',
                               'data'))
evoked_fname = op.join(base_dir, 'test-ave.fif')


def _get_example_figures():
    """Create two example figures."""
    fig1 = plt.plot([1, 2], [1, 2])[0].figure
    fig2 = plt.plot([3, 4], [3, 4])[0].figure
    return [fig1, fig2]


@pytest.mark.slowtest
@testing.requires_testing_data
def test_render_report(renderer, tmpdir):
    """Test rendering -*.fif files for mne report."""
    tempdir = str(tmpdir)
    raw_fname_new = op.join(tempdir, 'temp_raw.fif')
    raw_fname_new_bids = op.join(tempdir, 'temp_meg.fif')
    ms_fname_new = op.join(tempdir, 'temp_ms_raw.fif')
    event_fname_new = op.join(tempdir, 'temp_raw-eve.fif')
    cov_fname_new = op.join(tempdir, 'temp_raw-cov.fif')
    proj_fname_new = op.join(tempdir, 'temp_ecg-proj.fif')
    fwd_fname_new = op.join(tempdir, 'temp_raw-fwd.fif')
    inv_fname_new = op.join(tempdir, 'temp_raw-inv.fif')
    for a, b in [[raw_fname, raw_fname_new],
                 [raw_fname, raw_fname_new_bids],
                 [ms_fname, ms_fname_new],
                 [event_fname, event_fname_new],
                 [cov_fname, cov_fname_new],
                 [proj_fname, proj_fname_new],
                 [fwd_fname, fwd_fname_new],
                 [inv_fname, inv_fname_new]]:
        shutil.copyfile(a, b)

    # create and add -epo.fif and -ave.fif files
    epochs_fname = op.join(tempdir, 'temp-epo.fif')
    evoked_fname = op.join(tempdir, 'temp-ave.fif')
    # Speed it up by picking channels
    raw = read_raw_fif(raw_fname_new, preload=True)
    raw.pick_channels(['MEG 0111', 'MEG 0121', 'EEG 001', 'EEG 002'])
    raw.del_proj()
    raw.set_eeg_reference(projection=True)
    epochs = Epochs(raw, read_events(event_fname), 1, -0.2, 0.2)
    epochs.save(epochs_fname, overwrite=True)
    # This can take forever (stall Travis), so let's make it fast
    # Also, make sure crop range is wide enough to avoid rendering bug
    evoked = epochs.average().crop(0.1, 0.2)
    evoked.save(evoked_fname)

    report = Report(info_fname=raw_fname_new, subjects_dir=subjects_dir,
                    projs=True)
    with pytest.warns(RuntimeWarning, match='Cannot render MRI'):
        report.parse_folder(data_path=tempdir, on_error='raise')
    assert repr(report)

    # Check correct paths and filenames
    fnames = glob.glob(op.join(tempdir, '*.fif'))
    for fname in fnames:
        assert (op.basename(fname) in
                [op.basename(x) for x in report.fnames])
        assert (''.join(report.html).find(op.basename(fname)) != -1)

    assert_equal(len(report.fnames), len(fnames))
    assert_equal(len(report.html), len(report.fnames))
    assert_equal(len(report.fnames), len(report))

    # Check saving functionality
    report.data_path = tempdir
    fname = op.join(tempdir, 'report.html')
    report.save(fname=fname, open_browser=False)
    assert (op.isfile(fname))
    with open(fname, 'rb') as fid:
        html = fid.read().decode('utf-8')
    assert '(MaxShield on)' in html
    # Projectors in Raw.info
    assert '<h4>SSP Projectors</h4>' in html
    # Projectors in `proj_fname_new`
    assert f'SSP Projectors: {op.basename(proj_fname_new)}' in html
    # Evoked in `evoked_fname`
    assert f'Evoked: {op.basename(evoked_fname)} ({evoked.comment})' in html
    assert 'Topomap (ch_type =' in html
    assert f'Evoked: {op.basename(evoked_fname)} (GFPs)' in html

    assert_equal(len(report.html), len(fnames))
    assert_equal(len(report.html), len(report.fnames))

    # Check saving same report to new filename
    report.save(fname=op.join(tempdir, 'report2.html'), open_browser=False)
    assert (op.isfile(op.join(tempdir, 'report2.html')))

    # Check overwriting file
    report.save(fname=op.join(tempdir, 'report.html'), open_browser=False,
                overwrite=True)
    assert (op.isfile(op.join(tempdir, 'report.html')))

    # Check pattern matching with multiple patterns
    pattern = ['*raw.fif', '*eve.fif']
    with pytest.warns(RuntimeWarning, match='Cannot render MRI'):
        report.parse_folder(data_path=tempdir, pattern=pattern)
    assert (repr(report))

    fnames = glob.glob(op.join(tempdir, '*.raw')) + \
        glob.glob(op.join(tempdir, '*.raw'))
    for fname in fnames:
        assert (op.basename(fname) in
                [op.basename(x) for x in report.fnames])
        assert (''.join(report.html).find(op.basename(fname)) != -1)

    pytest.raises(ValueError, Report, image_format='foo')
    pytest.raises(ValueError, Report, image_format=None)

    # SVG rendering
    report = Report(info_fname=raw_fname_new, subjects_dir=subjects_dir,
                    image_format='svg')
    tempdir = pathlib.Path(tempdir)  # test using pathlib.Path
    with pytest.warns(RuntimeWarning, match='Cannot render MRI'):
        report.parse_folder(data_path=tempdir, on_error='raise')

    # ndarray support smoke test
    report.add_figs_to_section(np.zeros((2, 3, 3)), 'caption', 'section')

    with pytest.raises(TypeError, match='figure must be a'):
        report.add_figs_to_section('foo', 'caption', 'section')
    with pytest.raises(TypeError, match='figure must be a'):
        report.add_figs_to_section(['foo'], 'caption', 'section')


@testing.requires_testing_data
def test_report_raw_psd_and_date(tmpdir):
    """Test report raw PSD and DATE_NONE functionality."""
    with pytest.raises(TypeError, match='dict'):
        Report(raw_psd='foo')

    tempdir = str(tmpdir)
    raw = read_raw_fif(raw_fname).crop(0, 1.).load_data()
    raw_fname_new = op.join(tempdir, 'temp_raw.fif')
    raw.save(raw_fname_new)
    report = Report(raw_psd=True)
    report.parse_folder(data_path=tempdir, render_bem=False,
                        on_error='raise')
    assert isinstance(report.html, list)
    assert 'PSD' in ''.join(report.html)
    assert 'GMT' in ''.join(report.html)

    # test new anonymize functionality
    report = Report()
    raw.anonymize()
    raw.save(raw_fname_new, overwrite=True)
    report.parse_folder(data_path=tempdir, render_bem=False,
                        on_error='raise')
    assert isinstance(report.html, list)
    assert 'GMT' in ''.join(report.html)

    # DATE_NONE functionality
    report = Report()
    # old style (pre 0.20) date anonymization
    raw.info['meas_date'] = None
    for key in ('file_id', 'meas_id'):
        value = raw.info.get(key)
        if value is not None:
            assert 'msecs' not in value
            value['secs'] = DATE_NONE[0]
            value['usecs'] = DATE_NONE[1]
    raw.save(raw_fname_new, overwrite=True)
    report.parse_folder(data_path=tempdir, render_bem=False,
                        on_error='raise')
    assert isinstance(report.html, list)
    assert 'GMT' not in ''.join(report.html)


@testing.requires_testing_data
def test_render_add_sections(renderer, tmpdir):
    """Test adding figures/images to section."""
    tempdir = str(tmpdir)
    report = Report(subjects_dir=subjects_dir)
    # Check add_figs_to_section functionality
    fig = plt.plot([1, 2], [1, 2])[0].figure
    report.add_figs_to_section(figs=fig,  # test non-list input
                               captions=['evoked response'], scale=1.2,
                               image_format='svg')
    pytest.raises(ValueError, report.add_figs_to_section, figs=[fig, fig],
                  captions='H')
    pytest.raises(ValueError, report.add_figs_to_section, figs=fig,
                  captions=['foo'], scale=0, image_format='svg')
    pytest.raises(ValueError, report.add_figs_to_section, figs=fig,
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

    pytest.raises(ValueError, report.add_images_to_section,
                  fnames=[img_fname, img_fname], captions='H')

    pytest.raises(ValueError, report.add_images_to_section,
                  fnames=['foobar.xxx'], captions='H')

    evoked = read_evokeds(evoked_fname, condition='Left Auditory',
                          baseline=(-0.2, 0.0))
    fig = plot_alignment(evoked.info, trans_fname, subject='sample',
                         subjects_dir=subjects_dir)

    report.add_figs_to_section(figs=fig,  # test non-list input
                               captions='random image', scale=1.2)
    assert (repr(report))
    fname = op.join(str(tmpdir), 'test.html')
    report.save(fname, open_browser=False)
    with open(fname, 'r') as fid:
        html = fid.read()
    assert html.count('<li class="report_custom"') == 8  # several


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_nibabel()
def test_render_mri(renderer, tmpdir):
    """Test rendering MRI for mne report."""
    tempdir = str(tmpdir)
    trans_fname_new = op.join(tempdir, 'temp-trans.fif')
    for a, b in [[trans_fname, trans_fname_new]]:
        shutil.copyfile(a, b)
    report = Report(info_fname=raw_fname,
                    subject='sample', subjects_dir=subjects_dir)
    report.parse_folder(data_path=tempdir, mri_decim=30, pattern='*')
    fname = op.join(tempdir, 'report.html')
    report.save(fname, open_browser=False)
    with open(fname, 'r') as fid:
        html = fid.read()
    assert html.count('<li class="bem"') == 2  # left and content
    assert repr(report)
    report.add_bem_to_section('sample', caption='extra', section='foo',
                              subjects_dir=subjects_dir, decim=30)
    report.save(fname, open_browser=False, overwrite=True)
    with open(fname, 'r') as fid:
        html = fid.read()
    assert 'report_report' not in html
    assert html.count('<li class="report_foo"') == 2


@testing.requires_testing_data
@requires_nibabel()
def test_render_mri_without_bem(tmpdir):
    """Test rendering MRI without BEM for mne report."""
    tempdir = str(tmpdir)
    os.mkdir(op.join(tempdir, 'sample'))
    os.mkdir(op.join(tempdir, 'sample', 'mri'))
    shutil.copyfile(mri_fname, op.join(tempdir, 'sample', 'mri', 'T1.mgz'))
    report = Report(info_fname=raw_fname,
                    subject='sample', subjects_dir=tempdir)
    with pytest.raises(RuntimeError, match='No matching files found'):
        report.parse_folder(tempdir, render_bem=False)
    with pytest.warns(RuntimeWarning, match='No BEM surfaces found'):
        report.parse_folder(tempdir, render_bem=True, mri_decim=20)
    assert 'bem' in report.fnames
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
    assert (html in html_compare)
    assert (repr(report))


def test_add_slider_to_section(tmpdir):
    """Test adding a slider with a series of images to mne report."""
    tempdir = str(tmpdir)
    report = Report(info_fname=raw_fname,
                    subject='sample', subjects_dir=subjects_dir)
    section = 'slider_section'
    figs = _get_example_figures()
    report.add_slider_to_section(figs, section=section, title='my title')
    assert report.fnames[0] == 'my title-#-report_slider_section-#-custom'
    report.save(op.join(tempdir, 'report.html'), open_browser=False)

    pytest.raises(NotImplementedError, report.add_slider_to_section,
                  [figs, figs])
    pytest.raises(ValueError, report.add_slider_to_section, figs, ['wug'])
    pytest.raises(TypeError, report.add_slider_to_section, figs, 'wug')
    # need at least 2
    pytest.raises(ValueError, report.add_slider_to_section, figs[:1], 'wug')

    # Smoke test that SVG w/unicode can be added
    report = Report()
    fig, ax = plt.subplots()
    ax.set_xlabel('µ')
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
    pytest.raises(ValueError, report._validate_input, items, captions[:-1],
                  section, comments=None)
    pytest.raises(ValueError, report._validate_input, items, captions, section,
                  comments=comments[:-1])
    values = report._validate_input(items, captions, section, comments=None)
    items_new, captions_new, comments_new = values
    assert_equal(len(comments_new), len(items))


@requires_h5py
def test_open_report(tmpdir):
    """Test the open_report function."""
    tempdir = str(tmpdir)
    hdf5 = op.join(tempdir, 'report.h5')

    # Test creating a new report through the open_report function
    fig1 = _get_example_figures()[0]
    with open_report(hdf5, subjects_dir=subjects_dir) as report:
        assert report.subjects_dir == subjects_dir
        assert report._fname == hdf5
        report.add_figs_to_section(figs=fig1, captions=['evoked response'])
    # Exiting the context block should have triggered saving to HDF5
    assert op.exists(hdf5)

    # Load the HDF5 version of the report and check equivalence
    report2 = open_report(hdf5)
    assert report2._fname == hdf5
    assert report2.subjects_dir == report.subjects_dir
    assert report2.html == report.html
    assert report2.__getstate__() == report.__getstate__()
    assert '_fname' not in report2.__getstate__()

    # Check parameters when loading a report
    pytest.raises(ValueError, open_report, hdf5, foo='bar')  # non-existing
    pytest.raises(ValueError, open_report, hdf5, subjects_dir='foo')
    open_report(hdf5, subjects_dir=subjects_dir)  # This should work

    # Check that the context manager doesn't swallow exceptions
    with pytest.raises(ZeroDivisionError):
        with open_report(hdf5, subjects_dir=subjects_dir) as report:
            1 / 0


def test_remove():
    """Test removing figures from a report."""
    r = Report()
    fig1, fig2 = _get_example_figures()
    r.add_figs_to_section(fig1, 'figure1', 'mysection')
    r.add_slider_to_section([fig1, fig2], title='figure1',
                            section='othersection')
    r.add_figs_to_section(fig2, 'figure1', 'mysection')
    r.add_figs_to_section(fig2, 'figure2', 'mysection')

    # Test removal by caption
    r2 = copy.deepcopy(r)
    removed_index = r2.remove(caption='figure1')
    assert removed_index == 2
    assert len(r2.html) == 3
    assert r2.html[0] == r.html[0]
    assert r2.html[1] == r.html[1]
    assert r2.html[2] == r.html[3]

    # Test restricting to section
    r2 = copy.deepcopy(r)
    removed_index = r2.remove(caption='figure1', section='othersection')
    assert removed_index == 1
    assert len(r2.html) == 3
    assert r2.html[0] == r.html[0]
    assert r2.html[1] == r.html[2]
    assert r2.html[2] == r.html[3]

    # Test removal of empty sections
    r2 = copy.deepcopy(r)
    r2.remove(caption='figure1', section='othersection')
    assert r2.sections == ['mysection']
    assert r2._sectionvars == {'mysection': 'report_mysection'}


def test_add_or_replace():
    """Test replacing existing figures in a report."""
    r = Report()
    fig1, fig2 = _get_example_figures()
    r.add_figs_to_section(fig1, 'duplicate', 'mysection')
    r.add_figs_to_section(fig1, 'duplicate', 'mysection')
    r.add_figs_to_section(fig1, 'duplicate', 'othersection')
    r.add_figs_to_section(fig2, 'nonduplicate', 'mysection')
    # By default, replace=False, so all figures should be there
    assert len(r.html) == 4

    old_r = copy.deepcopy(r)

    # Re-add fig1 with replace=True, it should overwrite the last occurrence of
    # fig1 in section 'mysection'.
    r.add_figs_to_section(fig2, 'duplicate', 'mysection', replace=True)
    assert len(r.html) == 4
    assert r.html[1] != old_r.html[1]  # This figure should have changed
    # All other figures should be the same
    assert r.html[0] == old_r.html[0]
    assert r.html[2] == old_r.html[2]
    assert r.html[3] == old_r.html[3]


def test_scraper(tmpdir):
    """Test report scraping."""
    r = Report()
    fig1, fig2 = _get_example_figures()
    r.add_figs_to_section(fig1, 'a', 'mysection')
    r.add_figs_to_section(fig2, 'b', 'mysection')
    # Mock a Sphinx + sphinx_gallery config
    app = Bunch(builder=Bunch(srcdir=str(tmpdir),
                              outdir=op.join(str(tmpdir), '_build', 'html')))
    scraper = _ReportScraper()
    scraper.app = app
    gallery_conf = dict(src_dir=app.builder.srcdir, builder_name='html')
    img_fname = op.join(app.builder.srcdir, 'auto_examples', 'images',
                        'sg_img.png')
    target_file = op.join(app.builder.srcdir, 'auto_examples', 'sg.py')
    os.makedirs(op.dirname(img_fname))
    os.makedirs(app.builder.outdir)
    block_vars = dict(image_path_iterator=(img for img in [img_fname]),
                      example_globals=dict(a=1), target_file=target_file)
    # Nothing yet
    block = None
    rst = scraper(block, block_vars, gallery_conf)
    assert rst == ''
    # Still nothing
    block_vars['example_globals']['r'] = r
    rst = scraper(block, block_vars, gallery_conf)
    # Once it's saved, add it
    assert rst == ''
    fname = op.join(str(tmpdir), 'my_html.html')
    r.save(fname, open_browser=False)
    rst = scraper(block, block_vars, gallery_conf)
    out_html = op.join(app.builder.outdir, 'auto_examples', 'my_html.html')
    assert not op.isfile(out_html)
    os.makedirs(op.join(app.builder.outdir, 'auto_examples'))
    scraper.copyfiles()
    assert op.isfile(out_html)
    assert rst.count('"') == 6
    assert "<iframe" in rst
    assert op.isfile(img_fname.replace('png', 'svg'))


@testing.requires_testing_data
@pytest.mark.parametrize('split_naming', ('neuromag', 'bids',))
def test_split_files(tmpdir, split_naming):
    """Test that in the case of split files, we only parse the first."""
    raw = read_raw_fif(raw_fname)
    split_size = '7MB'  # Should produce 3 files
    buffer_size_sec = 1  # Tiny buffer so it's smaller than the split size
    raw.save(op.join(tmpdir, 'raw_meg.fif'), split_size=split_size,
             split_naming=split_naming, buffer_size_sec=buffer_size_sec)

    report = Report()
    report.parse_folder(tmpdir, render_bem=False)
    assert len(report.fnames) == 1


run_tests_if_main()
