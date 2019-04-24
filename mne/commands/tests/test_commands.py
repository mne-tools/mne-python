# -*- coding: utf-8 -*-
import os
from os import path as op
import shutil
import glob

import pytest
from numpy.testing import assert_equal, assert_allclose

from mne import concatenate_raws, read_bem_surfaces, read_surface
from mne.commands import (mne_browse_raw, mne_bti2fiff, mne_clean_eog_ecg,
                          mne_compute_proj_ecg, mne_compute_proj_eog,
                          mne_coreg, mne_kit2fiff,
                          mne_make_scalp_surfaces, mne_maxfilter,
                          mne_report, mne_surf2bem, mne_watershed_bem,
                          mne_compare_fiff, mne_flash_bem, mne_show_fiff,
                          mne_show_info)
from mne.datasets import testing, sample
from mne.io import read_raw_fif
from mne.utils import (run_tests_if_main, requires_mne,
                       requires_mayavi, requires_tvtk, requires_freesurfer,
                       traits_test, ArgvSetter, modified_env)

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')

subjects_dir = op.join(testing.data_path(download=False), 'subjects')


def check_usage(module, force_help=False):
    """Ensure we print usage."""
    args = ('--help',) if force_help else ()
    with ArgvSetter(args) as out:
        try:
            module.run()
        except SystemExit:
            pass
        assert 'Usage: ' in out.stdout.getvalue()


@pytest.mark.slowtest
def test_browse_raw():
    """Test mne browse_raw."""
    check_usage(mne_browse_raw)
    with ArgvSetter(('--raw', raw_fname)):
        with pytest.warns(None):  # mpl show warning sometimes
            mne_browse_raw.run()


def test_bti2fiff():
    """Test mne bti2fiff."""
    check_usage(mne_bti2fiff)


def test_compare_fiff():
    """Test mne compare_fiff."""
    check_usage(mne_compare_fiff)


def test_show_fiff():
    """Test mne compare_fiff."""
    check_usage(mne_show_fiff)
    with ArgvSetter((raw_fname,)):
        mne_show_fiff.run()
    with ArgvSetter((raw_fname, '--tag=102')):
        mne_show_fiff.run()


@requires_mne
def test_clean_eog_ecg(tmpdir):
    """Test mne clean_eog_ecg."""
    check_usage(mne_clean_eog_ecg)
    tempdir = str(tmpdir)
    raw = concatenate_raws([read_raw_fif(f)
                            for f in [raw_fname, raw_fname, raw_fname]])
    raw.info['bads'] = ['MEG 2443']
    use_fname = op.join(tempdir, op.basename(raw_fname))
    raw.save(use_fname)
    with ArgvSetter(('-i', use_fname, '--quiet')):
        mne_clean_eog_ecg.run()
    for key, count in (('proj', 2), ('-eve', 3)):
        fnames = glob.glob(op.join(tempdir, '*%s.fif' % key))
        assert len(fnames) == count


@pytest.mark.slowtest
@pytest.mark.parametrize('fun', (mne_compute_proj_ecg, mne_compute_proj_eog))
def test_compute_proj_exg(tmpdir, fun):
    """Test mne compute_proj_ecg/eog."""
    check_usage(fun)
    tempdir = str(tmpdir)
    use_fname = op.join(tempdir, op.basename(raw_fname))
    bad_fname = op.join(tempdir, 'bads.txt')
    with open(bad_fname, 'w') as fid:
        fid.write('MEG 2443\n')
    shutil.copyfile(raw_fname, use_fname)
    with ArgvSetter(('-i', use_fname, '--bad=' + bad_fname,
                     '--rej-eeg', '150')):
        with pytest.warns(None):  # samples, sometimes
            fun.run()
    fnames = glob.glob(op.join(tempdir, '*proj.fif'))
    assert len(fnames) == 1
    fnames = glob.glob(op.join(tempdir, '*-eve.fif'))
    assert len(fnames) == 1


def test_coreg():
    """Test mne coreg."""
    assert hasattr(mne_coreg, 'run')


def test_kit2fiff():
    """Test mne kit2fiff."""
    # Can't check
    check_usage(mne_kit2fiff, force_help=True)


@pytest.mark.slowtest  # slow on Travis OSX
@requires_tvtk
@testing.requires_testing_data
def test_make_scalp_surfaces(tmpdir):
    """Test mne make_scalp_surfaces."""
    check_usage(mne_make_scalp_surfaces)
    has = 'SUBJECTS_DIR' in os.environ
    # Copy necessary files to avoid FreeSurfer call
    tempdir = str(tmpdir)
    surf_path = op.join(subjects_dir, 'sample', 'surf')
    surf_path_new = op.join(tempdir, 'sample', 'surf')
    os.mkdir(op.join(tempdir, 'sample'))
    os.mkdir(surf_path_new)
    subj_dir = op.join(tempdir, 'sample', 'bem')
    os.mkdir(subj_dir)
    shutil.copy(op.join(surf_path, 'lh.seghead'), surf_path_new)

    cmd = ('-s', 'sample', '--subjects-dir', tempdir)
    with modified_env(**{'_MNE_TESTING_SCALP': 'true'}):
        dense_fname = op.join(subj_dir, 'sample-head-dense.fif')
        medium_fname = op.join(subj_dir, 'sample-head-medium.fif')
        with ArgvSetter(cmd, disable_stdout=False, disable_stderr=False):
            with modified_env(FREESURFER_HOME=None):
                pytest.raises(RuntimeError, mne_make_scalp_surfaces.run)
            with modified_env(FREESURFER_HOME=tempdir):
                mne_make_scalp_surfaces.run()
                assert op.isfile(dense_fname)
                assert op.isfile(medium_fname)
                with pytest.raises(IOError, match='overwrite'):
                    mne_make_scalp_surfaces.run()
    # actually check the outputs
    head_py = read_bem_surfaces(dense_fname)
    assert_equal(len(head_py), 1)
    head_py = head_py[0]
    head_c = read_bem_surfaces(op.join(subjects_dir, 'sample', 'bem',
                                       'sample-head-dense.fif'))[0]
    assert_allclose(head_py['rr'], head_c['rr'])
    if not has:
        assert 'SUBJECTS_DIR' not in os.environ


def test_maxfilter():
    """Test mne maxfilter."""
    check_usage(mne_maxfilter)
    with ArgvSetter(('-i', raw_fname, '--st', '--movecomp', '--linefreq', '60',
                     '--trans', raw_fname)) as out:
        with pytest.warns(RuntimeWarning, match="Don't use"):
            os.environ['_MNE_MAXFILTER_TEST'] = 'true'
            try:
                mne_maxfilter.run()
            finally:
                del os.environ['_MNE_MAXFILTER_TEST']
        for check in ('maxfilter', '-trans', '-movecomp'):
            assert check in out.stdout.getvalue(), check


@pytest.mark.slowtest
@requires_mayavi
@traits_test
@testing.requires_testing_data
def test_report(tmpdir):
    """Test mne report."""
    check_usage(mne_report)
    tempdir = str(tmpdir)
    use_fname = op.join(tempdir, op.basename(raw_fname))
    shutil.copyfile(raw_fname, use_fname)
    with ArgvSetter(('-p', tempdir, '-i', use_fname, '-d', subjects_dir,
                     '-s', 'sample', '--no-browser', '-m', '30')):
        with pytest.warns(None):  # contour levels
            mne_report.run()
    fnames = glob.glob(op.join(tempdir, '*.html'))
    assert len(fnames) == 1


def test_surf2bem():
    """Test mne surf2bem."""
    check_usage(mne_surf2bem)


@pytest.mark.timeout(600)  # took ~400 sec on a local test
@pytest.mark.slowtest
@pytest.mark.ultraslowtest
@requires_freesurfer
@testing.requires_testing_data
def test_watershed_bem(tmpdir):
    """Test mne watershed bem."""
    check_usage(mne_watershed_bem)
    # Copy necessary files to tempdir
    tempdir = str(tmpdir)
    mridata_path = op.join(subjects_dir, 'sample', 'mri')
    subject_path_new = op.join(tempdir, 'sample')
    mridata_path_new = op.join(subject_path_new, 'mri')
    os.mkdir(op.join(tempdir, 'sample'))
    os.mkdir(mridata_path_new)
    if op.exists(op.join(mridata_path, 'T1')):
        shutil.copytree(op.join(mridata_path, 'T1'), op.join(mridata_path_new,
                                                             'T1'))
    if op.exists(op.join(mridata_path, 'T1.mgz')):
        shutil.copyfile(op.join(mridata_path, 'T1.mgz'),
                        op.join(mridata_path_new, 'T1.mgz'))
    out_fnames = list()
    for kind in ('outer_skin', 'outer_skull', 'inner_skull'):
        out_fnames.append(op.join(subject_path_new, 'bem', 'inner_skull.surf'))
    assert not any(op.isfile(out_fname) for out_fname in out_fnames)
    with ArgvSetter(('-d', tempdir, '-s', 'sample', '-o'),
                    disable_stdout=False, disable_stderr=False):
        mne_watershed_bem.run()
    for out_fname in out_fnames:
        _, tris = read_surface(out_fname)
        assert len(tris) == 20480


@pytest.mark.timeout(300)  # took 200 sec locally
@pytest.mark.slowtest
@pytest.mark.ultraslowtest
@requires_freesurfer
@sample.requires_sample_data
def test_flash_bem(tmpdir):
    """Test mne flash_bem."""
    check_usage(mne_flash_bem, force_help=True)
    # Using the sample dataset
    subjects_dir = op.join(sample.data_path(download=False), 'subjects')
    # Copy necessary files to tempdir
    tempdir = str(tmpdir)
    mridata_path = op.join(subjects_dir, 'sample', 'mri')
    subject_path_new = op.join(tempdir, 'sample')
    mridata_path_new = op.join(subject_path_new, 'mri')
    os.makedirs(op.join(mridata_path_new, 'flash'))
    os.makedirs(op.join(subject_path_new, 'bem'))
    shutil.copyfile(op.join(mridata_path, 'T1.mgz'),
                    op.join(mridata_path_new, 'T1.mgz'))
    shutil.copyfile(op.join(mridata_path, 'brain.mgz'),
                    op.join(mridata_path_new, 'brain.mgz'))
    # Copy the available mri/flash/mef*.mgz files from the dataset
    flash_path = op.join(mridata_path_new, 'flash')
    for kind in (5, 30):
        in_fname = op.join(mridata_path, 'flash', 'mef%02d.mgz' % kind)
        shutil.copyfile(in_fname, op.join(flash_path, op.basename(in_fname)))
    # Test mne flash_bem with --noconvert option
    # (since there are no DICOM Flash images in dataset)
    out_fnames = list()
    for kind in ('outer_skin', 'outer_skull', 'inner_skull'):
        out_fnames.append(op.join(subject_path_new, 'bem', 'outer_skin.surf'))
    assert not any(op.isfile(out_fname) for out_fname in out_fnames)
    with ArgvSetter(('-d', tempdir, '-s', 'sample', '-n'),
                    disable_stdout=False, disable_stderr=False):
        mne_flash_bem.run()
    # do they exist and are expected size
    for out_fname in out_fnames:
        _, tris = read_surface(out_fname)
        assert len(tris) == 5120


def test_show_info():
    """Test mne show_info."""
    check_usage(mne_show_info)
    with ArgvSetter((raw_fname,)):
        mne_show_info.run()


run_tests_if_main()
