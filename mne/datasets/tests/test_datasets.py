import os
from os import path as op
import shutil
import zipfile
import sys

import pytest

from mne import datasets
from mne.datasets import testing
from mne.datasets._fsaverage.base import _set_montage_coreg_path
from mne.datasets.utils import _manifest_check_download

from mne.utils import (run_tests_if_main, requires_good_network, modified_env,
                       get_subjects_dir, ArgvSetter, _pl, use_log_level,
                       catch_logging)


subjects_dir = op.join(testing.data_path(download=False), 'subjects')


def test_datasets_basic(tmpdir):
    """Test simple dataset functions."""
    # XXX 'hf_sef' and 'misc' do not conform to these standards
    for dname in ('sample', 'somato', 'spm_face', 'testing', 'opm',
                  'bst_raw', 'bst_auditory', 'bst_resting', 'multimodal',
                  'bst_phantom_ctf', 'bst_phantom_elekta', 'kiloword',
                  'mtrf', 'phantom_4dbti',
                  'visual_92_categories', 'fieldtrip_cmc'):
        if dname.startswith('bst'):
            dataset = getattr(datasets.brainstorm, dname)
            check_name = 'brainstorm.%s' % (dname,)
        else:
            dataset = getattr(datasets, dname)
            check_name = dname
        if dataset.data_path(download=False) != '':
            assert isinstance(dataset.get_version(), str)
            assert datasets.utils.has_dataset(check_name)
        else:
            assert dataset.get_version() is None
            assert not datasets.utils.has_dataset(check_name)
        print('%s: %s' % (dname, datasets.utils.has_dataset(check_name)))
    tempdir = str(tmpdir)
    # don't let it read from the config file to get the directory,
    # force it to look for the default
    with modified_env(**{'_MNE_FAKE_HOME_DIR': tempdir, 'SUBJECTS_DIR': None}):
        assert (datasets.utils._get_path(None, 'foo', 'bar') ==
                op.join(tempdir, 'mne_data'))
        assert get_subjects_dir(None) is None
        _set_montage_coreg_path()
        sd = get_subjects_dir()
        assert sd.endswith('MNE-fsaverage-data')


def _fake_fetch_file(url, destination, print_destination=False):
    with open(destination, 'w') as fid:
        fid.write(url)


@requires_good_network
def test_downloads(tmpdir):
    """Test dataset URL handling."""
    # Try actually downloading a dataset
    path = datasets._fake.data_path(path=str(tmpdir), update_path=False)
    assert op.isfile(op.join(path, 'bar'))
    assert datasets._fake.get_version() is None


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_good_network
def test_fetch_parcellations(tmpdir):
    """Test fetching parcellations."""
    this_subjects_dir = str(tmpdir)
    os.mkdir(op.join(this_subjects_dir, 'fsaverage'))
    os.mkdir(op.join(this_subjects_dir, 'fsaverage', 'label'))
    os.mkdir(op.join(this_subjects_dir, 'fsaverage', 'surf'))
    for hemi in ('lh', 'rh'):
        shutil.copyfile(
            op.join(subjects_dir, 'fsaverage', 'surf', '%s.white' % hemi),
            op.join(this_subjects_dir, 'fsaverage', 'surf', '%s.white' % hemi))
    # speed up by prenteding we have one of them
    with open(op.join(this_subjects_dir, 'fsaverage', 'label',
                      'lh.aparc_sub.annot'), 'wb'):
        pass
    datasets.fetch_aparc_sub_parcellation(subjects_dir=this_subjects_dir)
    with ArgvSetter(('--accept-hcpmmp-license',)):
        datasets.fetch_hcp_mmp_parcellation(subjects_dir=this_subjects_dir)
    for hemi in ('lh', 'rh'):
        assert op.isfile(op.join(this_subjects_dir, 'fsaverage', 'label',
                                 '%s.aparc_sub.annot' % hemi))


_zip_fnames = ['foo/foo.txt', 'foo/bar.txt', 'foo/baz.txt']


def _fake_zip_fetch(url, fname, hash_):
    with zipfile.ZipFile(fname, 'w') as zipf:
        with zipf.open('foo/', 'w'):
            pass
        for fname in _zip_fnames:
            with zipf.open(fname, 'w'):
                pass


@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="writing zip files requires python3.6 or higher")
@pytest.mark.parametrize('n_have', range(len(_zip_fnames)))
def test_manifest_check_download(tmpdir, n_have, monkeypatch):
    """Test our manifest downloader."""
    monkeypatch.setattr(datasets.utils, '_fetch_file', _fake_zip_fetch)
    destination = op.join(str(tmpdir), 'empty')
    manifest_path = op.join(str(tmpdir), 'manifest.txt')
    with open(manifest_path, 'w') as fid:
        for fname in _zip_fnames:
            fid.write('%s\n' % fname)
    assert n_have in range(len(_zip_fnames) + 1)
    assert not op.isdir(destination)
    if n_have > 0:
        os.makedirs(op.join(destination, 'foo'))
        assert op.isdir(op.join(destination, 'foo'))
    for fname in _zip_fnames:
        assert not op.isfile(op.join(destination, fname))
    for fname in _zip_fnames[:n_have]:
        with open(op.join(destination, fname), 'w'):
            pass
    with catch_logging() as log:
        with use_log_level(True):
            url = hash_ = ''  # we mock the _fetch_file so these are not used
            _manifest_check_download(manifest_path, destination, url, hash_)
    log = log.getvalue()
    n_missing = 3 - n_have
    assert ('%d file%s missing from' % (n_missing, _pl(n_missing))) in log
    for want in ('Extracting missing', 'Successfully '):
        if n_missing > 0:
            assert want in log
        else:
            assert want not in log
    assert op.isdir(destination)
    for fname in _zip_fnames:
        assert op.isfile(op.join(destination, fname))


run_tests_if_main()
