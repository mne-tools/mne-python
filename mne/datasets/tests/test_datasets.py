from functools import partial
import os
from os import path as op
import re
import shutil
import zipfile

import pooch
import pytest

from mne import datasets, read_labels_from_annot, write_labels_to_annot
from mne.datasets import (testing, fetch_infant_template, fetch_phantom,
                          fetch_dataset)
from mne.datasets._fsaverage.base import _set_montage_coreg_path
from mne.datasets._infant import base as infant_base
from mne.datasets._phantom import base as phantom_base
from mne.datasets.utils import _manifest_check_download

from mne.utils import (requires_good_network,
                       get_subjects_dir, ArgvSetter, _pl, use_log_level,
                       catch_logging, hashfunc)


subjects_dir = testing.data_path(download=False) / 'subjects'


def test_datasets_basic(tmp_path, monkeypatch):
    """Test simple dataset functions."""
    # XXX 'hf_sef' and 'misc' do not conform to these standards
    for dname in ('sample', 'somato', 'spm_face', 'testing', 'opm',
                  'bst_raw', 'bst_auditory', 'bst_resting', 'multimodal',
                  'bst_phantom_ctf', 'bst_phantom_elekta', 'kiloword',
                  'mtrf', 'phantom_4dbti',
                  'visual_92_categories', 'fieldtrip_cmc'):
        if dname.startswith('bst'):
            dataset = getattr(datasets.brainstorm, dname)
        else:
            dataset = getattr(datasets, dname)
        if str(dataset.data_path(download=False)) != '.':
            assert isinstance(dataset.get_version(), str)
            assert datasets.has_dataset(dname)
        else:
            assert dataset.get_version() is None
            assert not datasets.has_dataset(dname)
        print('%s: %s' % (dname, datasets.has_dataset(dname)))
    tempdir = str(tmp_path)
    # Explicitly test one that isn't preset (given the config)
    monkeypatch.setenv('MNE_DATASETS_SAMPLE_PATH', tempdir)
    dataset = datasets.sample
    assert str(dataset.data_path(download=False)) == '.'
    assert dataset.get_version() != ''
    assert dataset.get_version() is None
    # don't let it read from the config file to get the directory,
    # force it to look for the default
    monkeypatch.setenv('_MNE_FAKE_HOME_DIR', tempdir)
    monkeypatch.delenv('SUBJECTS_DIR', raising=False)
    assert (str(datasets.utils._get_path(None, 'foo', 'bar')) ==
            op.join(tempdir, 'mne_data'))
    assert get_subjects_dir(None) is None
    _set_montage_coreg_path()
    sd = get_subjects_dir()
    assert sd.endswith('MNE-fsaverage-data')
    monkeypatch.setenv('MNE_DATA', str(tmp_path / 'foo'))
    with pytest.raises(FileNotFoundError, match='as specified by MNE_DAT'):
        testing.data_path(download=False)


@requires_good_network
def test_downloads(tmp_path, monkeypatch, capsys):
    """Test dataset URL and version handling."""
    # Try actually downloading a dataset
    kwargs = dict(path=str(tmp_path), verbose=True)
    # XXX we shouldn't need to disable capsys here, but there's a pytest bug
    # that we're hitting (https://github.com/pytest-dev/pytest/issues/5997)
    # now that we use pooch
    with capsys.disabled():
        with pytest.raises(RuntimeError, match='Do not download .* in tests'):
            path = datasets._fake.data_path(update_path=False, **kwargs)
        monkeypatch.setattr(
            datasets.utils,
            '_MODULES_TO_ENSURE_DOWNLOAD_IS_FALSE_IN_TESTS', ())
        path = datasets._fake.data_path(update_path=False, **kwargs)
    assert op.isdir(path)
    assert op.isfile(op.join(path, 'bar'))
    assert not datasets.has_dataset('fake')  # not in the desired path
    assert datasets._fake.get_version() is None
    assert datasets.utils._get_version('fake') is None
    monkeypatch.setenv('_MNE_FAKE_HOME_DIR', str(tmp_path))
    with pytest.warns(RuntimeWarning, match='non-standard config'):
        new_path = datasets._fake.data_path(update_path=True, **kwargs)
    assert path == new_path
    out, _ = capsys.readouterr()
    assert 'Downloading' not in out
    # No version: shown as existing but unknown version
    assert datasets.has_dataset('fake')
    # XXX logic bug, should be "unknown"
    assert datasets._fake.get_version() == '0.0'
    # With a version but no required one: shown as existing and gives version
    fname = tmp_path / 'foo' / 'version.txt'
    with open(fname, 'w') as fid:
        fid.write('0.1')
    assert datasets.has_dataset('fake')
    assert datasets._fake.get_version() == '0.1'
    datasets._fake.data_path(download=False, **kwargs)
    out, _ = capsys.readouterr()
    assert 'out of date' not in out
    # With the required version: shown as existing with the required version
    monkeypatch.setattr(datasets._fetch, '_FAKE_VERSION', '0.1')
    assert datasets.has_dataset('fake')
    assert datasets._fake.get_version() == '0.1'
    datasets._fake.data_path(download=False, **kwargs)
    out, _ = capsys.readouterr()
    assert 'out of date' not in out
    monkeypatch.setattr(datasets._fetch, '_FAKE_VERSION', '0.2')
    # With an older version:
    # 1. Marked as not actually being present
    assert not datasets.has_dataset('fake')
    # 2. Will try to update when `data_path` gets called, with logged message
    want_msg = 'Correctly trying to download newer version'

    def _error_download(self, fname, downloader, processor):
        url = self.get_url(fname)
        full_path = self.abspath / fname
        assert 'foo.tgz' in url
        assert str(tmp_path) in str(full_path)
        raise RuntimeError(want_msg)

    monkeypatch.setattr(pooch.Pooch, 'fetch', _error_download)
    with pytest.raises(RuntimeError, match=want_msg):
        datasets._fake.data_path(**kwargs)
    out, _ = capsys.readouterr()
    assert re.match(r'.* 0\.1 .*out of date.* 0\.2.*', out, re.MULTILINE), out


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_good_network
def test_fetch_parcellations(tmp_path):
    """Test fetching parcellations."""
    this_subjects_dir = str(tmp_path)
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
    # test our annot round-trips here
    kwargs = dict(subject='fsaverage', hemi='both', sort=False,
                  subjects_dir=this_subjects_dir)
    labels = read_labels_from_annot(parc='HCPMMP1', **kwargs)
    write_labels_to_annot(
        labels, parc='HCPMMP1_round',
        table_name='./left.fsaverage164.label.gii', **kwargs)
    orig = op.join(this_subjects_dir, 'fsaverage', 'label', 'lh.HCPMMP1.annot')
    first = hashfunc(orig)
    new = orig[:-6] + '_round.annot'
    second = hashfunc(new)
    assert first == second


_zip_fnames = ['foo/foo.txt', 'foo/bar.txt', 'foo/baz.txt']


def _fake_zip_fetch(url, path, fname, known_hash):
    fname = op.join(path, fname)
    with zipfile.ZipFile(fname, 'w') as zipf:
        with zipf.open('foo/', 'w'):
            pass
        for fname in _zip_fnames:
            with zipf.open(fname, 'w'):
                pass


@pytest.mark.parametrize('n_have', range(len(_zip_fnames)))
def test_manifest_check_download(tmp_path, n_have, monkeypatch):
    """Test our manifest downloader."""
    monkeypatch.setattr(pooch, 'retrieve', _fake_zip_fetch)
    destination = op.join(str(tmp_path), 'empty')
    manifest_path = op.join(str(tmp_path), 'manifest.txt')
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
            # we mock the pooch.retrieve so these are not used
            url = hash_ = ''
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


def _fake_mcd(manifest_path, destination, url, hash_, name=None,
              fake_files=False):
    if name is None:
        name = url.split('/')[-1].split('.')[0]
        assert name in url
        assert name in destination
    assert name in manifest_path
    assert len(hash_) == 32
    if fake_files:
        with open(manifest_path) as fid:
            for path in fid:
                path = path.strip()
                if not path:
                    continue
                fname = op.join(destination, path)
                os.makedirs(op.dirname(fname), exist_ok=True)
                with open(fname, 'wb'):
                    pass


def test_infant(tmp_path, monkeypatch):
    """Test fetch_infant_template."""
    monkeypatch.setattr(infant_base, '_manifest_check_download', _fake_mcd)
    fetch_infant_template('12mo', subjects_dir=tmp_path)
    with pytest.raises(ValueError, match='Invalid value for'):
        fetch_infant_template('0mo', subjects_dir=tmp_path)


def test_phantom(tmp_path, monkeypatch):
    """Test phantom data downloading."""
    # The Otaniemi file is only ~6MB, so in principle maybe we could test
    # an actual download here. But it doesn't seem worth it given that
    # CircleCI will at least test the VectorView one, and this file should
    # not change often.
    monkeypatch.setattr(phantom_base, '_manifest_check_download',
                        partial(_fake_mcd, name='phantom_otaniemi',
                                fake_files=True))
    fetch_phantom('otaniemi', subjects_dir=tmp_path)
    assert op.isfile(tmp_path / 'phantom_otaniemi' / 'mri' / 'T1.mgz')


def test_fetch_uncompressed_file(tmp_path):
    """Test downloading an uncompressed file with our fetch function."""
    dataset_dict = dict(
        dataset_name='license',
        url=('https://raw.githubusercontent.com/mne-tools/mne-python/main/'
             'LICENSE.txt'),
        archive_name='LICENSE.foo',
        folder_name=op.join(tmp_path, 'foo'),
        hash=None)
    fetch_dataset(dataset_dict, path=None, force_update=True)
    assert (tmp_path / 'foo' / 'LICENSE.foo').is_file()
