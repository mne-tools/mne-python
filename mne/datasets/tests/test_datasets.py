import os
from os import path as op
import re
import shutil
import zipfile
import sys

import pytest

from mne import datasets, read_labels_from_annot, write_labels_to_annot
from mne.datasets import testing, fetch_infant_template
from mne.datasets._infant import base as infant_base
from mne.datasets._fsaverage.base import _set_montage_coreg_path
from mne.datasets.utils import _manifest_check_download

from mne.utils import (requires_good_network,
                       get_subjects_dir, ArgvSetter, _pl, use_log_level,
                       catch_logging, hashfunc)


subjects_dir = op.join(testing.data_path(download=False), 'subjects')


def test_datasets_basic(tmpdir, monkeypatch):
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
    monkeypatch.setenv('_MNE_FAKE_HOME_DIR', tempdir)
    monkeypatch.delenv('SUBJECTS_DIR', raising=False)
    assert (datasets.utils._get_path(None, 'foo', 'bar') ==
            op.join(tempdir, 'mne_data'))
    assert get_subjects_dir(None) is None
    _set_montage_coreg_path()
    sd = get_subjects_dir()
    assert sd.endswith('MNE-fsaverage-data')
    monkeypatch.setenv('MNE_DATA', str(tmpdir.join('foo')))
    with pytest.raises(FileNotFoundError, match='as specified by MNE_DAT'):
        testing.data_path(download=False)


@requires_good_network
def test_downloads(tmpdir, monkeypatch, capsys):
    """Test dataset URL and version handling."""
    # Try actually downloading a dataset
    kwargs = dict(path=str(tmpdir), verbose=True)
    path = datasets._fake.data_path(update_path=False, **kwargs)
    out, _ = capsys.readouterr()
    assert 'Downloading' in out
    assert op.isdir(path)
    assert op.isfile(op.join(path, 'bar'))
    assert not datasets.utils.has_dataset('fake')  # not in the desired path
    assert datasets._fake.get_version() is None
    assert datasets.utils._get_version('fake') is None
    monkeypatch.setenv('_MNE_FAKE_HOME_DIR', str(tmpdir))
    with pytest.warns(RuntimeWarning, match='non-standard config'):
        new_path = datasets._fake.data_path(update_path=True, **kwargs)
    assert path == new_path
    out, _ = capsys.readouterr()
    assert 'Downloading' not in out
    # No version: shown as existing but unknown version
    assert datasets.utils.has_dataset('fake')
    # XXX logic bug, should be "unknown"
    assert datasets._fake.get_version() == '0.7'
    # With a version but no required one: shown as existing and gives version
    fname = tmpdir / 'foo' / 'version.txt'
    with open(fname, 'w') as fid:
        fid.write('0.1')
    assert datasets.utils.has_dataset('fake')
    assert datasets._fake.get_version() == '0.1'
    datasets._fake.data_path(download=False, **kwargs)
    out, _ = capsys.readouterr()
    assert 'out of date' not in out
    # With the required version: shown as existing with the required version
    monkeypatch.setattr(datasets.utils, '_FAKE_VERSION', '0.1')
    assert datasets.utils.has_dataset('fake')
    assert datasets._fake.get_version() == '0.1'
    datasets._fake.data_path(download=False, **kwargs)
    out, _ = capsys.readouterr()
    assert 'out of date' not in out
    monkeypatch.setattr(datasets.utils, '_FAKE_VERSION', '0.2')
    # With an older version:
    # 1. Marked as not actually being present
    assert not datasets.utils.has_dataset('fake')
    # 2. Will try to update when `data_path` gets called, with logged message
    want_msg = 'Correctly trying to download newer version'

    def _error_download(url, full_name, print_destination, hash_, hash_type):
        assert 'foo.tgz' in url
        assert str(tmpdir) in full_name
        raise RuntimeError(want_msg)

    monkeypatch.setattr(datasets.utils, '_fetch_file', _error_download)
    with pytest.raises(RuntimeError, match=want_msg):
        datasets._fake.data_path(**kwargs)
    out, _ = capsys.readouterr()
    assert re.match(r'.* 0\.1 .*out of date.* 0\.2.*', out, re.MULTILINE), out


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


def _fake_mcd(manifest_path, destination, url, hash_):
    name = url.split('/')[-1].split('.')[0]
    assert name in manifest_path
    assert name in destination
    assert name in url
    assert len(hash_) == 32


def test_infant(tmpdir, monkeypatch):
    """Test fetch_infant_template."""
    monkeypatch.setattr(infant_base, '_manifest_check_download', _fake_mcd)
    fetch_infant_template('12mo', subjects_dir=tmpdir)
    with pytest.raises(ValueError, match='Invalid value for'):
        fetch_infant_template('0mo', subjects_dir=tmpdir)
