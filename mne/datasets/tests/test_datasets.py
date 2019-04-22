import os
from os import path as op
import shutil

import pytest

from mne import datasets
from mne.datasets import testing, set_montage_coreg_path
from mne.utils import (run_tests_if_main, requires_good_network, modified_env,
                       get_subjects_dir, ArgvSetter)


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
        set_montage_coreg_path()
        sd = get_subjects_dir()
        assert sd.endswith('MNE-fsaverage-data')


@requires_good_network
def test_megsim(tmpdir):
    """Test MEGSIM URL handling."""
    paths = datasets.megsim.load_data(
        'index', 'text', 'text', path=str(tmpdir), update_path=False)
    assert len(paths) == 1
    assert paths[0].endswith('index.html')


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


run_tests_if_main()
