# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import os
import os.path as op
from shutil import copyfile

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse

from mne.datasets import testing
from mne.utils import catch_logging, _record_warnings
from mne import read_morph_map

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')


@pytest.mark.slowtest
@testing.requires_testing_data
def test_make_morph_maps(tmp_path):
    """Test reading and creating morph maps."""
    # make a new fake subjects_dir
    tempdir = str(tmp_path)
    for subject in ('sample', 'sample_ds', 'fsaverage_ds'):
        os.mkdir(op.join(tempdir, subject))
        os.mkdir(op.join(tempdir, subject, 'surf'))
        regs = ('reg', 'left_right') if subject == 'fsaverage_ds' else ('reg',)
        for hemi in ['lh', 'rh']:
            for reg in regs:
                args = [subject, 'surf', hemi + '.sphere.' + reg]
                copyfile(op.join(subjects_dir, *args),
                         op.join(tempdir, *args))

    for subject_from, subject_to, xhemi in (
            ('fsaverage_ds', 'sample_ds', False),
            ('fsaverage_ds', 'fsaverage_ds', True)):
        # trigger the creation of morph-maps dir and create the map
        with catch_logging() as log:
            mmap = read_morph_map(subject_from, subject_to, tempdir,
                                  xhemi=xhemi, verbose=True)
        log = log.getvalue()
        assert 'does not exist' in log
        assert 'Creating' in log
        mmap2 = read_morph_map(subject_from, subject_to, subjects_dir,
                               xhemi=xhemi)
        assert len(mmap) == len(mmap2)
        for m1, m2 in zip(mmap, mmap2):
            # deal with sparse matrix stuff
            diff = (m1 - m2).data
            assert_allclose(diff, np.zeros_like(diff), atol=1e-3, rtol=0)

    # This will also trigger creation, but it's trivial
    with _record_warnings():
        mmap = read_morph_map('sample', 'sample', subjects_dir=tempdir)
    for mm in mmap:
        assert (mm - sparse.eye(mm.shape[0], mm.shape[0])).sum() == 0
