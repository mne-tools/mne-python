# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
from shutil import copyfile

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import read_morph_map
from mne.datasets import testing
from mne.fixes import _eye_array
from mne.utils import _record_warnings, catch_logging

data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"


@pytest.mark.slowtest
@testing.requires_testing_data
def test_make_morph_maps(tmp_path):
    """Test reading and creating morph maps."""
    pytest.importorskip("nibabel")
    # make a new fake subjects_dir
    for subject in ("sample", "sample_ds", "fsaverage_ds"):
        os.mkdir(tmp_path / subject)
        os.mkdir(tmp_path / subject / "surf")
        regs = ("reg", "left_right") if subject == "fsaverage_ds" else ("reg",)
        for hemi in ["lh", "rh"]:
            for reg in regs:
                copyfile(
                    subjects_dir / subject / "surf" / f"{hemi}.sphere.{reg}",
                    tmp_path / subject / "surf" / f"{hemi}.sphere.{reg}",
                )

    for subject_from, subject_to, xhemi in (
        ("fsaverage_ds", "sample_ds", False),
        ("fsaverage_ds", "fsaverage_ds", True),
    ):
        # trigger the creation of morph-maps dir and create the map
        with catch_logging() as log:
            mmap = read_morph_map(
                subject_from, subject_to, tmp_path, xhemi=xhemi, verbose=True
            )
        log = log.getvalue()
        assert "does not exist" in log
        assert "Creating" in log
        mmap2 = read_morph_map(subject_from, subject_to, subjects_dir, xhemi=xhemi)
        assert len(mmap) == len(mmap2)
        for m1, m2 in zip(mmap, mmap2):
            # deal with sparse matrix stuff
            diff = (m1 - m2).data
            assert_allclose(diff, np.zeros_like(diff), atol=1e-3, rtol=0)

    # This will also trigger creation, but it's trivial
    with _record_warnings():
        mmap = read_morph_map("sample", "sample", subjects_dir=tmp_path)
    for mm in mmap:
        assert (mm - _eye_array(mm.shape[0])).sum() == 0
