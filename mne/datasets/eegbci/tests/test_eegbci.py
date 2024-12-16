# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import pytest

from mne.datasets import eegbci


def test_eegbci_download(tmp_path, fake_retrieve):
    """Test Sleep Physionet URL handling."""
    subjects = range(1, 5)
    for subj in subjects:
        fnames = eegbci.load_data(subj, runs=[3], path=tmp_path, update_path=False)
        assert len(fnames) == 1, subj
    assert fake_retrieve.call_count == 4

    # XXX: remove in version 1.9
    with pytest.warns(FutureWarning, match="The ``subject``"):
        fnames = eegbci.load_data(
            subject=subjects, runs=[3], path=tmp_path, update_path=False
        )
        assert len(fnames) == 4

    # XXX: remove in version 1.9
    with pytest.raises(ValueError, match="You must pass the parameters"):
        fnames = eegbci.load_data(path=tmp_path, update_path=False)
