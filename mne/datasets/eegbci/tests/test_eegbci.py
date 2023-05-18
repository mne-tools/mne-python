# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD Style.

from mne.datasets import eegbci


def test_eegbci_download(tmp_path, fake_retrieve):
    """Test Sleep Physionet URL handling."""
    for subj in range(4):
        fnames = eegbci.load_data(subj + 1, runs=[3], path=tmp_path, update_path=False)
        assert len(fnames) == 1, subj
    assert fake_retrieve.call_count == 4
