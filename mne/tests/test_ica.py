import mne
from mne.datasets import testing

data_path = testing.data_path(download=False)


@testing.requires_testing_data
def test_ica_get_sources_concatenated():
    """Test ICA get_sources method with concatenated raws."""
    # load data
    data_raw_file = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
    raw = (
        mne.io.read_raw_fif(data_raw_file).crop(0, 15).load_data()
    )  # raw has 15 seconds of data
    # concatenate
    raw_concat = mne.concatenate_raws(
        [raw.copy(), raw.copy()]
    )  # raw_concat has 30 seconds of data
    # do ICA
    ica = mne.preprocessing.ICA(n_components=20)
    ica.fit(raw_concat)
    # get sources
    raw_sources = ica.get_sources(raw_concat)  # but this only has 15 seconds of data
    assert raw_concat.n_times == raw_sources.n_times  # this will fail
