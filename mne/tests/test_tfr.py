def test_epochstfr_from_rawtfr_and_concatenate():
    import numpy as np

    import mne
    from mne import create_info
    from mne.time_frequency import EpochsTFRArray, RawTFR, concatenate_epochs_tfr

    sfreq = 100
    freqs = np.arange(10, 20)
    times = np.linspace(0, 1, 100)
    n_channels = 2

    data = np.random.rand(n_channels, len(freqs), len(times))
    info = create_info(ch_names=["EEG 001", "EEG 002"], sfreq=sfreq)
    raw_tfr = RawTFR(data[np.newaxis, ...], info, times, freqs, nave=1)

    # Add annotations to simulate events
    raw_tfr.set_annotations(
        mne.Annotations(onset=[0.5], duration=[0], description=["stim"])
    )

    epochs = EpochsTFRArray(data=raw_tfr)
    assert epochs.data.shape[0] == 1

    # Concatenate the same object twice
    combined = concatenate_epochs_tfr([epochs, epochs])
    assert combined.data.shape[0] == 2
