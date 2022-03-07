from mne.io.xdf import read_raw_xdf


def test_read():
    path_to_sample_xdf = './eeg_baseline.raw'
    # this file one 128ch type='EEG' data stream, and one type='Markers' stream.
    # It contains other string streams (used for logging) that be used as string markers

    # read xdf with default values
    raw = read_raw_xdf(fname=path_to_sample_xdf)
    assert raw is not None

    # read xdf with an alternative string marker source
    raw = read_raw_xdf(fname=path_to_sample_xdf, data_type_markers='msg_states')
    assert raw is not None


    # read xdf with specific names
    raw = read_raw_xdf(fname=path_to_sample_xdf, name_stream_eeg='BrainAmpSeries-Dev_1',
                       name_stream_markers='ServerStates')
    assert raw is not None


    '''
    import matplotlib
    import pyxdf
    from mne.io.xdf import read_raw_xdf
    matplotlib.use("TkAgg")

    raw = read_raw_xdf(fname=path_to_sample_xdf, name_stream_eeg='BrainAmpSeries-Dev_1',
                       name_stream_markers='ServerStates')

    raw.plot(scalings='auto')
    '''



