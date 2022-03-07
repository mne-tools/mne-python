from pyxdf import load_xdf, match_streaminfos, resolve_streams
from ...utils import verbose, logger, warn
from ...io import RawArray
from ... import create_info


def read_raw_xdf(fname,
                 name_stream_eeg: str = None,
                 name_stream_markers: str = None,
                 data_type: str = 'EEG',
                 data_type_markers: str = 'Markers',
                 *args, **kwargs):
    """Read XDF file.
    Either specify the name or the stream id


    Note that it does not recognize different data types in the same stream  (e.g.: eeg + misc)
    Parameters
    ----------
    fname : str
        Name of the XDF file.
    name_stream_eeg : int
        name of the data stream to load (optional).
    name_stream_markers : int
        name of a specific marker stream to load (optional), otherwise inserts all marker streams.
    data_type : str
        type of the data stream to load (default: 'EEG')
    data_type_markers : str
        type of the marker stream to load (default: 'Markers')

    Returns
    -------
    raw : mne.io.Raw
        fromn XDF file data.
    """

    # load the xdf file
    streams, header = load_xdf(fname)


    # Build up a query with data type and name for data stream
    if name_stream_eeg is not None:
        eeg_stream_query = [{'name': name_stream_eeg}]
    else:
        eeg_stream_query = [{'type': data_type}]

    # Search for a specific eeg stream
    eeg_stream_found = False

    for stream in streams:
        eeg_streams = match_streaminfos(resolve_streams(fname), eeg_stream_query)
        if stream["info"]["stream_id"] in eeg_streams:
            eeg_stream_found = True
            break  # only selects the first matching stream

    assert eeg_stream_found, 'No EEG stream found'


    # Load EEG data information to compose the info
    n_chans = int(stream["info"]["channel_count"][0])
    fs = float(stream["info"]["nominal_srate"][0])
    labels, types, units = [], [], []
    try:
        for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
            labels.append(str(ch["label"][0]))
            if ch["type"]:
                types.append(ch["type"][0])
            if ch["unit"]:
                units.append(ch["unit"][0])
    except (TypeError, IndexError):  # no channel labels found
        pass
    if not labels:
        labels = [str(n) for n in range(n_chans)]
    if not units:
        units = ["NA" for _ in range(n_chans)]

    info = create_info(ch_names=labels, sfreq=fs, ch_types=data_type.lower())  # check types as a list
    # Create the info
    raw = RawArray((stream["time_series"]).T, info)
    # define manually the _filenames which may cause errors otherwise
    raw._filenames = [fname]

    # keep the first sample timestamp to align markers
    first_samp = stream["time_stamps"][0]


    # Find the markers streams

    # Define the query
    if name_stream_markers is not None:
        marker_stream_query = [{'name': name_stream_markers}]  # optional name for markers
    else:
        marker_stream_query = [{'type': data_type_markers}]

    markers = match_streaminfos(resolve_streams(fname), marker_stream_query)

    # Iterate over marker stream ids
    for stream_id in markers:
        for stream in streams:

            # if stream_id matches, add the markers as annotations
            if stream["info"]["stream_id"] == stream_id:
                # realigns the first time stamp to the data
                onsets = stream["time_stamps"] - first_samp  # **** IMPORTANT ***
                # extract description labels
                descriptions = [item for sub in stream["time_series"] for item in sub]
                # add to the raw as annoations
                raw.annotations.append(onsets, [0] * len(onsets), descriptions)
    return raw
