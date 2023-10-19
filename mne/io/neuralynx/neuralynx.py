import numpy as np

from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...utils import _check_fname, _soft_import, fill_doc, logger, verbose
from ..base import BaseRaw


@fill_doc
def read_raw_neuralynx(fname, preload=False, verbose=None):
    """Reader for an Neuralynx files.

    Parameters
    ----------
    fname : path-like
        Path to a folder with Neuralynx .ncs files
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawNeuralynx
        A Raw object containing Neuralynx data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawNeuralynx.
    """
    return RawNeuralynx(fname, preload, verbose)


@fill_doc
class RawNeuralynx(BaseRaw):
    """RawNeuralynx class."""

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        _soft_import("neo", "Reading NeuralynxIO files", strict=True)
        from neo.io import NeuralynxIO

        fname = _check_fname(fname, "read", True, "fname", need_dir=True)

        logger.info(f"Checking files in {fname}")

        # get basic file info
        nlx_reader = NeuralynxIO(dirname=fname)

        info = create_info(
            ch_types="seeg",
            ch_names=nlx_reader.header["signal_channels"]["name"].tolist(),
            sfreq=nlx_reader.get_signal_sampling_rate(),
        )

        # find total number of samples per .ncs file (`channel`) by summing
        # the sample sizes of all segments
        n_segments = nlx_reader.header["nb_segment"][0]
        block_id = 0  # assumes there's only one block of recording
        n_total_samples = sum(
            nlx_reader.get_signal_size(block_id, segment)
            for segment in range(n_segments)
        )
        super(RawNeuralynx, self).__init__(
            info=info,
            last_samps=[n_total_samples - 1],
            filenames=[fname],
            preload=preload,
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        from neo.io import NeuralynxIO

        nlx_reader = NeuralynxIO(dirname=self._filenames[fi])
        bl = nlx_reader.read(lazy=True)
        # TODO: This is massively inefficient -- we should not need to read *all*
        # samples just to get the ones from `idx` channels and `start:stop` time span.
        # KA: neuralynx stores continuous recording in smaller segments (read in as neo.Segment objects)
        # below code tries to be more efficient it delays reading into memory (nlx_reader.read(lazy=True)) 
        # and then reads in the idx chanels and specified times by passing these as arguments in neo.Segment.AnalogSignal.load()

        sr = nlx_reader.header["signal_channels"]["sampling_rate"][0]

        # check that every segment has 1 associated neo.AnalogSignal() object (not sure what multiple analogsignals per neo.Segment would mean)
        assert sum([len(segment.analogsignals) for segment in bl[0].segments]) == len(bl[0].segments)

        # just a wrapper for readability
        def _find_first_last_segment(segments, start_spl:int, stop_spl:int, sr:float):

            # gather the start and stop times (in sec) for all signals in segments (assumes 1 signal per segment)
            # shape = (n_segments, 2) where 2nd dim is (start_time, stop_time)
            onst_offt = np.array([(signal.t_start.item(), signal.t_stop.item()) for segment in segments for signal in segment.analogsignals])  

            # if start/stop are not speficied, read all segments
            start_segment = 0
            stop_segment = len(segments) - 1  # -1 because python indexing

            # determine which segment contains the start sample and which the stop sample if start/stop is specified
            if start_spl > 0: 
                start_time = start_spl/sr
                start_segment = np.where((onst_offt[:, 0] <= start_time) & (onst_offt[:, 1] >= start_time))[0].item()
            if stop_spl is not None:
                stop_time = stop_spl/sr
                stop_segment = np.where((onst_offt[:, 0] <= stop_time) & (onst_offt[:, 1] >= stop_time))[0].item()

            return start_segment, stop_segment, onst_offt
        

        stop_spl = None if stop == data.shape[-1] else stop  # assume we read until the end if not speficied otherwise

        first_seg, last_seg, start_stop_times = _find_first_last_segment(bl[0].segments, 
                                                                        start_spl=start, 
                                                                        stop_spl=stop_spl, 
                                                                        sr=sr
                                                                        )
        

        # now select only segments between the one that containst the start sample
        # and the one that contains the stop sample
        sel_times = start_stop_times[first_seg:last_seg+1, :]
        
        if (start is not None) or (start > 0):
            sel_times[0, 0] = start/sr   # for the segment containting the start sample, reset the segment start time to the requested `start` value
        if stop_spl is not None:
            sel_times[-1, 1] = stop/sr   # same but for stop sample

        # now load the data arrays via neo.Segment.Analogsignal.load() only from the selected segments and channels
        all_data = np.concatenate(
            [signal.load(channel_indexes=idx, time_slice=(time[0], time[-1])).magnitude for seg, time in zip(bl[0].segments[first_seg:last_seg+1], sel_times) for signal in seg.analogsignals]
        ).T

        block = all_data  # shape = (len(idx), n_samples))

        # Convert uV to V
        block *= 1e-6

        # Then store the result where it needs to go
        _mult_cal_one(data, block, idx, cals, mult)
