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

        # construct an array of shape (n_total_samples,) indicating
        # segment membership for each sample
        sample2segment = np.concatenate(
            [
                np.full(shape=(nlx_reader.get_signal_size(block_id, i),), fill_value=i)
                for i in range(n_segments)
            ]
        )

        super(RawNeuralynx, self).__init__(
            info=info,
            last_samps=[n_total_samples - 1],
            filenames=[fname],
            preload=preload,
            raw_extras=[dict(smp2seg=sample2segment)],
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        from neo.io import NeuralynxIO

        nlx_reader = NeuralynxIO(dirname=self._filenames[fi])
        neo_block = nlx_reader.read(lazy=True)
        sr = nlx_reader.header["signal_channels"]["sampling_rate"][0]

        # check that every segment has 1 associated neo.AnalogSignal() object
        # (not sure what multiple analogsignals per neo.Segment would mean)
        assert sum([len(segment.analogsignals) for segment in neo_block[0].segments]) == len(neo_block[0].segments)

        # gather the start and stop times (in sec) for all signals in segments (assumes 1 signal per segment)
        # shape = (n_segments, 2) where 2nd dim is (start_time, stop_time)
        start_stop_times = np.array(
            [(signal.t_start.item(), signal.t_stop.item()) 
            for segment in neo_block[0].segments 
            for signal in segment.analogsignals
            ]
        )

        # get times (in sec) for the first and last segment
        start_time = start/sr
        stop_time = stop/sr

        first_seg = self._raw_extras[0]["smp2seg"][start]  # segment index for the first sample
        last_seg = self._raw_extras[0]["smp2seg"][stop-1]  # segment index for the last sample

        # now select only segments between the one that containst the start sample
        # and the one that contains the stop sample
        sel_times = start_stop_times[first_seg:last_seg+1, :]
        
        # if we're reading in later than first sample in first segment
        # or earlier than the last sample in last segment
        # we need to adjust the start and stop times accordingly
        if start_time > sel_times[0, 0]:
            sel_times[0, 0] = start_time
        if stop_time < sel_times[-1, -1]:
            sel_times[-1, 1] = stop_time

        # now load the data arrays via neo.Segment.Analogsignal.load()
        # only from the selected segments and channels
        all_data = np.concatenate(
            [
                signal.load(
                    channel_indexes=idx, time_slice=(time[0], time[-1])
                ).magnitude
                for seg, time in zip(
                    bl[0].segments[first_seg : last_seg + 1], sel_times
                )
                for signal in seg.analogsignals
            ]
        ).T
        all_data *= 1e-6  # Convert uV to V

        n_channels = len(nlx_reader.header["signal_channels"]["name"])
        block = np.zeros((n_channels, stop - start), dtype=data.dtype)
        block[idx] = all_data  # shape = (n_channels, n_samples)

        # Then store the result where it needs to go
        _mult_cal_one(data, block, idx, cals, mult)
