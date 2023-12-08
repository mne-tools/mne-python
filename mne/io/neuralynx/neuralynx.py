# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
import glob
import os

import numpy as np
from numpy.testing import assert_allclose

from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...utils import _check_fname, _soft_import, fill_doc, logger, verbose
from ...annotations import Annotations
from ..base import BaseRaw


from neo import AnalogSignal

class AnalogSignalGap(object):
    """Dummy object to represent gaps in Neuralynx data as
    AnalogSignalProxy-like objects. Propagate `signal`, `units`, and 
    `sampling_rate` attributes to the `AnalogSignal` object returned by `load()`. 
    """
    def __init__(self, signal, units, sampling_rate):

        self.signal = signal
        self.units = units
        self.sampling_rate = sampling_rate

    def load(self, channel_indexes):
        """Dummy method such that it returns object and we access .magnitude"""

        # self.magnitude = self.magnitude[channel_indexes, :]
        sig = AnalogSignal(signal=self.signal[channel_indexes, :], 
                           units=self.units,
                           sampling_rate=self.sampling_rate) 
        return sig



@fill_doc
def read_raw_neuralynx(
    fname, *, preload=False, exclude_fname_patterns=None, verbose=None
):
    """Reader for Neuralynx files.

    Parameters
    ----------
    fname : path-like
        Path to a folder with Neuralynx .ncs files.
    %(preload)s
    exclude_fname_patterns : list of str
        List of glob-like string patterns to exclude from channel list.
        Useful when not all channels have the same number of samples
        so you can read separate instances.
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
    return RawNeuralynx(
        fname,
        preload=preload,
        exclude_fname_patterns=exclude_fname_patterns,
        verbose=verbose,
    )


@fill_doc
class RawNeuralynx(BaseRaw):
    """RawNeuralynx class."""

    @verbose
    def __init__(
        self,
        fname,
        *,
        preload=False,
        exclude_fname_patterns=None,
        verbose=None,
    ):
        _soft_import("neo", "Reading NeuralynxIO files", strict=True)
        from neo.io import NeuralynxIO

        fname = _check_fname(fname, "read", True, "fname", need_dir=True)

        logger.info(f"Checking files in {fname}")

        # construct a list of filenames to ignore
        exclude_fnames = None
        if exclude_fname_patterns:
            exclude_fnames = []
            for pattern in exclude_fname_patterns:
                fnames = glob.glob(os.path.join(fname, pattern))
                fnames = [os.path.basename(fname) for fname in fnames]
                exclude_fnames.extend(fnames)

            logger.info("Ignoring .ncs files:\n" + "\n".join(exclude_fnames))

        # get basic file info from header, throw Error if NeuralynxIO can't parse
        try:
            nlx_reader = NeuralynxIO(dirname=fname, exclude_filename=exclude_fnames)
        except ValueError as e:
            raise ValueError(
                "It seems some .ncs channels might have different number of samples. "
                + "This is likely due to different sampling rates. "
                + "Try excluding them with `exclude_fname_patterns` input arg."
                + f"\nOriginal neo.NeuralynxIO.parse_header() ValueError:\n{e}"
            )

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

        # get segment start/stop times
        start_times = np.array([nlx_reader.segment_t_start(block_id, i) for i in range(n_segments)])
        stop_times = np.array([nlx_reader.segment_t_stop(block_id, i) for i in range(n_segments)])

        # find discontinuous boundaries (of length n-1)
        next_start_times = start_times[1::]
        previous_stop_times = stop_times[:-1]
        seg_diffs = next_start_times - previous_stop_times

        # mark as discontinuous any two segments that have 
        # start/stop delta larger than sampling period (1/sampling_rate)

        delta = 1/info["sfreq"] 
        gaps = seg_diffs > delta
        has_gaps = gaps.any()

        seg_gap_dict = {}
        gap_segment_sizes = []
        gap_annotations = {}

        if has_gaps:

            logger.info(f"N = {gaps.sum()} discontinuous Neo segments detected with delta > {delta} sec.\n(max = {seg_diffs[gaps].max()} sec, min = {seg_diffs[gaps].min()})")

            gap_starts = stop_times[:-1][gaps]  # gap starts at segment offset
            gap_stops = start_times[1::][gaps]  # gap stops at segment onset

            # (n_gaps,) array of ints giving number of samples per inferred gap
            gap_n_samps = np.array(
                [len(np.arange(on_spl, off_spl)) 
                for on_spl, off_spl in zip(gap_starts*info["sfreq"], gap_stops*info["sfreq"])
                ]
            )

            # add the inferred gaps into the right place in the segment list
            all_starts_ids = np.argsort(np.concatenate([start_times, gap_starts]))
            all_stops_ids = np.argsort(np.concatenate([stop_times, gap_stops]))
            
            # sort the valid segment and gap times by time
            all_starts = np.concatenate([start_times, gap_starts])[all_starts_ids]
            all_stops = np.concatenate([stop_times, gap_stops])[all_stops_ids]
            
            # variable indicating whether each segment is a gap or not
            gap_indicator = np.concatenate(
                [np.full(len(start_times), fill_value=0),
                np.full(len(gap_starts), fill_value=1)
                ]
            )
            gap_indicator = gap_indicator[all_starts_ids].astype(bool)

            # store this in a dict to be passed to _raw_extras
            seg_gap_dict = {
                "onsets": all_starts, # onsets in seconds
                "offsets": all_stops,
                "gap_n_samps": gap_n_samps,
                "isgap": gap_indicator, # 0 (data segment) or 1 (invalid segment for BAD_SKIP_ACQ)
            }

            # TMP: annotations dict for use with mne.Annotations
            gap_annotations = dict(onset=gap_starts, duration=seg_diffs[gaps], orig_time=None, description="BAD_ACQ_SKIP")

            gap_segment_sizes = [
                n for n in gap_n_samps
            ]

        else:
            logger.info(f"All Neo segments temporally continuous at {delta} sec precision.")

        # check that segment[-1] stop and segment[i] start times
        # matched to microsecond precision (1e-6)
        #breakpoint()
        #assert_allclose(stop_times[:-1]-start_times[1::], 0, atol=1e-3,
        #                err_msg="Segments start/end times are not temporally contiguous."
        #)

        valid_segment_sizes = [
            nlx_reader.get_signal_size(block_id, i)
            for i in range(n_segments)
        ]
        
        if has_gaps:
            sizes_sorted = np.concatenate([valid_segment_sizes, gap_segment_sizes])[all_starts_ids]
        else:
            sizes_sorted = np.array(valid_segment_sizes)

        # now construct an (n_samples,) indicator variable
        sample2segment = np.concatenate(
            [np.full(shape=(n,), fill_value=i)
            for i, n in enumerate(sizes_sorted)]
        )

        # construct an array of shape (n_total_samples,) indicating
        # segment membership for each sample
        #sample2segment = np.concatenate(
        #    [
        #        np.full(shape=(nlx_reader.get_signal_size(block_id, i),), fill_value=i)
        #        for i in range(n_segments)
        #    ]
        #)

        super(RawNeuralynx, self).__init__(
            info=info,
            last_samps=[n_total_samples - 1],
            filenames=[fname],
            preload=preload,
            raw_extras=[dict(smp2seg=sample2segment, exclude_fnames=exclude_fnames, segment_sizes=sizes_sorted, seg_gap_dict=seg_gap_dict, gap_annotations=gap_annotations)],
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        from quantities import Hz
        from neo import Segment
        from neo.io import NeuralynxIO

        nlx_reader = NeuralynxIO(
            dirname=self._filenames[fi],
            exclude_filename=self._raw_extras[0]["exclude_fnames"],
        )
        neo_block = nlx_reader.read(lazy=True)

        # check that every segment has 1 associated neo.AnalogSignal() object
        # (not sure what multiple analogsignals per neo.Segment would mean)
        assert sum(
            [len(segment.analogsignals) for segment in neo_block[0].segments]
        ) == len(neo_block[0].segments)

        # collect sizes of each segment
        #segment_sizes = np.array(
        #    [
        #        nlx_reader.get_signal_size(0, segment_id)
        #        for segment_id in range(len(neo_block[0].segments))
        #    ]
        #)

        segment_sizes = self._raw_extras[0]["segment_sizes"]

        # construct a (n_segments, 2) array of the first and last
        # sample index for each segment relative to the start of the recording
        seg_starts = [0]  # first chunk starts at sample 0
        seg_stops = [segment_sizes[0] - 1]
        for i in range(1, len(segment_sizes)):
            ons_new = (
                seg_stops[i - 1] + 1
            )  # current chunk starts one sample after the previous one
            seg_starts.append(ons_new)
            off_new = (
                seg_stops[i - 1] + segment_sizes[i]
            )  # the last sample is len(chunk) samples after the previous ended
            seg_stops.append(off_new)

        start_stop_samples = np.stack([np.array(seg_starts), np.array(seg_stops)]).T

        first_seg = self._raw_extras[0]["smp2seg"][
            start
        ]  # segment containing start sample
        last_seg = self._raw_extras[0]["smp2seg"][
            stop - 1
        ]  # segment containing stop sample

        # select all segments between the one that contains the start sample
        # and the one that contains the stop sample
        sel_samples_global = start_stop_samples[first_seg : last_seg + 1, :]

        # express end samples relative to segment onsets
        # to be used for slicing the arrays below
        sel_samples_local = sel_samples_global.copy()
        sel_samples_local[0:-1, 1] = (
            sel_samples_global[0:-1, 1] - sel_samples_global[0:-1, 0]
        )
        sel_samples_local[
            1::, 0
        ] = 0  # now set the start sample for all segments after the first to 0

        sel_samples_local[0, 0] = (
            start - sel_samples_global[0, 0]
        )  # express start sample relative to segment onset
        sel_samples_local[-1, -1] = (stop - 1) - sel_samples_global[
            -1, 0
        ]  # express stop sample relative to segment onset

        # array containing Segments
        segments_arr = np.array(neo_block[0].segments, dtype=object)

        # if gaps were detected correctly insert gap Segments in between valid Segments
        if self._raw_extras[0]["seg_gap_dict"]:

            gap_samples = self._raw_extras[0]["seg_gap_dict"]["gap_n_samps"]
            gap_segments = [Segment(f"gap-{i}") for i in range(len(gap_samples))]

            # create AnalogSignal objects representing gap data
            sfreq = nlx_reader.get_signal_sampling_rate()
            n_chans = np.arange(idx.start, idx.stop, idx.step).size

            for seg, n in zip(gap_segments, gap_samples):
                asig = AnalogSignalGap(signal=np.zeros((n, n_chans)), units="uV", sampling_rate=sfreq * Hz)
                seg.analogsignals.append(asig)

            n_total_segments = len(neo_block[0].segments + gap_segments)
            segments_arr = np.zeros((n_total_segments,), dtype=object)
            isgap = self._raw_extras[0]["seg_gap_dict"]["isgap"]
            segments_arr[~isgap] = neo_block[0].segments
            segments_arr[isgap] = gap_segments

        # now load data from selected segments/channels via
        # neo.Segment.AnalogSignal.load()
        all_data = np.concatenate(
            [
                signal.load(channel_indexes=idx).magnitude[
                    samples[0] : samples[-1] + 1, :
                ]
                for seg, samples in zip(
                    segments_arr[first_seg : last_seg + 1], sel_samples_local
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
