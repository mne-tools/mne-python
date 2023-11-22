# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
import glob
import os

import numpy as np

from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...utils import _check_fname, _soft_import, fill_doc, logger, verbose
from ..base import BaseRaw


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
    return RawNeuralynx(fname, preload, verbose, exclude_fname_patterns)


@fill_doc
class RawNeuralynx(BaseRaw):
    """RawNeuralynx class."""

    @verbose
    def __init__(self, fname, preload=False, verbose=None, exclude_fname_patterns=None):
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
            raw_extras=[dict(smp2seg=sample2segment, exclude_fnames=exclude_fnames)],
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
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
        segment_sizes = np.array(
            [
                nlx_reader.get_signal_size(0, segment_id)
                for segment_id in range(len(neo_block[0].segments))
            ]
        )

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

        # now load data from selected segments/channels via
        # neo.Segment.AnalogSignal.load()
        all_data = np.concatenate(
            [
                signal.load(channel_indexes=idx).magnitude[
                    samples[0] : samples[-1] + 1, :
                ]
                for seg, samples in zip(
                    neo_block[0].segments[first_seg : last_seg + 1], sel_samples_local
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
