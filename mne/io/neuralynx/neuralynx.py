# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime
import glob
import inspect
import os

import numpy as np

from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...annotations import Annotations
from ...utils import _check_fname, _soft_import, fill_doc, logger, verbose
from ..base import BaseRaw


@fill_doc
def read_raw_neuralynx(
    fname, *, preload=False, exclude_fname_patterns=None, verbose=None
) -> "RawNeuralynx":
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

    Notes
    -----
    Neuralynx files are read from disk using the `Neo package
    <http://neuralensemble.org/neo/>`__.
    Currently, only reading of the ``.ncs files`` is supported.

    ``raw.info["meas_date"]`` is read from the ``recording_opened`` property
    of the first ``.ncs`` file (i.e. channel) in the dataset (a warning is issued
    if files have different dates of acquisition).

    Channel-specific high and lowpass frequencies of online filters are determined
    based on the ``DspLowCutFrequency`` and ``DspHighCutFrequency`` header fields,
    respectively. If no filters were used for a channel, the default lowpass is set
    to the Nyquist frequency and the default highpass is set to 0.
    If channels have different high/low cutoffs, ``raw.info["highpass"]`` and
    ``raw.info["lowpass"]`` are then set to the maximum highpass and minimumlowpass
    values across channels, respectively.

    Other header variables can be inspected using Neo directly. For example::

        from neo.io import NeuralynxIO  # doctest: +SKIP
        fname = 'path/to/your/data'  # doctest: +SKIP
        nlx_reader = NeuralynxIO(dirname=fname)  # doctest: +SKIP
        print(nlx_reader.header)  # doctest: +SKIP
        print(nlx_reader.file_headers.items())  # doctest: +SKIP
    """
    return RawNeuralynx(
        fname,
        preload=preload,
        exclude_fname_patterns=exclude_fname_patterns,
        verbose=verbose,
    )


# Helper for neo deprecation of exclude_filename -> exclude_filenames in 0.13.2
def _exclude_kwarg(exclude_fnames):
    from neo.io import NeuralynxIO

    key = "exclude_filename"
    if "exclude_filenames" in inspect.getfullargspec(NeuralynxIO).args:
        key += "s"
    return {key: exclude_fnames}


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
        fname = _check_fname(fname, "read", True, "fname", need_dir=True)

        _soft_import("neo", "Reading NeuralynxIO files", strict=True)
        from neo.io import NeuralynxIO

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
            nlx_reader = NeuralynxIO(dirname=fname, **_exclude_kwarg(exclude_fnames))
        except ValueError as e:
            # give a more informative error message and what the user can do about it
            if "Incompatible section structures across streams" in str(e):
                raise ValueError(
                    "It seems .ncs channels have different numbers of samples. "
                    + "This is likely due to different sampling rates. "
                    + "Try reading in only channels with uniform sampling rate "
                    + "by excluding other channels with `exclude_fname_patterns` "
                    + "input argument."
                    + f"\nOriginal neo.NeuralynxRawIO ValueError:\n{e}"
                ) from None
            else:
                raise

        info = create_info(
            ch_types="seeg",
            ch_names=nlx_reader.header["signal_channels"]["name"].tolist(),
            sfreq=nlx_reader.get_signal_sampling_rate(),
        )

        ncs_fnames = nlx_reader.ncs_filenames.values()
        ncs_hdrs = [
            hdr
            for hdr_key, hdr in nlx_reader.file_headers.items()
            if hdr_key in ncs_fnames
        ]

        # if all files have the same recording_opened date, write it to info
        meas_dates = np.array([hdr["recording_opened"] for hdr in ncs_hdrs])
        # to be sure, only write if all dates are the same
        meas_diff = []
        for md in meas_dates:
            meas_diff.append((md - meas_dates[0]).total_seconds())

        # tolerate a +/-1 second meas_date difference (arbitrary threshold)
        # else issue a warning
        warn_meas = (np.abs(meas_diff) > 1.0).any()
        if warn_meas:
            logger.warning(
                "Not all .ncs files have the same recording_opened date. "
                + "Writing meas_date based on the first .ncs file."
            )

        # Neuarlynx allows channel specific low/highpass filters
        # if not enabled, assume default lowpass = nyquist, highpass = 0
        default_lowpass = info["sfreq"] / 2  # nyquist
        default_highpass = 0

        has_hp = [hdr["DSPLowCutFilterEnabled"] for hdr in ncs_hdrs]
        has_lp = [hdr["DSPHighCutFilterEnabled"] for hdr in ncs_hdrs]
        if not all(has_hp) or not all(has_lp):
            logger.warning(
                "Not all .ncs files have the same high/lowpass filter settings. "
                + "Assuming default highpass = 0, lowpass = nyquist."
            )

        highpass_freqs = [
            float(hdr["DspLowCutFrequency"])
            if hdr["DSPLowCutFilterEnabled"]
            else default_highpass
            for hdr in ncs_hdrs
        ]

        lowpass_freqs = [
            float(hdr["DspHighCutFrequency"])
            if hdr["DSPHighCutFilterEnabled"]
            else default_lowpass
            for hdr in ncs_hdrs
        ]

        with info._unlock():
            info["meas_date"] = meas_dates[0].astimezone(datetime.timezone.utc)
            info["highpass"] = np.max(highpass_freqs)
            info["lowpass"] = np.min(lowpass_freqs)

        # Neo reads only valid contiguous .ncs samples grouped as segments
        n_segments = nlx_reader.header["nb_segment"][0]
        block_id = 0  # assumes there's only one block of recording

        # get segment start/stop times
        start_times = np.array(
            [nlx_reader.segment_t_start(block_id, i) for i in range(n_segments)]
        )
        stop_times = np.array(
            [nlx_reader.segment_t_stop(block_id, i) for i in range(n_segments)]
        )

        # find discontinuous boundaries (of length n-1)
        next_start_times = start_times[1::]
        previous_stop_times = stop_times[:-1]
        seg_diffs = next_start_times - previous_stop_times

        # mark as discontinuous any two segments that have
        # start/stop delta larger than sampling period (1.5/sampling_rate)
        logger.info("Checking for temporal discontinuities in Neo data segments.")
        delta = 1.5 / info["sfreq"]
        gaps = seg_diffs > delta

        seg_gap_dict = {}

        logger.info(
            f"N = {gaps.sum()} discontinuous Neo segments detected "
            + f"with delta > {delta} sec. "
            + "Annotating gaps as BAD_ACQ_SKIP."
            if gaps.any()
            else "No discontinuities detected."
        )

        gap_starts = stop_times[:-1][gaps]  # gap starts at segment offset
        gap_stops = start_times[1::][gaps]  # gap stops at segment onset

        # (n_gaps,) array of ints giving number of samples per inferred gap
        gap_n_samps = np.array(
            [
                int(round(stop * info["sfreq"])) - int(round(start * info["sfreq"]))
                for start, stop in zip(gap_starts, gap_stops)
            ]
        ).astype(int)  # force an int array (if no gaps, empty array is a float)

        # get sort indices for all segments (valid and gap) in ascending order
        all_starts_ids = np.argsort(np.concatenate([start_times, gap_starts]))

        # variable indicating whether each segment is a gap or not
        gap_indicator = np.concatenate(
            [
                np.full(len(start_times), fill_value=0),
                np.full(len(gap_starts), fill_value=1),
            ]
        )
        gap_indicator = gap_indicator[all_starts_ids].astype(bool)

        # store this in a dict to be passed to _raw_extras
        seg_gap_dict = {
            "gap_n_samps": gap_n_samps,
            "isgap": gap_indicator,  # False (data segment) or True (gap segment)
        }

        valid_segment_sizes = [
            nlx_reader.get_signal_size(block_id, i) for i in range(n_segments)
        ]

        sizes_sorted = np.concatenate([valid_segment_sizes, gap_n_samps])[
            all_starts_ids
        ]

        # now construct an (n_samples,) indicator variable
        sample2segment = np.concatenate(
            [np.full(shape=(n,), fill_value=i) for i, n in enumerate(sizes_sorted)]
        )

        # get the start sample index for each gap segment ()
        gap_start_ids = np.cumsum(np.hstack([[0], sizes_sorted[:-1]]))[gap_indicator]

        # recreate time axis for gap annotations
        mne_times = np.arange(0, len(sample2segment)) / info["sfreq"]

        assert len(gap_start_ids) == len(gap_n_samps)
        annotations = Annotations(
            onset=[mne_times[onset_id] for onset_id in gap_start_ids],
            duration=[
                mne_times[onset_id + (n - 1)] - mne_times[onset_id]
                for onset_id, n in zip(gap_start_ids, gap_n_samps)
            ],
            description=["BAD_ACQ_SKIP"] * len(gap_start_ids),
        )

        super().__init__(
            info=info,
            last_samps=[sizes_sorted.sum() - 1],
            filenames=[fname],
            preload=preload,
            raw_extras=[
                dict(
                    smp2seg=sample2segment,
                    exclude_fnames=exclude_fnames,
                    segment_sizes=sizes_sorted,
                    seg_gap_dict=seg_gap_dict,
                )
            ],
        )

        self.set_annotations(annotations)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        from neo import AnalogSignal, Segment
        from neo.io import NeuralynxIO
        from neo.io.proxyobjects import AnalogSignalProxy

        # quantities is a dependency of neo so we are guaranteed it exists
        from quantities import Hz

        nlx_reader = NeuralynxIO(
            dirname=self.filenames[fi],
            **_exclude_kwarg(self._raw_extras[0]["exclude_fnames"]),
        )
        neo_block = nlx_reader.read(lazy=True)

        # check that every segment has 1 associated neo.AnalogSignal() object
        # (not sure what multiple analogsignals per neo.Segment would mean)
        assert sum(
            [len(segment.analogsignals) for segment in neo_block[0].segments]
        ) == len(neo_block[0].segments)

        segment_sizes = self._raw_extras[fi]["segment_sizes"]

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
        sel_samples_local[1::, 0] = (
            0  # now set the start sample for all segments after the first to 0
        )

        sel_samples_local[0, 0] = (
            start - sel_samples_global[0, 0]
        )  # express start sample relative to segment onset
        sel_samples_local[-1, -1] = (stop - 1) - sel_samples_global[
            -1, 0
        ]  # express stop sample relative to segment onset

        # array containing Segments
        segments_arr = np.array(neo_block[0].segments, dtype=object)

        # if gaps were detected, correctly insert gap Segments in between valid Segments
        gap_samples = self._raw_extras[fi]["seg_gap_dict"]["gap_n_samps"]
        gap_segments = [Segment(f"gap-{i}") for i in range(len(gap_samples))]

        # create AnalogSignal objects representing gap data filled with 0's
        sfreq = nlx_reader.get_signal_sampling_rate()
        n_chans = (
            np.arange(idx.start, idx.stop, idx.step).size
            if type(idx) is slice
            else len(idx)  # idx can be a slice or an np.array so check both
        )

        for seg, n in zip(gap_segments, gap_samples):
            asig = AnalogSignal(
                signal=np.zeros((n, n_chans)), units="uV", sampling_rate=sfreq * Hz
            )
            seg.analogsignals.append(asig)

        n_total_segments = len(neo_block[0].segments + gap_segments)
        segments_arr = np.zeros((n_total_segments,), dtype=object)

        # insert inferred gap segments at the right place in between valid segments
        isgap = self._raw_extras[0]["seg_gap_dict"]["isgap"]
        segments_arr[~isgap] = neo_block[0].segments
        segments_arr[isgap] = gap_segments

        # now load data for selected segments/channels via
        # neo.Segment.AnalogSignalProxy.load() or
        # pad directly as AnalogSignal.magnitude for any gap data
        all_data = np.concatenate(
            [
                signal.load(channel_indexes=idx).magnitude[
                    samples[0] : samples[-1] + 1, :
                ]
                if isinstance(signal, AnalogSignalProxy)
                else signal.magnitude[samples[0] : samples[-1] + 1, :]
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
