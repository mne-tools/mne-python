import os.path as op
from ..base import BaseRaw
from ..._fiff.utils import _read_segments_file
from ..._fiff.meas_info import create_info
from ...utils import logger, verbose, fill_doc


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
        try:
            from neo.io import NeuralynxIO
        except Exception:
            raise ImportError("Missing the neo-python package") from None

        datadir = op.abspath(fname)

        logger.info(f"Checking files in {datadir}")

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
            [
                nlx_reader.get_signal_size(block_id, segment)
                for segment in range(n_segments)
            ]
        )

        # loop over found filenames and collect names and store last sample numbers
        last_samps = []
        ncs_fnames = []

        for chan_key in nlx_reader.ncs_filenames.keys():
            ncs_fname = nlx_reader.ncs_filenames[chan_key]
            ncs_fnames.append(ncs_fname)
            last_samps.append(
                n_total_samples - 1
            )  # assumes the same sample size for all files/channels

        super(RawNeuralynx, self).__init__(
            info=info,
            last_samps=last_samps[0:1],
            filenames=ncs_fnames,
            preload=preload,
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        _read_segments_file(
            self, data, idx, fi, start, stop, cals, mult, dtype="<i2", n_channels=1
        )
