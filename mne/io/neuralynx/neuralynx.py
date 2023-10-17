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
        bl = nlx_reader.read(lazy=False)
        # TODO: This is massively inefficient -- we should not need to read *all*
        # samples just to get the ones from `idx` channels and `start:stop` time span.
        # But let's start here and make it efficient later.
        all_data = np.concatenate(
            [sig.magnitude for seg in bl[0].segments for sig in seg.analogsignals]
        ).T
        # ... but to get something that works, let's do it:
        block = all_data[:, start:stop]
        # Convert uV to V
        block *= 1e-6
        # Then store the result where it needs to go
        _mult_cal_one(data, block, idx, cals, mult)
