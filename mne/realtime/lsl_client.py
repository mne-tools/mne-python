# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import numpy as np
from .base_client import _BaseClient

try:
    import pylsl
except ImportError as err:
    print('Need to install pylsl')

class LSLClient(_BaseClient):
    """Base Realtime Client.

    Parameters
    ----------
    identifier : str
        The identifier of the server. IP address or LSL id or raw filename.
    port : int | None
        Port to use for the connection.
    tmin : float | None
        Time instant to start receiving buffers. If None, start from the latest
        samples available.
    tmax : float
        Time instant to stop receiving buffers.
    buffer_size : int
        Size of each buffer in terms of number of samples.
    wait_max : float
        Maximum time (in seconds) to wait for real-time buffer to start
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    """
    def __init__(self, identifier, port=None, tmin=None, tmax=np.inf,
                 buffer_size=1000, verbose=None):
        self.identifier = identifier
        self.port = port
        self.tmin = tmin
        self.tmax = tmax
        self.buffer_size = buffer_size
        self.verbose = verbose

    def connect(self):
        self.client = pylsl.resolve_byprop('source_id', self.identifier,
                                           timeout=self.wait_max)[0]

        return self

    def iter_raw_buffers(self):
        """Return an iterator over raw buffers.
        """
        inlet = pylsl.StreamInlet(self.client)

        while True:
            sample, timestamp = inlet.pull_chunk(max_samples=self.buffer_size)

            yield sample
