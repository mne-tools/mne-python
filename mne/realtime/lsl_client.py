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
        stream = pylsl.resolve_byprop('source_id', self.identifier,
                                           timeout=self.wait_max)[0]
        self.client = pylsl.StreamInlet(stream)

        return self

    def create_info(self):
        sfreq = self.client.info().nominal_srate()

        lsl_info = self.client.info()
        ch_info = lsl_info.desc().child("channels").child("channel")
        ch_names = list()
        ch_types = list()
        for k in range(lsl_info.channel_count()):
            ch_names.append(ch_info.child_value("label"))
            ch_types.append(ch_info.child_value("type"))
            ch_info = ch_info.next_sibling()

        info = create_info(ch_names, sfreq, ch_types)

        self.info = info

        return self

    def iter_raw_buffers(self):
        """Return an iterator over raw buffers.
        """
        inlet = pylsl.StreamInlet(self.client)

        ## add tmin and tmax to this logic

        while True:
            samples, timestamps = inlet.pull_chunk(max_samples=self.buffer_size)

            yield samples
