# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import numpy as np

from .base_client import _BaseClient
from ..epochs import EpochsArray
from ..io.meas_info import create_info
from ..io.pick import _picks_to_idx, pick_info
from ..utils import fill_doc

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
    wait_max : float
        Maximum time (in seconds) to wait for real-time buffer to start
    tmin : float | None
        Time instant to start receiving buffers. If None, start from the latest
        samples available.
    tmax : float
        Time instant to stop receiving buffers.
    buffer_size : int
        Size of each buffer in terms of number of samples.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    """

    def __init__(self, identifier, port=None, wait_max=10., tmin=None,
                 tmax=np.inf, buffer_size=1000, verbose=None):  # noqa: D102
        super(LSLClient, self).__init__(identifier, port, tmin, tmax,
                                        buffer_size, wait_max, verbose)

    @fill_doc
    def get_data_as_epoch(self, n_samples=1024, picks=None):
        """Return last n_samples from current time.

        Parameters
        ----------
        n_samples : int
            Number of samples to fetch.
        %(picks_all)s

        Returns
        -------
        epoch : instance of Epochs
            The samples fetched as an Epochs object.

        See Also
        --------
        mne.Epochs.iter_evoked
        """
        wait_time = n_samples * 5. / self.info['sfreq']

        # create an event at the start of the data collection
        events = np.expand_dims(np.array([0, 1, 1]), axis=0)
        samples, _ = self.client.pull_chunk(max_samples=n_samples,
                                            timeout=wait_time)
        data = np.vstack(samples).T

        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        info = pick_info(self.info, picks)
        return EpochsArray(data[picks][np.newaxis], info, events)

    def iter_raw_buffers(self):
        """Return an iterator over raw buffers."""
        inlet = pylsl.StreamInlet(self.client)

        while True:
            samples, _ = inlet.pull_chunk(max_samples=self.buffer_size)

            yield np.vstack(samples).T

    def _connect(self):
        stream_info = pylsl.resolve_byprop('source_id', self.identifier,
                                           timeout=1)[0]
        self.client = pylsl.StreamInlet(info=stream_info,
                                        max_buflen=self.buffer_size)

        return self

    def _create_info(self):
        sfreq = self.client.info().nominal_srate()

        lsl_info = self.client.info()
        ch_info = lsl_info.desc().child("channels").child("channel")
        ch_names = list()
        ch_types = list()
        ch_type = lsl_info.type()
        for k in range(1,  lsl_info.channel_count()+1):
            ch_names.append(ch_info.child_value("label")
                            or '{} {:03d}'.format(ch_type.upper(), k))
            ch_types.append(ch_info.child_value("type")
                            or ch_type.lower())
            ch_info = ch_info.next_sibling()

        info = create_info(ch_names, sfreq, ch_types)

        self.info = info

        return self

    def _disconnect(self):
        self.client.close_stream()

        return self
