# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Mainak Jas <mainakjas@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from .base_client import _BaseClient
from ..epochs import EpochsArray
from ..io.meas_info import create_info
from ..io.pick import _picks_to_idx, pick_info
from ..utils import fill_doc, _check_pylsl_installed, deprecated

RT_MSG = ('The realtime module is being deprecated from `mne-python` '
          'and moved to its own package, `mne-realtime`. '
          'To install, please use `$ pip install mne_realtime`.')


@deprecated(RT_MSG)
class LSLClient(_BaseClient):
    """LSL Realtime Client.

    Parameters
    ----------
    info : instance of mne.Info | None
        The measurement info read in from a file. If None, it is generated from
        the LSL stream. This method may result in less info than expected. If
        the channel type is EEG, the `standard_1005` montage is used for
        electrode location.
    host : str
        The LSL identifier of the server. This is the source_id designated
        when the LSL stream was created. Make sure the source_id is unique on
        the LSL subnet. For more information on LSL, please check the
        docstrings on `StreamInfo` and `StreamInlet` in the pylsl.
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
        # set up timeout in case LSL process hang. wait arb 5x expected time
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
        while True:
            samples, _ = self.client.pull_chunk(max_samples=self.buffer_size)

            yield np.vstack(samples).T

    def _connect(self):
        # To use this function with an LSL stream which has a 'name' but no
        # 'source_id', change the keyword in pylsl.resolve_byprop accordingly.
        pylsl = _check_pylsl_installed(strict=True)
        stream_info = pylsl.resolve_byprop('source_id', self.host,
                                           timeout=self.wait_max)[0]
        self.client = pylsl.StreamInlet(info=stream_info,
                                        max_buflen=self.buffer_size)

        return self

    def _create_info(self):
        montage = None
        sfreq = self.client.info().nominal_srate()

        lsl_info = self.client.info()
        ch_info = lsl_info.desc().child("channels").child("channel")
        ch_names = list()
        ch_types = list()
        ch_type = lsl_info.type().lower()
        for k in range(1,  lsl_info.channel_count() + 1):
            ch_names.append(ch_info.child_value("label") or
                            '{} {:03d}'.format(ch_type.upper(), k))
            ch_types.append(ch_info.child_value("type") or ch_type)
            ch_info = ch_info.next_sibling()
        if ch_type == "eeg":
            try:
                montage = 'standard_1005'
                info = create_info(ch_names, sfreq, ch_types, montage=montage)
            except ValueError:
                info = create_info(ch_names, sfreq, ch_types)

        return info

    def _disconnect(self):
        self.client.close_stream()

        return self
