# Author: Mainak Jas
#
# License: BSD (3-clause)

import copy
import re
import threading
import time

import numpy as np

from .base_client import _BaseClient
from ..io import _empty_info
from ..io.pick import _picks_to_idx, pick_info
from ..io.constants import FIFF
from ..epochs import EpochsArray
from ..utils import logger, warn, fill_doc, deprecated
from ..externals.FieldTrip import Client as FtClient

RT_MSG = ('The realtime module is being deprecated from `mne-python` '
          'and moved to its own package, `mne-realtime`. '
          'To install, please use `$ pip install mne_realtime`.')


@fill_doc
@deprecated(RT_MSG)
class FieldTripClient(_BaseClient):
    """Realtime FieldTrip client.

    Parameters
    ----------
    info : dict | None
        The measurement info read in from a file. If None, it is guessed from
        the Fieldtrip Header object.
    host : str
        Hostname (or IP address) of the host where Fieldtrip buffer is running.
    port : int
        Port to use for the connection.
    wait_max : float
        Maximum time (in seconds) to wait for Fieldtrip buffer to start
    tmin : float | None
        Time instant to start receiving buffers. If None, start from the latest
        samples available.
    tmax : float
        Time instant to stop receiving buffers.
    buffer_size : int
        Size of each buffer in terms of number of samples.
    %(verbose)s
    """

    def __init__(self, info=None, host='localhost', port=1972, wait_max=30,
                 tmin=None, tmax=np.inf, buffer_size=1000, verbose=None):
        super().__init__(info, host, port, wait_max, tmin, tmax,
                         buffer_size, verbose)

    def __exit__(self, type, value, traceback):  # noqa: D105
        self.client.disconnect()

        return self

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
        ft_header = self.client.getHeader()
        last_samp = ft_header.nSamples - 1
        start = last_samp - n_samples + 1
        stop = last_samp
        events = np.expand_dims(np.array([start, 1, 1]), axis=0)

        # get the data
        data = self.client.getData([start, stop]).transpose()

        # create epoch from data
        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        info = pick_info(self.info, picks)
        return EpochsArray(data[picks][np.newaxis], info, events)

    def iter_raw_buffers(self):
        """Return an iterator over raw buffers.

        Returns
        -------
        raw_buffer : generator
            Generator for iteration over raw buffers.
        """
        # self.tmax_samp should be included
        iter_times = list(zip(
            list(range(self.tmin_samp, self.tmax_samp, self.buffer_size)),
            list(range(self.tmin_samp + self.buffer_size,
                       self.tmax_samp + 1, self.buffer_size))))
        last_iter_sample = iter_times[-1][1] if iter_times else self.tmin_samp
        if last_iter_sample < self.tmax_samp + 1:
            iter_times.append((last_iter_sample, self.tmax_samp + 1))

        for ii, (start, stop) in enumerate(iter_times):

            # wait for correct number of samples to be available
            self.client.wait(stop, np.iinfo(np.uint32).max,
                             np.iinfo(np.uint32).max)

            # get the samples (stop index is inclusive)
            raw_buffer = self.client.getData([start, stop - 1]).transpose()

            yield raw_buffer

            if self._recv_thread != threading.current_thread():
                # stop_receive_thread has been called
                break

    def _connect(self):
        self.client = FtClient()
        self.client.connect(self.host, self.port)

        # retrieve header
        logger.info("FieldTripClient: Retrieving header")
        while True:
            self.ft_header = self.client.getHeader()
            if self.ft_header is None:
                time.sleep(0.1)
            else:
                break
        logger.info("FieldTripClient: Header retrieved")

        return self

    def _disconnect(self):
        self.client.stop_receive_thread()

    def _create_info(self):
        """Create a minimal Info dictionary for epoching, averaging, etc."""
        if self.info is None:
            warn('Info dictionary not provided. Trying to guess it from '
                 'FieldTrip Header object')

            info = _empty_info(self.ft_header.fSample)  # create info

            # modify info attributes according to the FieldTrip Header object
            info['comps'] = list()
            info['projs'] = list()
            info['bads'] = list()

            # channel dictionary list
            info['chs'] = []

            # unrecognized channels
            chs_unknown = []

            for idx, ch in enumerate(self.ft_header.labels):
                this_info = dict()

                this_info['scanno'] = idx

                # extract numerical part of channel name
                this_info['logno'] = \
                    int(re.findall(r'[^\W\d_]+|\d+', ch)[-1])

                if ch.startswith('EEG'):
                    this_info['kind'] = FIFF.FIFFV_EEG_CH
                elif ch.startswith('MEG'):
                    this_info['kind'] = FIFF.FIFFV_MEG_CH
                elif ch.startswith('MCG'):
                    this_info['kind'] = FIFF.FIFFV_MCG_CH
                elif ch.startswith('EOG'):
                    this_info['kind'] = FIFF.FIFFV_EOG_CH
                elif ch.startswith('EMG'):
                    this_info['kind'] = FIFF.FIFFV_EMG_CH
                elif ch.startswith('STI'):
                    this_info['kind'] = FIFF.FIFFV_STIM_CH
                elif ch.startswith('ECG'):
                    this_info['kind'] = FIFF.FIFFV_ECG_CH
                elif ch.startswith('MISC'):
                    this_info['kind'] = FIFF.FIFFV_MISC_CH
                elif ch.startswith('SYS'):
                    this_info['kind'] = FIFF.FIFFV_SYST_CH
                else:
                    # cannot guess channel type, mark as MISC and warn later
                    this_info['kind'] = FIFF.FIFFV_MISC_CH
                    chs_unknown.append(ch)

                # Set coil_type (does FT supply this information somehow?)
                this_info['coil_type'] = FIFF.FIFFV_COIL_NONE

                # Fieldtrip already does calibration
                this_info['range'] = 1.0
                this_info['cal'] = 1.0

                this_info['ch_name'] = ch
                this_info['loc'] = np.zeros(12)

                if ch.startswith('EEG'):
                    this_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
                elif ch.startswith('MEG'):
                    this_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                else:
                    this_info['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN

                if ch.startswith('MEG') and ch.endswith('1'):
                    this_info['unit'] = FIFF.FIFF_UNIT_T
                elif ch.startswith('MEG') and (ch.endswith('2') or
                                               ch.endswith('3')):
                    this_info['unit'] = FIFF.FIFF_UNIT_T_M
                else:
                    this_info['unit'] = FIFF.FIFF_UNIT_V

                this_info['unit_mul'] = 0

                info['chs'].append(this_info)
                info._update_redundant()
                info._check_consistency()

            if chs_unknown:
                msg = ('Following channels in the FieldTrip header were '
                       'unrecognized and marked as MISC: ')
                warn(msg + ', '.join(chs_unknown))

        else:

            # XXX: the data in real-time mode and offline mode
            # does not match unless this is done
            self.info['projs'] = list()

            # FieldTrip buffer already does the calibration
            for this_info in self.info['chs']:
                this_info['range'] = 1.0
                this_info['cal'] = 1.0
                this_info['unit_mul'] = 0

            info = copy.deepcopy(self.info)

        return info

    def _enter_extra(self):
        self.ch_names = self.ft_header.labels

        # find start and end samples
        sfreq = self.info['sfreq']

        if self.tmin is None:
            self.tmin_samp = max(0, self.ft_header.nSamples - 1)
        else:
            self.tmin_samp = int(round(sfreq * self.tmin))

        if self.tmax != np.inf:
            self.tmax_samp = int(round(sfreq * self.tmax))
        else:
            self.tmax_samp = np.iinfo(np.uint32).max

        return self
