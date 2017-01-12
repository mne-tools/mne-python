# Author: Mainak Jas
#
# License: BSD (3-clause)

import copy
import re
import threading
import time

import numpy as np

from ..io import _empty_info
from ..io.pick import pick_info
from ..io.constants import FIFF
from ..epochs import EpochsArray
from ..utils import logger, warn
from ..externals.FieldTrip import Client as FtClient


def _buffer_recv_worker(ft_client):
    """Worker thread that constantly receives buffers."""
    try:
        for raw_buffer in ft_client.iter_raw_buffers():
            ft_client._push_raw_buffer(raw_buffer)
    except RuntimeError as err:
        # something is wrong, the server stopped (or something)
        ft_client._recv_thread = None
        print('Buffer receive thread stopped: %s' % err)


class FieldTripClient(object):
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
    verbose : bool, str, int, or None
        Log verbosity (see :func:`mne.verbose` and
        :ref:`Logging documentation <tut_logging>` for more).
    """

    def __init__(self, info=None, host='localhost', port=1972, wait_max=30,
                 tmin=None, tmax=np.inf, buffer_size=1000,
                 verbose=None):  # noqa: D102
        self.verbose = verbose

        self.info = info
        self.wait_max = wait_max
        self.tmin = tmin
        self.tmax = tmax
        self.buffer_size = buffer_size

        self.host = host
        self.port = port

        self._recv_thread = None
        self._recv_callbacks = list()

    def __enter__(self):  # noqa: D105
        # instantiate Fieldtrip client and connect
        self.ft_client = FtClient()

        # connect to FieldTrip buffer
        logger.info("FieldTripClient: Waiting for server to start")
        start_time, current_time = time.time(), time.time()
        success = False
        while current_time < (start_time + self.wait_max):
            try:
                self.ft_client.connect(self.host, self.port)
                logger.info("FieldTripClient: Connected")
                success = True
                break
            except:
                current_time = time.time()
                time.sleep(0.1)

        if not success:
            raise RuntimeError('Could not connect to FieldTrip Buffer')

        # retrieve header
        logger.info("FieldTripClient: Retrieving header")
        start_time, current_time = time.time(), time.time()
        while current_time < (start_time + self.wait_max):
            self.ft_header = self.ft_client.getHeader()
            if self.ft_header is None:
                current_time = time.time()
                time.sleep(0.1)
            else:
                break

        if self.ft_header is None:
            raise RuntimeError('Failed to retrieve Fieldtrip header!')
        else:
            logger.info("FieldTripClient: Header retrieved")

        self.info = self._guess_measurement_info()
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

    def __exit__(self, type, value, traceback):  # noqa: D105
        self.ft_client.disconnect()

    def _guess_measurement_info(self):
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
                this_info['logno'] = int(re.findall('[^\W\d_]+|\d+', ch)[-1])

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

                # Fieldtrip already does calibration
                this_info['range'] = 1.0
                this_info['cal'] = 1.0

                this_info['ch_name'] = ch
                this_info['loc'] = None

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

    def get_measurement_info(self):
        """Return the measurement info.

        Returns
        -------
        self.info : dict
            The measurement info.
        """
        return self.info

    def get_data_as_epoch(self, n_samples=1024, picks=None):
        """Return last n_samples from current time.

        Parameters
        ----------
        n_samples : int
            Number of samples to fetch.
        picks : array-like of int | None
            If None all channels are kept
            otherwise the channels indices in picks are kept.

        Returns
        -------
        epoch : instance of Epochs
            The samples fetched as an Epochs object.

        See Also
        --------
        mne.Epochs.iter_evoked
        """
        ft_header = self.ft_client.getHeader()
        last_samp = ft_header.nSamples - 1
        start = last_samp - n_samples + 1
        stop = last_samp
        events = np.expand_dims(np.array([start, 1, 1]), axis=0)

        # get the data
        data = self.ft_client.getData([start, stop]).transpose()

        # create epoch from data
        info = self.info
        if picks is not None:
            info = pick_info(info, picks)
        else:
            picks = range(info['nchan'])
        epoch = EpochsArray(data[picks][np.newaxis], info, events)

        return epoch

    def register_receive_callback(self, callback):
        """Register a raw buffer receive callback.

        Parameters
        ----------
        callback : callable
            The callback. The raw buffer is passed as the first parameter
            to callback.
        """
        if callback not in self._recv_callbacks:
            self._recv_callbacks.append(callback)

    def unregister_receive_callback(self, callback):
        """Unregister a raw buffer receive callback.

        Parameters
        ----------
        callback : callable
            The callback to unregister.
        """
        if callback in self._recv_callbacks:
            self._recv_callbacks.remove(callback)

    def _push_raw_buffer(self, raw_buffer):
        """Push raw buffer to clients using callbacks."""
        for callback in self._recv_callbacks:
            callback(raw_buffer)

    def start_receive_thread(self, nchan):
        """Start the receive thread.

        If the measurement has not been started, it will also be started.

        Parameters
        ----------
        nchan : int
            The number of channels in the data.
        """
        if self._recv_thread is None:

            self._recv_thread = threading.Thread(target=_buffer_recv_worker,
                                                 args=(self, ))
            self._recv_thread.daemon = True
            self._recv_thread.start()

    def stop_receive_thread(self, stop_measurement=False):
        """Stop the receive thread.

        Parameters
        ----------
        stop_measurement : bool
            Also stop the measurement.
        """
        if self._recv_thread is not None:
            self._recv_thread.stop()
            self._recv_thread = None

    def iter_raw_buffers(self):
        """Return an iterator over raw buffers.

        Returns
        -------
        raw_buffer : generator
            Generator for iteration over raw buffers.
        """
        iter_times = zip(range(self.tmin_samp, self.tmax_samp,
                               self.buffer_size),
                         range(self.tmin_samp + self.buffer_size - 1,
                               self.tmax_samp, self.buffer_size))

        for ii, (start, stop) in enumerate(iter_times):

            # wait for correct number of samples to be available
            self.ft_client.wait(stop, np.iinfo(np.uint32).max,
                                np.iinfo(np.uint32).max)

            # get the samples
            raw_buffer = self.ft_client.getData([start, stop]).transpose()

            yield raw_buffer
