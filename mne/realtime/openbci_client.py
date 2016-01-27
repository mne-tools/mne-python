# Authors: Romain Trachel <trachelr@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
# License: BSD (3-clause)

import re
import copy
import time
import threading
import warnings
import numpy as np

from ..io.constants import FIFF
from ..io.meas_info import _empty_info
from ..io.pick import pick_info
from ..epochs import EpochsArray
from ..utils import logger
from ..externals.OpenBCI import OpenBCIBoard, OpenBCISample


def _buffer_recv_worker(client):
    """Worker thread that constantly receives buffers."""

    try:
        for obci_sample in client.iter_obci_sample():
            client._push_obci_sample(obci_sample)
    except RuntimeError as err:
        # something is wrong, the server stopped (or something)
        client._recv_thread = None
        print('Buffer receive thread stopped: %s' % err)


class OpenBCIClient(object):
    """ Realtime OpenBCI client

    Parameters
    ----------
    info : dict | None
        The measurement info read in from a file. If None, it is guessed from
        the Fieldtrip Header object.
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
        Log verbosity see mne.verbose.
    """
    def __init__(self, info=None, port=5000, wait_max=30,
                 tmin=None, tmax=np.inf, buffer_size=1000, verbose=True):
        self.verbose = verbose

        self.info = info
        self.wait_max = wait_max
        self.tmin = tmin
        self.tmax = tmax
        self.buffer_size = buffer_size

        self.port = port

        self._recv_thread = None
        self._recv_callbacks = list()

    def __enter__(self):
        # instantiate OpenBCI client and connect
        self.client = OpenBCIBoard(port=self.port, log=self.verbose)

        start_time, current_time = time.time(), time.time()
        success = False
        while current_time < (start_time + self.wait_max):
            try:
                self.client.check_connection()
                logger.info("OpenBCIClient: Connected")
                success = True
                break
            except:
                current_time = time.time()
                time.sleep(0.1)

        if not success:
            raise RuntimeError('Could not connect to OpenBCI')

        self.sfreq = self.client.getSampleRate()

        self.info = self._guess_measurement_info()

        self.client.ser.write('b')
        self.client.streaming = True
        self.client.check_connection()
        return self

    def __exit__(self, type, value, traceback):
        self.client.disconnect()

    def _guess_measurement_info(self):
        """
        Creates a minimal Info dictionary required for epoching, averaging
        et al.
        """

        if self.info is None:

            warnings.warn('Info dictionary not provided. Trying to guess it '
                          'from OpenBCIBoard object')

            info = _empty_info(self.sfreq)  # create info dictionary

            # modify info attributes according to the OpenBCIBoard object
            n_eeg = self.client.getNbEEGChannels()
            n_aux = self.client.getNbAUXChannels()
            # EEG + Auxiliary + 1 Stim channel
            info['nchan'] = n_eeg + n_aux + 1
            info['sfreq'] = self.client.getSampleRate()

            # NEED TO BE CHECKED
            # ch_names = ['Time']
            ch_names = ['EEG%i' % i for i in range(1, n_eeg + 1)]
            # adding auxiliary channels
            ch_names += ['MISC%i' % i for i in range(1, n_aux + 1)]
            # adding stimulation channel (for compatibility with RtEpochs)
            ch_names += ['STIM0']
            info['ch_names'] = ch_names

            info['comps'] = list()
            info['projs'] = list()
            info['bads'] = list()

            # channel dictionary list
            info['chs'] = []

            for idx, ch in enumerate(info['ch_names']):
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
        """Returns the measurement info.

        Returns
        -------
        self.info : dict
            The measurement info.
        """
        return self.info

    def get_data_as_epoch(self, n_samples=256, picks=None):
        """Returns last n_samples from current time.

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
        Epochs.iter_evoked
        """
        # create the data
        data = np.zeros([self.info['nchan'], n_samples])
        n_eeg = self.client.getNbEEGChannels()
        for i in range(n_samples):
            sample = self.client._read_serial_binary()
            if self.client.daisy:
                # odd sample: daisy sample, save for later
                if ~sample.id % 2:
                    self.client.last_odd_sample = sample
                    # even sample: concatenate and send if last sample was
                    # the fist part, otherwise drop the packet
                elif sample.id - 1 == self.client.last_odd_sample.id:
                    # the aux data will be the average between the two samples,
                    # as the channel samples themselves have been averaged by the board
                    avg_aux_data = list((np.array(sample.aux_data) + np.array(self.client.last_odd_sample.aux_data))/2)
                    sample = OpenBCISample(sample.id, sample.channel_data + self.client.last_odd_sample.channel_data, avg_aux_data)
            # EEG data
            data[:n_eeg, i] = sample.channel_data
            # Auxiliary data
            data[n_eeg:-1, i] = sample.aux_data
            # TODO: Stimulation data (i.e. triggers)

        events = np.expand_dims(np.array([0, 1, 1]), axis=0)
        # create epoch from data
        info = self.info
        if picks is not None:
            info = pick_info(info, picks, copy=True)
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
        """Unregister a raw buffer receive callback

        Parameters
        ----------
        callback : callable
            The callback to unregister.
        """
        if callback in self._recv_callbacks:
            self._recv_callbacks.remove(callback)

    def _push_obci_sample(self, obci_sample):
        """Push raw buffer to clients using callbacks."""
        for callback in self._recv_callbacks:
            callback(obci_sample)

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
        """Stop the receive thread

        Parameters
        ----------
        stop_measurement : bool
            Also stop the measurement.
        """
        if self._recv_thread is not None:
            self._recv_thread.stop()
            self._recv_thread = None

    def iter_obci_sample(self):
        """Return an iterator over OpenBCISample

        Returns
        -------
        sample : generator
            Generator for iteration over OpenBCI samples.
        """

        while True:
            # get the samples
            sample = self.client._read_serial_binary()
            if self.client.daisy:
                # odd sample: daisy sample, save for later
                if ~sample.id % 2:
                    self.client.last_odd_sample = sample
                    # even sample: concatenate and send if last sample was
                    # the fist part, otherwise drop the packet
                elif sample.id - 1 == self.client.last_odd_sample.id:
                    # the aux data will be the average between the two samples,
                    # as the channel samples themselves have been averaged by the board
                    avg_aux_data = list((np.array(sample.aux_data) + np.array(self.client.last_odd_sample.aux_data))/2)
                    sample = OpenBCISample(sample.id, sample.channel_data + self.client.last_odd_sample.channel_data, avg_aux_data)

            yield sample
