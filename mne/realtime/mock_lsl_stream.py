# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
import time
from multiprocessing import Process

from numpy.random import rand
import numpy as np

from ..utils import _check_pylsl_installed


class MockLSLStream:
    """Mock LSL Stream.

    Parameters
    ----------
    host : str
        The LSL identifier of the server.
    n_channels : int
        Number of channels.
    ch_type : str
        The type of device that is being mocked.
    sfreq : float
        Sampling frequency of mock device.
    testing : bool
        Setting used to determine whether the data stream will be used
        for testing where the data is uniform and expected or with some
        random variation. Default is False.
    """

    def __init__(self, host, n_channels=8, ch_type="eeg", sfreq=100,
                 testing=False):
        self._host = host
        self._n_channels = n_channels
        self._ch_type = ch_type
        self._sfreq = sfreq
        self._testing = testing
        self._streaming = False

    def start(self):
        """Start a mock LSL stream."""
        print("now sending data...")
        self.process = Process(target=self._initiate_stream)
        self.process.daemon = True
        self.process.start()

        return self

    def stop(self):
        """Stop a mock LSL stream."""
        self._streaming = False
        self.process.terminate()

        print("Stopping stream...")

        return self

    def _initiate_stream(self):
        # outlet needs to be made on the same process
        pylsl = _check_pylsl_installed(strict=True)
        self._streaming = True
        info = pylsl.StreamInfo(name='MNE', type=self._ch_type.upper(),
                                channel_count=self._n_channels,
                                nominal_srate=self._sfreq,
                                channel_format='float32', source_id=self._host)
        info.desc().append_child_value("manufacturer", "MNE")
        channels = info.desc().append_child("channels")
        for c_id in range(1, self._n_channels + 1):
            channels.append_child("channel") \
                    .append_child_value("label", "MNE {:03d}".format(c_id)) \
                    .append_child_value("type", self._ch_type.lower()) \
                    .append_child_value("unit", "microvolt")

        # next make an outlet
        outlet = pylsl.StreamOutlet(info)

        # let's make some data
        counter = 0
        trigger = 0
        while self._streaming:
            sample = counter % self._sfreq
            # let's bound trigger to be between 1 and 10 so the max cycle
            # is ten seconds
            trigger = 0 if trigger == 10 else trigger
            if sample == 0:
                trigger += 1

            if self._testing:
                const = trigger
            else:
                const = np.sin(2 * np.pi * sample / self._sfreq) * 1e-6

            mysample = rand(self._n_channels).dot(const).tolist()
            # now send it and wait for a bit
            outlet.push_sample(mysample)
            counter += 1
            time.sleep(self._sfreq**-1)
