# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
import time
from multiprocessing import Process
from numpy.random import rand

from ..utils import fill_doc, _check_pylsl_installed


class MockLSLStream:
    """Mock LSL Stream

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
    """

    def __init__(self, host, n_channels=8, ch_type="eeg", sfreq=100):
        self.host = host
        self.n_channels = n_channels
        self.ch_type = ch_type
        self.sfreq = sfreq
        self.streaming = False
        self._test = None

    def start(self):
        """Starts a mock LSL stream."""
        pylsl = _check_pylsl_installed(strict=True)
        self.streaming = True
        info = pylsl.StreamInfo('MNE', self.ch_type.upper(), self.n_channels,
                                self.sfreq, 'float32', self.host)
        info.desc().append_child_value("manufacturer", "MNE")
        channels = info.desc().append_child("channels")
        for c_id in range(1, self.n_channels + 1):
            channels.append_child("channel") \
                    .append_child_value("label", "MNE {:03d}".format(c_id)) \
                    .append_child_value("type", self.ch_type.lower()) \
                    .append_child_value("unit", "microvolt")

        # next make an outlet
        outlet = pylsl.StreamOutlet(info)

        print("now sending data...")
        self.process = Process(target=self._initiate_stream, args=(outlet,))
        self.process.daemon = True
        self.process.start()

        return self

    def close(self):
        """Stops a mock LSL stream."""
        self.process.terminate()

        print("Stopping stream...")

        return self

    def _initiate_stream(self, outlet):
        counter = 0
        while True:
            sample = counter % 40
            const = np.sin(2 * np.pi * sample / 40)
            mysample = rand(self.n_channels).dot(1e-6).tolist()
            # now send it and wait for a bit
            outlet.push_sample(mysample)
            counter += 1
            time.sleep(self.sfreq**-1)
