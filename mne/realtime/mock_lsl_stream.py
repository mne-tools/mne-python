# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
import time
from multiprocessing import Process

from ..utils import _check_pylsl_installed
from ..io import constants


class MockLSLStream:
    """Mock LSL Stream.

    Parameters
    ----------
    host : str
        The LSL identifier of the server.
    raw : instance of Raw object
        An instance of Raw object to be streamed.
    ch_type : str
        The type of data that is being streamed.
    time_dilation : int
        A scale factor to speed up or slow down the rate of
        the data being streamed.
    """

    def __init__(self, host, raw, ch_type, time_dilation=1):
        self._host = host
        self._ch_type = ch_type
        self._time_dilation = time_dilation

        raw.load_data().pick(ch_type)
        self._raw = raw
        self._sfreq = int(self._raw.info['sfreq'])

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
                                channel_count=self._raw.info['nchan'],
                                nominal_srate=self._sfreq,
                                channel_format='float32', source_id=self._host)
        info.desc().append_child_value("manufacturer", "MNE")
        channels = info.desc().append_child("channels")
        for ch in self._raw.info['chs']:
            unit = ch['unit']
            keys, values = zip(*list(constants.FIFF.items()))
            unit = keys[values.index(unit)]
            channels.append_child("channel") \
                    .append_child_value("label", ch['ch_name']) \
                    .append_child_value("type", self._ch_type.lower()) \
                    .append_child_value("unit", unit)

        # next make an outlet
        outlet = pylsl.StreamOutlet(info)

        # let's make some data
        counter = 0
        while self._streaming:
            mysample = self._raw[:, counter][0].ravel().tolist()
            # now send it and wait for a bit
            outlet.push_sample(mysample)
            counter += 1
            time.sleep(self._time_dilation / self._sfreq)
