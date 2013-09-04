from mne.realtime import StimServer, StimClient
from nose.tools import assert_equal


def test_connection():
    """Test TCP/IP connection for StimServer <-> StimClient
    """
    stim_server = StimServer('localhost', port=4218)
    stim_server.start()

    stim_client = StimClient('localhost', port=4218)

    # Check if data is ok
    stim_server.add_trigger(20)
    assert_equal(stim_client.get_trigger(), 20)

    stim_server.shutdown()
