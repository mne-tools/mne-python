from mne.realtime import StimServer, StimClient
from mne.realtime import send_trigger_worker, recv_trigger_worker
import time
import Queue

#def test_connection():

stim_server = StimServer(port=4218, buffer_size=512)
stim_client = StimClient('localhost', port=4218, timeout=1.0)

# Instantiate queue for communication between threads
trig = Queue.Queue()
isi = Queue.Queue()

# Add stuff to the queue
trig.put(20)
#trig.put(50)
isi.put(0.1)
#isi.put(0.1)

# start the server
stim_server.start('localhost', stim_client, trig, isi)

# start the receive thread
stim_client.start_receive_thread()

# stop the receive thread
stim_client.stop_receive_thread()

stim_server.shutdown()
