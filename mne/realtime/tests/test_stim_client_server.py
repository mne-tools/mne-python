from mne.realtime import StimServer, StimClient

#def test_connection():

stim_server = StimServer(port=4218)
stim_client = StimClient('localhost', port=4218)

# start the server
stim_server.start('localhost', stim_client)

stim_server.add_trigger(20)
stim_server.add_trigger(50)

trig1 = stim_client.get_trigger()

stim_server.add_trigger(100)

trig2 = stim_client.get_trigger()
trig3 = stim_client.get_trigger()

# Should give timeout error
trig4 = stim_client.get_trigger()

stim_server.shutdown()
