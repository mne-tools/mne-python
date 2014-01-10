# Authors: Mainak Jas <mainak@neuro.hut.fi>
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.realtime import FtClient, MneFtClient, RtEpochs

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.fiff.Raw(raw_fname, preload=True)

ftc = FtClient()
ftc.connect('localhost', 1972)

# select gradiometers
picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=raw.info['bads'])

# select the left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5

# create the MNEFtClient object
rt_client = MneFtClient(ft_client=ftc, raw=raw)

# create the real-time epochs object
rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                     decim=1, reject=dict(grad=4000e-13, eog=150e-6))

# start the acquisition
rt_epochs.start()

# send raw buffers
rt_client.send_data(rt_epochs, picks, tmin=0, tmax=150, buffer_size=1000)

ftc.disconnect()
