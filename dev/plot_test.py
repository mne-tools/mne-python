

import os
import matplotlib.pyplot as plt

import mne


boxy_data_folder = mne.datasets.boxy_example.data_path()
boxy_raw_dir = os.path.join(boxy_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()

### plot the raw data ###
raw_intensity.plot(n_channels=10)
