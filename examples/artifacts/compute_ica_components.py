import mne
from mne.fiff import Raw
from mne.artifacts.ica import decompose_raw

from mne.datasets import sample
data_path = sample.data_path('examples/')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname)

picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False)

start, stop = raw.time_to_index(100, 115)

sources, mix = decompose_raw(raw, noise_cov=None, n_components=25, start=start,
                             stop=stop, exclude=raw.info['bads'])

