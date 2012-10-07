import mne
from mne.fiff import Raw
# from mne.artifacts.ica import decompose_raw
from mne.artifacts.ica import ICA
from mne.viz import plot_ica_panel

from mne.datasets import sample
data_path = sample.data_path('examples/')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False,
                            exclude=raw.info['bads'])

start, stop = raw.time_to_index(100, 160)


ica = ICA(raw, picks, noise_cov=None, n_components=25, start=start, stop=stop,
          exclude=raw.info['bads'])

ica.fit_raw()

start_ica, stop_ica = raw.time_to_index(100, 103)

plot_ica_panel(ica, start=start_ica, stop=stop_ica)
