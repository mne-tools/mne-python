import mne
from mne.fiff import Raw
# from mne.artifacts.ica import decompose_raw
from mne.artifacts.ica import ICA
from mne.viz import plot_ica_panel

from mne.datasets import sample
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, stim=False,
                            exclude=raw.info['bads'])

# 1 minute exposure should be sufficient for artifact detection
start, stop = raw.time_to_index(100, 160)

# setup ica seed
ica = ICA(raw, picks, noise_cov=None, n_components=25, start=start, stop=stop,
          exclude=raw.info['bads'])

# fit sourcs for raw data
ica.fit_raw()

# setup reasonable time window for inspection
start_ica, stop_ica = raw.time_to_index(100, 103)

# plot components
plot_ica_panel(ica, start=start_ica, stop=stop_ica)

# sign and order of components is non deterministic.
# However a distinct cardiac and one EOG component should be visible

raw_den = ica.denoise_raw(bads=[16, 8], make_raw=True)

tmin, tmax, event_id = -0.2, 0.5, 1
baseline = (None, 0)
reject = None
# reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

events = mne.find_events(raw_den, stim_channel='STI 014')

picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=False,
                            exclude=raw.info['bads'])

epochs_den = mne.Epochs(raw_den, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=False, reject=reject)

ev_den = epochs_den.average()



epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=False, reject=reject)

ev = epochs.average()

