import numpy as np
import pylab as pl
import mne
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator


data_path = sample.data_path('..')
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

event_id, tmin, tmax = 1, -0.2, 0.5
snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
inverse_operator = read_inverse_operator(fname_inv)
label = mne.read_label(fname_label)
raw = Raw(fname_raw)
events = mne.read_events(fname_event)

# Set up pick list
include = []
exclude = raw.info['bads'] + ['EEG 053']  # bads + 1 more

# pick MEG channels
picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                                            include=include, exclude=exclude)
# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# Compute inverse solution and stcs for each epoch
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, label,
                            pick_normal=True)


##############################################################################
from mne import ndvar
reload(mne.dimensions)
reload(ndvar)
Y = ndvar.from_stc(stcs)
Y = ndvar.resample(Y, 50)

pl.figure()
# extract the mean in a label
Ylbl = Y.summary(source=label)

# plot all cases; don't worry about the axes in the object, just be explicit:
pl.plot(Ylbl.time.times, Ylbl.get_data(('time', 'case')), color=(.5, .5, .5))

# plot the mean across cases
Ylblm = Ylbl.summary('case')
pl.plot(Ylblm.time.times, Ylblm.get_data('time'), 'r-', linewidth=2)

# plot a spcific case
pl.plot(Ylblm.time.times, Ylbl[1].get_data('time'), color=(1, .5, 0))

# or write a plot-function
def plot_uts(Y, **kwargs):
    if Y.has_case:
        x = Y.get_data(('time', 'case'))
    else:
        x = Y.get_data(('time',))
    pl.plot(Y.time.times, x, **kwargs)
    pl.xlabel('Time (s)')

pl.figure()
plot_uts(Ylbl, color='gray')
plot_uts(Ylbl[1], color='orange')
plot_uts(Ylbl.summary('case'), color='red')

pl.show()
