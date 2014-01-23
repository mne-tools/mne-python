"""
==============================================
Compute effect-matched-spatial filtering (EMS)
==============================================

This example computes the EMS to reconstruct the time course of
the experimental effect as described in:

Aaron Schurger, Sebastien Marti, and Stanislas Dehaene, "Reducing multi-sensor
data to a single time course that reveals experimental effects",
BMC Neuroscience 2013, 14:122
"""

# Author: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)


print(__doc__)

import os.path as op
import numpy as np

import mne
from mne import fiff
from mne.datasets import sample
from mne.epochs import combine_event_ids
data_path = sample.data_path()

# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_ids = {'AudL': 1, 'VisL': 2}
tmin = -0.2
tmax = 0.5

#   Setup for reading the raw data
raw = fiff.Raw(raw_fname, preload=True)
raw.filter(1, 45)
events = mne.read_events(event_fname)

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick EEG channels
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                        include=include, exclude='bads')
# Read epochs

reject = dict(grad=4000e-13, eog=150e-6)
# reject = dict(mag=4e-12, eog=150e-6)

epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, picks=picks,
                    baseline=None, reject=reject)

# Let's equalize the trial counts in each condition
epochs.equalize_event_counts(['AudL', 'VisL'], copy=False)
# Now let's combine some conditions

picks2 = fiff.pick_types(epochs.info, meg='grad', exclude='bads')

data = epochs.get_data()[:, picks2, :]

# the matlab routine expects n_sensors, n_times, n_epochs

data2 = np.transpose(data, [1, 2, 0])

# # create bool indices
conditions = [epochs.events[:, 2] == 1, epochs.events[:, 2] == 2]

# # matlab io functions don't deal with bool values
# # so we need tom make a detour via int
conditions = [c.astype(int) for c in conditions]


###############################################################################
# Now it's time for some hacking ...

from scipy import io

io.savemat('epochs_data.mat', {'data': data2,
                               'conditions': conditions})

var_name1, var_name2 = 'surrogates', 'spatial_filter'
my_pwd = op.abspath(op.curdir)  # expand path

# this requires
# https://gist.github.com/dengemann/640d202f84befff1545d
# in the local directory

my_matlab_code = """
disp('reading data ...');
epochs = load('epochs_data.mat');
conditions = boolean(epochs.conditions');
disp('computing trial surrogates');
[{0}, {1}] = ems_ncond(epochs.data, conditions);
disp('saving results ...');
save('{pwd}/{0}.mat', '{0}');
save('{pwd}/{1}.mat', '{1}');
quit;
""".format(var_name1, var_name2, pwd=my_pwd).strip('\n').replace('\n', '')

run_matlab = ['matlab', '-nojvm', '-nodesktop', '-nodisplay', '-r']

run_matlab.append(my_matlab_code)

from subprocess import Popen, PIPE

process = Popen(run_matlab, stdin=PIPE, stdout=None, shell=False)

process.communicate()  # call and quit matlab

surrogates = io.loadmat(var_name1 + '.mat')[var_name1]
spatial_filter = io.loadmat(var_name2 + '.mat')[var_name2]

from mne.decoding import compute_ems

surrogates_py, spatial_filter_py = compute_ems(data, conditions)

iter_comparisons = [
    (surrogates, spatial_filter),
    (surrogates_py, spatial_filter_py)
]

from numpy.testing import asser_array_almost_equal

asser_array_almost_equal(surrogates, surrogates_py)
asser_array_almost_equal(spatial_filter, spatial_filter_py)


import matplotlib.pyplot as plt

for ii, (tsurrogate, sfilter) in enumerate(iter_comparisons):

    lang = 'python' if ii > 0 else 'matlab'

    order = epochs.events[:, 2].argsort()
    times = epochs.times * 1e3

    plt.figure()
    plt.title('single surrogate trial - %s' % lang)
    plt.imshow(tsurrogate[order], origin='lower', aspect='auto',
               extent=[times[0], times[-1], 1, len(epochs)])
    plt.xlabel('Time (ms)')
    plt.ylabel('Trials (reordered by condition)')
    plt.savefig('fig-%s-1.png' % lang)

    plt.figure()
    plt.title('Average EMS signal - %s' % lang)
    for key, value in epochs.event_id.items():
        ems_ave = tsurrogate[epochs.events[:, 2] == value]
        ems_ave /= 4e-11
        plt.plot(times, ems_ave.mean(0), label=key)
    plt.xlabel('Time (ms)')
    plt.ylabel('fT/cm')
    plt.legend(loc='best')
    plt.savefig('fig-%s-2.png' % lang)

    # visualize spatial filter
    evoked = epochs.average()
    evoked.data = sfilter
    evoked.plot_topomap(ch_type='grad', title=lang)
    plt.savefig('fig-%s-3.png' % lang)
