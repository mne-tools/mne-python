# author: ngayraud
#
# Created on Mon Mar 12 15:53:38 2018.

import warnings
import numpy as np
from numpy.testing import assert_equal
from nose.tools import assert_true, assert_raises

from mne.simulation import Simulation, simulate_raw_signal
from mne import (datasets, read_forward_solution, pick_types_forward,
                 convert_forward_solution, read_label)

from mne.utils import run_tests_if_main
from mne.datasets import testing


def _same(times, fwd_f, n_dipoles, labels, location, subject, subjects_dir,
          waveform, events):
    sim_1 = Simulation(fwd_f, n_dipoles=n_dipoles, labels=labels,
                       location=location, subject=subject,
                       subjects_dir=subjects_dir, waveform=waveform)
    raw_1 = simulate_raw_signal(sim_1, times, cov=None, events=events,
                                verbose=0)
    data_1, _ = raw_1[:]
    sim_2 = Simulation(fwd_f, n_dipoles=n_dipoles, labels=labels,
                       location=location, subject=subject,
                       subjects_dir=subjects_dir, waveform=waveform)
    raw_2 = simulate_raw_signal(sim_2, times, cov=None, events=events,
                                verbose=0)
    data_2, _ = raw_2[:]

    # This should work
    assert_equal(data_1, data_2)
    assert_true(~np.isnan(np.sum(data_1)) or ~np.isinf(np.sum(data_1)))
    assert_true(~np.isnan(np.sum(data_2)) or ~np.isinf(np.sum(data_1)))


def _without_events(fwd_f, labels, subject, subjects_dir):

    freq = 256.0
    time = 2.0
    times = np.arange(0, time, 1.0 / freq)

    # Should work
    _same(times, fwd_f, n_dipoles=2, labels=labels, location='center',
          subject=subject, subjects_dir=subjects_dir, waveform='sin',
          events=None)
    _same(times, fwd_f, n_dipoles=2, labels=labels, location='center',
          subject=subject, subjects_dir=subjects_dir,
          waveform=['p300_target', 'sin'], events=None)

    # These should raise warnings
    warnings.simplefilter("error")
    # different labels and dipoles
    assert_raises(RuntimeWarning, Simulation, fwd_f, n_dipoles=1,
                  labels=labels, waveform='sin')
    assert_raises(RuntimeWarning, Simulation, fwd_f, n_dipoles=3,
                  labels=labels, waveform='sin')
    # different waveforms and dipoles
    assert_raises(RuntimeWarning, Simulation, fwd_f, n_dipoles=2,
                  labels=labels, waveform=['sin', 'p300_target', 'sin'])
    assert_raises(ValueError, Simulation, fwd_f, n_dipoles=2,
                  labels=labels, waveform=['sin'])
    # different window times and waveforms
    assert_raises(RuntimeWarning, Simulation, fwd_f, n_dipoles=2,
                  labels=labels, subjects_dir=subjects_dir,
                  waveform=['sin', 'p300_target'],
                  window_times=['all', 'all', 'all'])
    assert_raises(RuntimeWarning, Simulation, fwd_f, n_dipoles=2,
                  labels=labels, subjects_dir=subjects_dir,
                  waveform=['sin', 'p300_target'], window_times=['all'])
    warnings.simplefilter("always")


def _with_events(fwd_f, labels, subject, subjects_dir):

    freq = 256.0
    waveform = ['p300_target', 'sin']
    time = 2.0
    times = np.arange(0, time, 1.0 / freq)
    window_times = np.arange(0, time / 3.0, 1.0 / freq)
    indices = np.array([0, np.floor(len(times) / 4)])
    events = np.zeros((len(indices), 3))
    events[:, 0] = indices
    events[:, 2] = 1

    # These should work
    _same(times, fwd_f, n_dipoles=2, labels=labels, location='center',
          subject=subject, subjects_dir=subjects_dir,
          waveform=waveform, events=[np.array(events)] * 2)
    _same(times, fwd_f, n_dipoles=2, labels=labels, location='center',
          subject=subject, subjects_dir=subjects_dir,
          waveform=waveform, events=events)

    # These should raise warnings
    warnings.simplefilter("error")

    # wrong number of events
    sim = Simulation(fwd_f, n_dipoles=2, labels=labels, location='center',
                     subject=subject, subjects_dir=subjects_dir,
                     waveform=waveform)
    assert_raises(RuntimeWarning, simulate_raw_signal, sim, times, cov=0,
                  events=[events], verbose=0)
    assert_raises(RuntimeWarning, simulate_raw_signal, sim, times, cov=0,
                  events=[events, events, events], verbose=0)
    # Too large window
    window_times = np.arange(0, time + 1.0, 1.0 / freq)
    sim = Simulation(fwd_f, n_dipoles=2, labels=labels, location='center',
                     subject=subject, subjects_dir=subjects_dir,
                     waveform=waveform, window_times=['all', window_times])
    assert_raises(RuntimeWarning, simulate_raw_signal, sim, times, cov=0,
                  events=events, verbose=0)
    # add an index that is too large
    events[0, 0] = len(times)
    sim = Simulation(fwd_f, n_dipoles=2, labels=labels, location='center',
                     subject=subject, subjects_dir=subjects_dir,
                     waveform=waveform)
    assert_raises(RuntimeWarning, simulate_raw_signal, sim, times, cov=0,
                  events=events, verbose=0)

    warnings.simplefilter("always")

# Initialize
data_path = data_path = datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
subjects_dir = data_path + '/subjects'
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'

subject = 'sample'


@testing.requires_testing_data
def test_simulate_raw():
    """Test simulation of raw data.
    """
    # Get forward model

    fwd = read_forward_solution(fwd_fname)
    fwd = pick_types_forward(fwd, meg=True, eeg=True)
    fwd_f = convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                     use_cps=True)

    label_names = ['Aud-lh', 'Aud-rh']

    label_names = ['Vis-lh', 'Vis-rh']
    labels = [read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
              for ln in label_names]
    for label in labels:
        label.values[:] = 1

    _without_events(fwd_f, labels, subject, subjects_dir)
    _with_events(fwd_f, labels, subject, subjects_dir)
    # TODO: add test for noise
run_tests_if_main()
