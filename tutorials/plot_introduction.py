# -*- coding: utf-8 -*-
"""
.. _intro_tutorial:

Basic MEG and EEG data processing
=================================

.. image:: http://mne-tools.github.io/stable/_static/mne_logo.png

MNE-Python reimplements most of MNE-C's (the original MNE command line utils)
functionality and offers transparent scripting.
On top of that it extends MNE-C's functionality considerably
(customize events, compute contrasts, group statistics, time-frequency
analysis, EEG-sensor space analyses, etc.) It uses the same files as standard
MNE unix commands: no need to convert your files to a new system or database.

What you can do with MNE Python
-------------------------------

   - **Raw data visualization** to visualize recordings, can also use
     *mne_browse_raw* for extended functionality (see :ref:`ch_browse`)
   - **Epoching**: Define epochs, baseline correction, handle conditions etc.
   - **Averaging** to get Evoked data
   - **Compute SSP projectors** to remove ECG and EOG artifacts
   - **Compute ICA** to remove artifacts or select latent sources.
   - **Maxwell filtering** to remove environmental noise.
   - **Boundary Element Modeling**: single and three-layer BEM model
     creation and solution computation.
   - **Forward modeling**: BEM computation and mesh creation
     (see :ref:`ch_forward`)
   - **Linear inverse solvers** (dSPM, sLORETA, MNE, LCMV, DICS)
   - **Sparse inverse solvers** (L1/L2 mixed norm MxNE, Gamma Map,
     Time-Frequency MxNE)
   - **Connectivity estimation** in sensor and source space
   - **Visualization of sensor and source space data**
   - **Time-frequency** analysis with Morlet wavelets (induced power,
     intertrial coherence, phase lock value) also in the source space
   - **Spectrum estimation** using multi-taper method
   - **Mixed Source Models** combining cortical and subcortical structures
   - **Dipole Fitting**
   - **Decoding** multivariate pattern analyis of M/EEG topographies
   - **Compute contrasts** between conditions, between sensors, across
     subjects etc.
   - **Non-parametric statistics** in time, space and frequency
     (including cluster-level)
   - **Scripting** (batch and parallel computing)

What you're not supposed to do with MNE Python
----------------------------------------------

    - **Brain and head surface segmentation** for use with BEM
      models -- use Freesurfer.


.. note:: This package is based on the FIF file format from Neuromag. It
          can read and convert CTF, BTI/4D, KIT and various EEG formats to
          FIF.


Installation of the required materials
---------------------------------------

See :ref:`install_python_and_mne_python`.

.. note:: The expected location for the MNE-sample data is
    ``~/mne_data``. If you downloaded data and an example asks
    you whether to download it again, make sure
    the data reside in the examples directory and you run the script from its
    current directory.

    From IPython e.g. say::

        cd examples/preprocessing


    %run plot_find_ecg_artifacts.py

From raw data to evoked data
----------------------------

.. _ipython: http://ipython.scipy.org/

Now, launch `ipython`_ (Advanced Python shell) using the QT backend, which
is best supported across systems::

  $ ipython --matplotlib=qt

First, load the mne package:

.. note:: In IPython, you can press **shift-enter** with a given cell
          selected to execute it and advance to the next cell:
"""

import mne

##############################################################################
# If you'd like to turn information status messages off:

mne.set_log_level('WARNING')

##############################################################################
# But it's generally a good idea to leave them on:

mne.set_log_level('INFO')

##############################################################################
# You can set the default level by setting the environment variable
# "MNE_LOGGING_LEVEL", or by having mne-python write preferences to a file:

mne.set_config('MNE_LOGGING_LEVEL', 'WARNING', set_env=True)

##############################################################################
# Note that the location of the mne-python preferences file (for easier manual
# editing) can be found using:

mne.get_config_path()

##############################################################################
# By default logging messages print to the console, but look at
# :func:`mne.set_log_file` to save output to a file.
#
# Access raw data
# ^^^^^^^^^^^^^^^

from mne.datasets import sample  # noqa
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
print(raw_fname)

##############################################################################
# .. note:: The MNE sample dataset should be downloaded automatically but be
#           patient (approx. 2GB)
#
# Read data from file:

raw = mne.io.read_raw_fif(raw_fname)
print(raw)
print(raw.info)

##############################################################################
# Look at the channels in raw:

print(raw.ch_names)

##############################################################################
# Read and plot a segment of raw data

start, stop = raw.time_as_index([100, 115])  # 100 s to 115 s data segment
data, times = raw[:, start:stop]
print(data.shape)
print(times.shape)
data, times = raw[2:20:3, start:stop]  # access underlying data
raw.plot()

##############################################################################
# Save a segment of 150s of raw data (MEG only):

picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True,
                       exclude='bads')
raw.save('sample_audvis_meg_raw.fif', tmin=0, tmax=150, picks=picks,
         overwrite=True)

##############################################################################
# Define and read epochs
# ^^^^^^^^^^^^^^^^^^^^^^
#
# First extract events:

events = mne.find_events(raw, stim_channel='STI 014')
print(events[:5])

##############################################################################
# Note that, by default, we use stim_channel='STI 014'. If you have a different
# system (e.g., a newer system that uses channel 'STI101' by default), you can
# use the following to set the default stim channel to use for finding events:

mne.set_config('MNE_STIM_CHANNEL', 'STI101', set_env=True)

##############################################################################
# Events are stored as a 2D numpy array where the first column is the time
# instant and the last one is the event number. It is therefore easy to
# manipulate.
#
# Define epochs parameters:

event_id = dict(aud_l=1, aud_r=2)  # event trigger and conditions
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)

##############################################################################
# Exclude some channels (original bads + 2 more):

raw.info['bads'] += ['MEG 2443', 'EEG 053']

##############################################################################
# The variable raw.info['bads'] is just a python list.
#
# Pick the good channels, excluding raw.info['bads']:

picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True, stim=False,
                       exclude='bads')

##############################################################################
# Alternatively one can restrict to magnetometers or gradiometers with:

mag_picks = mne.pick_types(raw.info, meg='mag', eog=True, exclude='bads')
grad_picks = mne.pick_types(raw.info, meg='grad', eog=True, exclude='bads')

##############################################################################
# Define the baseline period:

baseline = (None, 0)  # means from the first instant to t = 0

##############################################################################
# Define peak-to-peak rejection parameters for gradiometers, magnetometers
# and EOG:

reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

##############################################################################
# Read epochs:

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=False, reject=reject)
print(epochs)

##############################################################################
# Get single epochs for one condition:

epochs_data = epochs['aud_l'].get_data()
print(epochs_data.shape)

##############################################################################
# epochs_data is a 3D array of dimension (55 epochs, 365 channels, 106 time
# instants).
#
# Scipy supports read and write of matlab files. You can save your single
# trials with:

from scipy import io  # noqa
io.savemat('epochs_data.mat', dict(epochs_data=epochs_data), oned_as='row')

##############################################################################
# or if you want to keep all the information about the data you can save your
# epochs in a fif file:

epochs.save('sample-epo.fif')

##############################################################################
# and read them later with:

saved_epochs = mne.read_epochs('sample-epo.fif')

##############################################################################
# Compute evoked responses for auditory responses by averaging and plot it:

evoked = epochs['aud_l'].average()
print(evoked)
evoked.plot()

##############################################################################
# .. topic:: Exercise
#
#   1. Extract the max value of each epoch

max_in_each_epoch = [e.max() for e in epochs['aud_l']]  # doctest:+ELLIPSIS
print(max_in_each_epoch[:4])  # doctest:+ELLIPSIS

##############################################################################
# It is also possible to read evoked data stored in a fif file:

evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked1 = mne.read_evokeds(
    evoked_fname, condition='Left Auditory', baseline=(None, 0), proj=True)

##############################################################################
# Or another one stored in the same file:

evoked2 = mne.read_evokeds(
    evoked_fname, condition='Right Auditory', baseline=(None, 0), proj=True)

##############################################################################
# Two evoked objects can be contrasted using :func:`mne.combine_evoked`.
# This function can use ``weights='equal'``, which provides a simple
# element-by-element subtraction (and sets the
# ``mne.Evoked.nave`` attribute properly based on the underlying number
# of trials) using either equivalent call:

contrast = mne.combine_evoked([evoked1, evoked2], weights=[0.5, -0.5])
contrast = mne.combine_evoked([evoked1, -evoked2], weights='equal')
print(contrast)

##############################################################################
# To do a weighted sum based on the number of averages, which will give
# you what you would have gotten from pooling all trials together in
# :class:`mne.Epochs` before creating the :class:`mne.Evoked` instance,
# you can use ``weights='nave'``:

average = mne.combine_evoked([evoked1, evoked2], weights='nave')
print(contrast)

##############################################################################
# Instead of dealing with mismatches in the number of averages, we can use
# trial-count equalization before computing a contrast, which can have some
# benefits in inverse imaging (note that here ``weights='nave'`` will
# give the same result as ``weights='equal'``):

epochs_eq = epochs.copy().equalize_event_counts(['aud_l', 'aud_r'])[0]
evoked1, evoked2 = epochs_eq['aud_l'].average(), epochs_eq['aud_r'].average()
print(evoked1)
print(evoked2)
contrast = mne.combine_evoked([evoked1, -evoked2], weights='equal')
print(contrast)

##############################################################################
# Time-Frequency: Induced power and inter trial coherence
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Define parameters:

import numpy as np  # noqa
n_cycles = 2  # number of cycles in Morlet wavelet
freqs = np.arange(7, 30, 3)  # frequencies of interest

##############################################################################
# Compute induced power and phase-locking values and plot gradiometers:

from mne.time_frequency import tfr_morlet  # noqa
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                        return_itc=True, decim=3, n_jobs=1)
power.plot([power.ch_names.index('MEG 1332')])

##############################################################################
# Inverse modeling: MNE and dSPM on evoked and raw data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Import the required functions:

from mne.minimum_norm import apply_inverse, read_inverse_operator  # noqa

##############################################################################
# Read the inverse operator:

fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
inverse_operator = read_inverse_operator(fname_inv)

##############################################################################
# Define the inverse parameters:

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"

##############################################################################
# Compute the inverse solution:

stc = apply_inverse(evoked, inverse_operator, lambda2, method)

##############################################################################
# Save the source time courses to disk:

stc.save('mne_dSPM_inverse')

##############################################################################
# Now, let's compute dSPM on a raw file within a label:

fname_label = data_path + '/MEG/sample/labels/Aud-lh.label'
label = mne.read_label(fname_label)

##############################################################################
# Compute inverse solution during the first 15s:

from mne.minimum_norm import apply_inverse_raw  # noqa
start, stop = raw.time_as_index([0, 15])  # read the first 15s of data
stc = apply_inverse_raw(raw, inverse_operator, lambda2, method, label,
                        start, stop)

##############################################################################
# Save result in stc files:

stc.save('mne_dSPM_raw_inverse_Aud')

##############################################################################
# What else can you do?
# ^^^^^^^^^^^^^^^^^^^^^
#
#     - detect heart beat QRS component
#     - detect eye blinks and EOG artifacts
#     - compute SSP projections to remove ECG or EOG artifacts
#     - compute Independent Component Analysis (ICA) to remove artifacts or
#       select latent sources
#     - estimate noise covariance matrix from Raw and Epochs
#     - visualize cross-trial response dynamics using epochs images
#     - compute forward solutions
#     - estimate power in the source space
#     - estimate connectivity in sensor and source space
#     - morph stc from one brain to another for group studies
#     - compute mass univariate statistics base on custom contrasts
#     - visualize source estimates
#     - export raw, epochs, and evoked data to other python data analysis
#       libraries e.g. pandas
#     - and many more things ...
#
# Want to know more ?
# ^^^^^^^^^^^^^^^^^^^
#
# Browse `the examples gallery <auto_examples/index.html>`_.

print("Done!")
