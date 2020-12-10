#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:33:57 2020

@author: Jeff Stout

This code only tests for a specific bug found in the code.  Further test
functions should be written to test the full simulation code.
"""


import mne
import os.path as op
import numpy as np
from mne.datasets import sample
import copy


def test_simulation_cascade():
    '''Test to verify cascading operations do not overwrite data within \
    raw._data matrix.  This should fail in ver 0.21'''

    # Path Information
    data_path = sample.data_path()
    subject = 'sample'
    meg_path = op.join(data_path, 'MEG', subject, 'sample_audvis_raw.fif')

    # Create 10 second raw dataset with zeros in the data matrix
    raw_null = mne.io.read_raw_fif(meg_path)
    raw_null.crop(0, 10)
    raw_null.load_data()
    raw_null.pick_types(meg=True)
    raw_null._data = np.zeros(raw_null._data.shape)  # Zero out data structure

    # Calculate independent signal additions
    raw_eog = copy.deepcopy(raw_null)
    mne.simulation.add_eog(raw_eog, random_state=0)

    raw_ecg = copy.deepcopy(raw_null)
    mne.simulation.add_ecg(raw_ecg, random_state=0)

    raw_noise = copy.deepcopy(raw_null)
    cov = mne.make_ad_hoc_cov(raw_null.info)
    mne.simulation.add_noise(raw_noise, cov, random_state=0)

    # raw_chpi = copy.deepcopy(raw)
    # mne.simulation.add_chpi(raw_chpi) #, random_state=0)
    # << FIX chpi could not be added

    # Calculate Cascading signal additions
    raw_cascade = copy.deepcopy(raw_null)
    mne.simulation.add_eog(raw_cascade, random_state=0)
    mne.simulation.add_ecg(raw_cascade, random_state=0)
    mne.simulation.add_noise(raw_cascade, cov, random_state=0)

    cascade_data = raw_cascade._data
    serial_data = raw_eog._data + raw_ecg._data + raw_noise._data
    # Should add chpi once fixed above

    test = (cascade_data == serial_data)
    assert test.all()
