#!/usr/bin/env bash

# Generate events
mne_process_raw --raw test_raw.fif \
        --eventsout test-eve.fif

# Averaging
mne_process_raw --raw test_raw.fif --lowpass 40 --projoff \
        --saveavetag -ave --ave test.ave

# Compute the noise covariance matrix
mne_process_raw --raw test_raw.fif --lowpass 40 --projoff \
        --savecovtag -cov --cov test.cov
