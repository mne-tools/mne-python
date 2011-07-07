#!/usr/bin/env bash

# Generate events
mne_process_raw --raw test_raw.fif --eventsout test-eve.fif

# Averaging no filter
mne_process_raw --raw test_raw.fif --projon --filteroff \
        --saveavetag -nf-ave --ave test-no-reject.ave

# Averaging 40Hz
mne_process_raw --raw test_raw.fif --lowpass 40 --projoff \
        --saveavetag -ave --ave test.ave

# Compute the noise covariance matrix
mne_process_raw --raw test_raw.fif --filteroff --projon \
        --savecovtag -cov --cov test.cov

# Compute the noise covariance matrix with keepsamplemean
mne_process_raw --raw test_raw.fif --filteroff --projon \
        --savecovtag -km-cov --cov test_keepmean.cov

# Compute projection
mne_process_raw --raw test_raw.fif --events test-eve.fif --makeproj \
           --projtmin -0.2 --projtmax 0.3 --saveprojtag _proj \
           --projnmag 1 --projngrad 1 --projevent 1 \
           --projmagrej 600000 --projgradrej 500000  --filteroff
