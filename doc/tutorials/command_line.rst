.. _command_line_tutorial:

Getting started with MNE Unix command line tools
================================================

This tutorial is a really short step by step presentation
of an analysis pipeline using the MNE-C command line tools.
These tools are UNIX commands and therefore only run on
Mac OS or Linux.

See :ref:`install_mne_c` to setup your system for using the
MNE-C tools.

The quick start guide shows how to run a standard processing of the
sample data set provided with MNE. The sample dataset is further
described in :ref:`datasets`.

All the following lines are to be run in a terminal and not in a Python
interpreter.

First define your subject::

    export SUBJECT=sample

Build your source space::

    # MRI (this is not really needed for anything)
    mne_setup_mri --overwrite

    # Source space
    mne_setup_source_space --ico -6 --overwrite

Prepare for forward computation::

    # For homogeneous volume conductor (just inner skull)
    mne_setup_forward_model --homog --surf --ico 4

    # or for a three compartment model (inner and outer skull and skin)
    mne_setup_forward_model --surf --ico 4

List your bad channels in a file. Example sample_bads.bad contains::

    MEG 2443
    EEG 053

Mark bad channels::

    mne_mark_bad_channels --bad sample_bads.bad sample_audvis_raw.fif

Compute averaging::

    mne_process_raw --raw sample_audvis_raw.fif --lowpass 40 --projoff \
            --saveavetag -ave --ave audvis.ave

Compute the noise covariance matrix::

    mne_process_raw --raw sample_audvis_raw.fif --lowpass 40 --projoff \
            --savecovtag -cov --cov audvis.cov

Compute forward solution a.k.a. lead field::

    # for MEG only
    mne_do_forward_solution --mindist 5 --spacing oct-6 \
        --meas sample_audvis_raw.fif --bem sample-5120 --megonly --overwrite \
        --fwd sample_audvis-meg-oct-6-fwd.fif

    # for EEG only
    mne_do_forward_solution --mindist 5 --spacing oct-6 \
        --meas sample_audvis_raw.fif --bem sample-5120-5120-5120 --eegonly \
        --fwd sample_audvis-eeg-oct-6-fwd.fif

    # for both EEG and MEG
    mne_do_forward_solution --mindist 5 --spacing oct-6 \
        --meas sample_audvis_raw.fif --bem sample-5120-5120-5120 \
        --fwd sample_audvis-meg-eeg-oct-6-fwd.fif

Compute MNE inverse operators::

    # Note: The MEG/EEG forward solution could be used for all
    mne_do_inverse_operator --fwd sample_audvis-meg-oct-6-fwd.fif \
            --depth --loose 0.2 --meg

    mne_do_inverse_operator --fwd sample_audvis-eeg-oct-6-fwd.fif \
            --depth --loose 0.2 --eeg

    mne_do_inverse_operator --fwd sample_audvis-meg-eeg-oct-6-fwd.fif \
            --depth --loose 0.2 --eeg --meg

Produce stc files (activation files)::

    # for MEG
    mne_make_movie --inv sample_audvis-meg-oct-6-${mod}-inv.fif \
        --meas sample_audvis-ave.fif \
        --tmin 0 --tmax 250 --tstep 10 --spm \
        --smooth 5 --bmin -100 --bmax 0 --stc sample_audvis-meg

    # for EEG
    mne_make_movie --inv sample_audvis-eeg-oct-6-${mod}-inv.fif \
        --meas sample_audvis-ave.fif \
        --tmin 0 --tmax 250 --tstep 10 --spm \
        --smooth 5 --bmin -100 --bmax 0 --stc sample_audvis-eeg

    # for MEG and EEG combined
    mne_make_movie --inv sample_audvis-meg-eeg-oct-6-${mod}-inv.fif \
        --meas sample_audvis-ave.fif \
        --tmin 0 --tmax 250 --tstep 10 --spm \
        --smooth 5 --bmin -100 --bmax 0 --stc sample_audvis-meg-eeg

And, we're done!

See also :ref:`python_commands` for more command line tools
using MNE-Python.