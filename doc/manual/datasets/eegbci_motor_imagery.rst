EEGBCI motor imagery
====================
:func:`mne.datasets.eegbci.load_data`

The EEGBCI dataset is documented in [2]_. The data set is available at PhysioNet [3]_.
The dataset contains 64-channel EEG recordings from 109 subjects and 14 runs on each subject in EDF+ format.
The recordings were made using the BCI2000 system. To load a subject, do::

    from mne.io import concatenate_raws, read_raw_edf
    from mne.datasets import eegbci
    raw_fnames = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_decoding_plot_decoding_csp_eeg.py`
