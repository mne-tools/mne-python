mTRF Dataset
============
:func:`mne.datasets.mtrf.data_path`.

This dataset contains 128 channel EEG as well as natural speech stimulus features,
which is also available `here <https://sourceforge.net/projects/aespa/files/>`_.

The experiment consisted of subjects listening to natural speech.
The dataset contains several feature representations of the speech stimulus,
suitable for using to fit continuous regression models of neural activity.
More details and a description of the package can be found in [5]_.

.. topic:: Examples

    * :ref:`Receptive Field Estimation and Prediction <sphx_glr_auto_examples_decoding_plot_receptive_field_mtrf.py>`: Partially replicates the results from Crosse et al. (2016).
