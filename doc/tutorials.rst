.. _tutorials:

Tutorials
=========

Once you have
:ref:`Python and MNE-Python up and running <install_python_and_mne_python>`,
you can use these tutorials to get started processing MEG/EEG.
You can find each step of the processing pipeline, and re-run the
Python code by copy-paste.

These tutorials aim to capture only the most important information.
For further reading:

- For a high-level overview of what you can do with MNE-Python:
  :ref:`what_can_you_do`
- For more examples of analyzing M/EEG data, including more sophisticated
  analysis: :ref:`general_examples`
- For details about analysis steps: :ref:`manual`
- For details about specific functions and classes: :ref:`api_reference`

.. note:: The default location for the MNE-sample data is
          my-path-to/mne-python/examples. If you downloaded data and an
          example asks you whether to download it again, make sure
          the data reside in the examples directory
          and that you run the script from its current directory.

          .. code-block:: bash

              $ cd examples/preprocessing

          Then in Python you can do::

              In [1]: %run plot_find_ecg_artifacts.py


          See :ref:`datasets` for a list of all available datasets and some
          advanced configuration options, e.g. to specify a custom
          location for storing the datasets.

.. container:: span box

  .. raw:: html

    <h2>Introduction to MNE and Python</h2>
 
  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_python_intro.rst
    tutorials/seven_stories_about_mne.rst

.. container:: span box

  .. raw:: html

    <h2>Preprocessing</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_artifacts_detection.rst
    auto_tutorials/plot_artifacts_correction_filtering.rst
    auto_tutorials/plot_artifacts_correction_rejection.rst
    auto_tutorials/plot_artifacts_correction_ssp.rst
    auto_tutorials/plot_artifacts_correction_ica.rst
    auto_tutorials/plot_artifacts_correction_maxwell_filtering.rst

.. container:: span box

  .. raw:: html

    <h2>Sensor-level analysis</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_epoching_and_averaging.rst
    auto_tutorials/plot_eeg_erp.rst
    auto_tutorials/plot_sensors_time_frequency.rst
    auto_tutorials/plot_sensors_decoding.rst

.. container:: span box

  .. raw:: html

    <h2>Visualization and Reporting</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_visualize_raw.rst
    auto_tutorials/plot_visualize_epochs.rst
    auto_tutorials/plot_visualize_evoked.rst
    tutorials/report.rst

.. container:: span box

  .. raw:: html

    <h2>Manipulating Data Structures and Containers</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_object_raw.rst
    auto_tutorials/plot_object_epochs.rst
    auto_tutorials/plot_object_evoked.rst
    auto_tutorials/plot_creating_data_structures.rst
    auto_tutorials/plot_info.rst

.. container:: span box

  .. raw:: html

    <h2>Source-level analysis</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_forward.rst
    auto_tutorials/plot_compute_covariance.rst
    auto_tutorials/plot_mne_dspm_source_localization.rst
    auto_tutorials/plot_dipole_fit.rst
    auto_tutorials/plot_brainstorm_auditory.rst

.. container:: span box

  .. raw:: html

    <h2>Sensor-space Univariate Statistics</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_stats_cluster_methods.rst
    auto_tutorials/plot_stats_spatio_temporal_cluster_sensors.rst
    auto_tutorials/plot_stats_cluster_1samp_test_time_frequency.rst
    auto_tutorials/plot_stats_cluster_time_frequency.rst

.. container:: span box

  .. raw:: html

    <h2>Source-space Univariate Statistics</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_stats_cluster_time_frequency_repeated_measures_anova.rst
    auto_tutorials/plot_stats_cluster_spatio_temporal_2samp.rst
    auto_tutorials/plot_stats_cluster_spatio_temporal_repeated_measures_anova.rst
    auto_tutorials/plot_stats_cluster_spatio_temporal.rst

.. container:: span box

  .. raw:: html

    <h2>Multivariate Statistics - Decoding</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_sensors_decoding.rst

.. container:: span box

  .. raw:: html

    <h2>Command line tools</h2>

  .. toctree::
    :maxdepth: 1

    tutorials/command_line.rst
    generated/commands.rst
