:orphan:

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

.. container:: span box

  .. raw:: html

    <h2>Introduction to MNE and Python</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_python_intro.rst
    tutorials/seven_stories_about_mne.rst
    auto_tutorials/plot_introduction.rst

.. container:: span box

  .. raw:: html

    <h2>Background information</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_background_filtering.rst
    auto_tutorials/plot_configuration.rst

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
    auto_tutorials/plot_modifying_data_inplace.rst
    auto_tutorials/plot_object_epochs.rst
    auto_tutorials/plot_object_evoked.rst
    auto_tutorials/plot_creating_data_structures.rst
    auto_tutorials/plot_info.rst
    auto_tutorials/plot_ecog.rst

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
    auto_tutorials/plot_brainstorm_phantom_ctf.rst
    auto_tutorials/plot_brainstorm_phantom_elekta.rst
    auto_tutorials/plot_point_spread.rst

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

    <h2>Decoding and Encoding</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_sensors_decoding.rst
    auto_tutorials/plot_receptive_field.rst

.. container:: span box

  .. raw:: html

    <h2>Command line tools</h2>

  .. toctree::
    :maxdepth: 1

    tutorials/command_line.rst
    generated/commands.rst
