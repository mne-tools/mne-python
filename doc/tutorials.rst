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

.. contents:: Categories
   :local:
   :depth: 1


.. container:: row

  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Introduction to MNE and Python</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        auto_tutorials/plot_python_intro.rst
        tutorials/seven_stories_about_mne.rst
        auto_tutorials/plot_introduction.rst


.. container:: row

  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Background Information</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        auto_tutorials/plot_background_filtering.rst


  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Preprocessing</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        auto_tutorials/plot_artifacts_detection.rst
        auto_tutorials/plot_artifacts_correction_filtering.rst
        auto_tutorials/plot_artifacts_correction_rejection.rst
        auto_tutorials/plot_artifacts_correction_ssp.rst
        auto_tutorials/plot_artifacts_correction_ica.rst
        auto_tutorials/plot_artifacts_correction_maxwell_filtering.rst


.. container:: row

  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Data Structures and Containers</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        auto_tutorials/plot_object_raw.rst
        auto_tutorials/plot_modifying_data_inplace.rst
        auto_tutorials/plot_object_epochs.rst
        auto_tutorials/plot_object_evoked.rst
        auto_tutorials/plot_creating_data_structures.rst
        auto_tutorials/plot_info.rst


  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Visualization and Reporting</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        auto_tutorials/plot_visualize_raw.rst
        auto_tutorials/plot_visualize_epochs.rst
        auto_tutorials/plot_visualize_evoked.rst
        tutorials/report.rst


.. container:: row

  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Sensor analysis</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        auto_tutorials/plot_epoching_and_averaging.rst
        auto_tutorials/plot_eeg_erp.rst
        auto_tutorials/plot_sensors_time_frequency.rst
        auto_tutorials/plot_sensors_decoding.rst

  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Source Analysis</h3></div>

    .. container:: panel-body nosize

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

.. container:: row

  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Sensor-space Statistics</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        auto_tutorials/plot_stats_cluster_methods.rst
        auto_tutorials/plot_stats_spatio_temporal_cluster_sensors.rst
        auto_tutorials/plot_stats_cluster_1samp_test_time_frequency.rst
        auto_tutorials/plot_stats_cluster_time_frequency.rst


  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Source-space Statistics</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        auto_tutorials/plot_stats_cluster_time_frequency_repeated_measures_anova.rst
        auto_tutorials/plot_stats_cluster_spatio_temporal_2samp.rst
        auto_tutorials/plot_stats_cluster_spatio_temporal_repeated_measures_anova.rst
        auto_tutorials/plot_stats_cluster_spatio_temporal.rst


.. container:: row

  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Decoding</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        auto_tutorials/plot_sensors_decoding.rst

.. container:: row

  .. container:: panel panel-default halfpad

    .. raw:: html

      <div class="panel-heading"><h3 class="panel-title">Command-line Tools</h3></div>

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 1

        generated/commands.rst
        tutorials/command_line.rst
