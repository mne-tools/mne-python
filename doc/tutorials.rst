.. _tutorials:


Tutorials
=========
These tutorials get you started to processing MEG/EEG data with MNE-Python. You can find each step of the processing pipeline, and re-run the python code by copy-paste.

.. toctree::
  :maxdepth: 1

  tutorials/introduction_to_MNE.rst

.. container:: span box

  .. raw:: html

    <h2>Introduction to MNE data structures</h2>
 
  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_creating_data_structures.rst
    auto_tutorials/plot_info.rst
    auto_tutorials/plot_raw_objects.rst
    auto_tutorials/plot_epochs_objects.rst
    auto_tutorials/plot_epochs_to_data_frame.rst

  .. raw:: html

    <h2>Preprocessing</h2>

  .. toctree::
    :maxdepth: 1

    tutorials/preprocessing/basic_preprocessing.rst
    tutorials/preprocessing/data_selection.rst
    tutorials/preprocessing/artifacts_suppression.rst
    auto_tutorials/plot_ica_from_raw.rst

  .. raw:: html

    <h2>Sensor-level analysis</h2>

  * Epoching and Averaging --TODO--
  * Time-Frequency analysis with multitapers --TODO
  * Connectivity study with phase-lag index --TODO--
  * Decoding --TODO--

  .. raw:: html

    <h2>Source reconstruction</h2>

  * noise/data covariance --TODO--

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_source_localization_basics.rst


.. container:: span box

  .. raw:: html

    <h2>Sensor-space Statistics</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_cluster_methods_tutorial.rst
    auto_tutorials/plot_spatio_temporal_cluster_stats_sensor.rst
    auto_tutorials/plot_cluster_1samp_test_time_frequency.rst
    auto_tutorials/plot_cluster_stats_time_frequency.rst

  .. raw:: html

    <h2>Source-space Statistics</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_cluster_stats_time_frequency_repeated_measures_anova.rst
    auto_tutorials/plot_cluster_stats_spatio_temporal_2samp.rst
    auto_tutorials/plot_cluster_stats_spatio_temporal_repeated_measures_anova.rst
    auto_tutorials/plot_cluster_stats_spatio_temporal.rst

  .. raw:: html

    <h2>Visualization and Reporting</h2>

  .. toctree::
    :maxdepth: 1

    tutorials/report.rst

  .. raw:: html

    <h2>Command line tools</h2>

  .. toctree::
    :maxdepth: 1

    tutorials/command_line.rst
