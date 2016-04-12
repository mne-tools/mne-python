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

    <h2>Introduction to MNE data structures</h2>
 
  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_creating_data_structures.rst
    auto_tutorials/plot_info.rst
    auto_tutorials/plot_raw_objects.rst

.. container:: span box

  .. raw:: html

    <h2>Preprocessing</h2>

  .. toctree::
    :maxdepth: 1

    tutorials/preprocessing/basic_preprocessing.rst
    tutorials/preprocessing/data_selection.rst
    tutorials/preprocessing/artifacts_suppression.rst
    auto_tutorials/plot_ica_from_raw.rst

.. container:: span box

  .. raw:: html

    <h2>Sensor-level analysis</h2>

  ..   * Time-Frequency analysis with multitapers --TODO
  ..   * Connectivity study with phase-lag index --TODO--
  ..   * Decoding --TODO--

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_epochs_objects.rst
    auto_tutorials/plot_epochs_to_data_frame.rst
    auto_tutorials/plot_epoching_and_averaging.rst

.. container:: span box

  .. raw:: html

    <h2>Source reconstruction</h2>

  .. * data covariance --TODO--

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_compute_covariance.rst
    auto_tutorials/plot_source_localization_basics.rst
    auto_tutorials/plot_brainstorm_auditory.rst

.. container:: span box

  .. raw:: html

    <h2>Sensor-space Analysis</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_cluster_methods_tutorial.rst
    auto_tutorials/plot_spatio_temporal_cluster_stats_sensor.rst
    auto_tutorials/plot_cluster_1samp_test_time_frequency.rst
    auto_tutorials/plot_cluster_stats_time_frequency.rst

.. container:: span box

  .. raw:: html

    <h2>Source-space Analysis</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_cluster_stats_time_frequency_repeated_measures_anova.rst
    auto_tutorials/plot_cluster_stats_spatio_temporal_2samp.rst
    auto_tutorials/plot_cluster_stats_spatio_temporal_repeated_measures_anova.rst
    auto_tutorials/plot_cluster_stats_spatio_temporal.rst

.. container:: span box

  .. raw:: html

    <h2>Visualization and Reporting</h2>

  .. toctree::
    :maxdepth: 1

    auto_tutorials/plot_visualize_evoked.rst
    tutorials/report.rst

.. container:: span box

  .. raw:: html

    <h2>Command line tools</h2>

  .. toctree::
    :maxdepth: 1

    tutorials/command_line.rst
    generated/commands.rst
