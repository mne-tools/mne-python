:orphan:

.. _guide:

A Guide to MNE-python
=====================

This is a collection of tutorials to guide you through the process
of working with your data using MNE. It also serves as a high-level guide
for many of the tools and techniques available to you with MNE. You can find
each step of the processing pipeline, and re-run the Python code by copy-paste.

**Getting Started**

Here are some steps to get you started.

- First you should get :ref:`Python and MNE-Python up and running <install_python_and_mne_python>`. 
- For a high-level overview of what you can do with MNE-Python:
  :ref:`what_can_you_do`
- For more examples of analyzing M/EEG data, including more sophisticated
  analysis: :ref:`general_examples`
- For details about specific functions and classes: :ref:`api_reference`
- For a high-level view of analysis pipelines and workflows in MNE-python,
  see the :ref:`cookbook`.

**Click one of the headers below to see what you can do in MNE-python.**

.. raw:: html

    <style class='text/css'>
    .panel-title a {
        display: block;
        padding: 5px;
        text-decoration: none;
    }

    .plus {
        float: right;
        color: #212121;
    }

    .panel {
        margin-bottom: 3px;
    }
    </style>

.. raw:: html

    <div class="panel-group">
      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_intro">Introduction to MNE and Python</a>
          </h4>
        </div>

        <div id="collapse_intro" class="panel-collapse collapse">
          <div class="panel-body">

* `Introduction to Python <auto_tutorials/plot_python_intro.rst>`_
* `Seven highlights of MNE <tutorials/seven_stories_about_mne.rst>`_
* `Analysis workflows in MNE <manual/cookbook.html>`_
* `Configuring MNE-python <auto_tutorials/plot_configuration.rst>`_      

.. raw:: html

          </div>
        </div>
      </div>


.. raw:: html

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_data">
              Data Structures and Containers
            </a>
          </h4>
        </div>

        <div id="collapse_data" class="panel-collapse collapse">
          <div class="panel-body">

**Tutorials**

* `Creating data structures <auto_tutorials/plot_creating_data_structures.rst>`_
* `Importing data <manual/io.html>`_
* `Modifying data in place <auto_tutorials≤/plot_modifying_data_inplace.rst>`_
* `Working with ECoG data in MNE <auto_tutorials/plot_ecog.rst>`_

**Data Structures in MNE**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_object_raw.rst
    auto_tutorials/plot_object_epochs.rst
    auto_tutorials/plot_object_evoked.rst
    auto_tutorials/plot_info.rst


.. raw:: html

          </div>
        </div>
      </div>


.. raw:: html

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_pre">Preprocessing and Filtering</a>
          </h4>
        </div>

        <div id="collapse_pre" class="panel-collapse collapse">
          <div class="panel-body">

**Tutorials**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_artifacts_detection.rst
    auto_tutorials/plot_artifacts_correction_filtering.rst
    auto_tutorials/plot_artifacts_correction_rejection.rst
    auto_tutorials/plot_artifacts_correction_ssp.rst
    auto_tutorials/plot_artifacts_correction_ica.rst
    auto_tutorials/plot_artifacts_correction_maxwell_filtering.rst
    auto_tutorials/plot_background_filtering.rst

**Theoretical Background**

.. toctree::
    :maxdepth: 1

    manual/preprocessing/ica.rst
    manual/preprocessing/maxwell.rst
    manual/preprocessing/ssp.rst
    manual/channel_interpolation.rst

.. raw:: html

          </div>
        </div>
      </div>


.. raw:: html

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_viz">Visualization and Reporting</a>
          </h4>
        </div>

        <div id="collapse_viz" class="panel-collapse collapse">
          <div class="panel-body">

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_visualize_raw.rst
    auto_tutorials/plot_visualize_epochs.rst
    auto_tutorials/plot_visualize_evoked.rst
    tutorials/report.rst

.. raw:: html

          </div>
        </div>
      </div>


.. raw:: html

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_analysis">Time- and frequency-domain analyses</a>
          </h4>
        </div>

        <div id="collapse_analysis" class="panel-collapse collapse">
          <div class="panel-body">

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_introduction.rst
    auto_tutorials/plot_epoching_and_averaging.rst
    auto_tutorials/plot_eeg_erp.rst
    auto_tutorials/plot_sensors_time_frequency.rst
    manual/time_frequency.rst

.. raw:: html

          </div>
        </div>
      </div>


.. raw:: html

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse6">Source-level analysis</a>
          </h4>
        </div>

        <div id="collapse6" class="panel-collapse collapse">
          <div class="panel-body">

**Tutorials**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_forward.rst
    auto_tutorials/plot_compute_covariance.rst
    auto_tutorials/plot_mne_dspm_source_localization.rst
    auto_tutorials/plot_dipole_fit.rst
    auto_tutorials/plot_point_spread.rst

**Theoretical Background**

.. toctree::
    :maxdepth: 1

    manual/source_localization/forward.rst
    manual/source_localization/inverse.rst
    manual/source_localization/morph.rst

.. raw:: html

          </div>
        </div>
      </div>


.. raw:: html

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse7">Statistics in sensor- and source-space</a>
          </h4>
        </div>

        <div id="collapse7" class="panel-collapse collapse">
          <div class="panel-body">


**Sensor Space**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_stats_cluster_methods.rst
    auto_tutorials/plot_stats_spatio_temporal_cluster_sensors.rst
    auto_tutorials/plot_stats_cluster_1samp_test_time_frequency.rst
    auto_tutorials/plot_stats_cluster_time_frequency.rst

**Source Space**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_stats_cluster_time_frequency_repeated_measures_anova.rst
    auto_tutorials/plot_stats_cluster_spatio_temporal_2samp.rst
    auto_tutorials/plot_stats_cluster_spatio_temporal_repeated_measures_anova.rst
    auto_tutorials/plot_stats_cluster_spatio_temporal.rst

**Theoretical Background**

.. toctree::
    :maxdepth: 1

    manual/statistics.rst

.. raw:: html

          </div>
        </div>
      </div>


.. raw:: html

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse9">Machine Learning (Decoding, Encoding, MVPA)</a>
          </h4>
        </div>

        <div id="collapse9" class="panel-collapse collapse">
          <div class="panel-body">

**Decoding**

* `An introduction to decoding with MNE-python <manual/decoding.html>`_
* `Decoding sensor space data <auto_tutorials/plot_sensors_decoding.html>`_

**Encoding**

* `Spectro-temporal receptive field modeling on continuous data <auto_tutorials/plot_receptive_field.html>`_

.. raw:: html

          </div>
        </div>
      </div>


.. raw:: html

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse11">Datasets and Other Software</a>
          </h4>
        </div>

        <div id="collapse11" class="panel-collapse collapse">
          <div class="panel-body">

**Datasets - Background**

* `Importing and exporting data with MNE <manual/io.html>`_
* `Datasets in MNE <manual/datasets_index.html>`_

**Datasets - Tutorials**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_brainstorm_auditory.rst
    auto_tutorials/plot_brainstorm_phantom_ctf.rst
    auto_tutorials/plot_brainstorm_phantom_elekta.rst

**MNE-C**

.. toctree::
    :maxdepth: 1

    tutorials/mne_c.rst
    tutorials/command_line.rst
    manual/c_reference.rst
    manual/gui/analyze.rst
    manual/gui/browse.rst
    manual/appendix/bem_model.rst
    manual/appendix/c_misc.rst

**Non-python MNE**

.. toctree::
    :maxdepth: 1

    manual/matlab.rst
    tutorials/mne_cpp.rst


.. raw:: html

          </div>
        </div>
      </div>


.. raw:: html

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_misc">Miscellaneous</a>
          </h4>
        </div>

        <div id="collapse_misc" class="panel-collapse collapse">
          <div class="panel-body">

.. toctree::
    :maxdepth: 1

    tutorials/contributing.rst
    tutorials/configure_git.rst
    tutorials/customizing_git.rst
    manual/memory.rst
    manual/pitfalls.rst
    tutorials/advanced_setup.rst
    generated/commands.rst

.. raw:: html

          </div>
        </div>
      </div>
    </div>
