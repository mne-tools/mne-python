:orphan:

.. _documentation:

Documentation
=============

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

**Getting started**

.. toctree::
    :maxdepth: 1

    getting_started.rst
    advanced_setup.rst
    auto_tutorials/plot_python_intro.rst

**MNE basics**

.. toctree::
    :maxdepth: 1

    getting_started.rst
    tutorials/philosophy.rst
    manual/cookbook.rst
    python_reference.rst
    auto_examples/index.rst
    generated/commands.rst
    auto_tutorials/plot_configuration.rst
    faq.rst

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_data">Data structures and containers</a>
          </h4>
        </div>

        <div id="collapse_data" class="panel-collapse collapse">
          <div class="panel-body">

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


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse12">Data I/O and datasets</a>
          </h4>
        </div>
        <div id="collapse12" class="panel-collapse collapse">
          <div class="panel-body">

**Getting your data into MNE**

.. toctree::
    :maxdepth: 1

    manual/io.rst
    auto_tutorials/plot_creating_data_structures.rst
    auto_tutorials/plot_modifying_data_inplace.rst
    auto_tutorials/plot_ecog.rst
    manual/memory.rst

**Working with public datasets**

.. toctree::
    :maxdepth: 1

    manual/datasets_index.rst
    auto_tutorials/plot_brainstorm_auditory.rst
    auto_tutorials/plot_brainstorm_phantom_ctf.rst
    auto_tutorials/plot_brainstorm_phantom_elekta.rst

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_pre">Preprocessing (filtering, SSP, ICA, Maxwell filtering, ...)</a>
          </h4>
        </div>
        <div id="collapse_pre" class="panel-collapse collapse">
          <div class="panel-body">

**Background**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_background_filtering.rst
    manual/preprocessing/ssp.rst
    manual/preprocessing/ica.rst
    manual/preprocessing/maxwell.rst
    manual/channel_interpolation.rst

**Preprocessing your data**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_artifacts_detection.rst
    auto_tutorials/plot_artifacts_correction_filtering.rst
    auto_tutorials/plot_artifacts_correction_rejection.rst
    auto_tutorials/plot_artifacts_correction_ssp.rst
    auto_tutorials/plot_artifacts_correction_ica.rst
    auto_tutorials/plot_artifacts_correction_maxwell_filtering.rst

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_viz">Visualization</a>
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


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse6">Source-level analysis</a>
          </h4>
        </div>
        <div id="collapse6" class="panel-collapse collapse">
          <div class="panel-body">

**Background**

.. toctree::
    :maxdepth: 1

    manual/source_localization/forward.rst
    manual/source_localization/inverse.rst
    manual/source_localization/morph.rst

**Getting data to source space**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_forward.rst
    auto_tutorials/plot_compute_covariance.rst
    auto_tutorials/plot_mne_dspm_source_localization.rst
    auto_tutorials/plot_dipole_fit.rst
    auto_tutorials/plot_point_spread.rst

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse7">Statistics in sensor- and source-space</a>
          </h4>
        </div>
        <div id="collapse7" class="panel-collapse collapse">
          <div class="panel-body">


**Background**

.. toctree::
    :maxdepth: 1

    manual/statistics.rst

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

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse9">Machine learning (decoding, encoding, MVPA)</a>
          </h4>
        </div>
        <div id="collapse9" class="panel-collapse collapse">
          <div class="panel-body">

**Decoding**

.. toctree::
    :maxdepth: 1

    manual/decoding.rst
    auto_tutorials/plot_sensors_decoding.rst

**Encoding**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_receptive_field.rst

.. raw:: html

          </div>
        </div>
      </div>

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse11">MNE-C and MNE-MATLAB</a>
          </h4>
        </div>
        <div id="collapse11" class="panel-collapse collapse">
          <div class="panel-body">

**MNE-C**

.. toctree::
    :maxdepth: 1

    tutorials/command_line.rst
    manual/c_reference.rst
    manual/gui/analyze.rst
    manual/gui/browse.rst
    manual/appendix/bem_model.rst
    manual/appendix/c_misc.rst

**MNE-MATLAB**

.. toctree::
    :maxdepth: 1

    manual/matlab.rst


.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_misc">Contributing</a>
          </h4>
        </div>

        <div id="collapse_misc" class="panel-collapse collapse">
          <div class="panel-body">

.. toctree::
    :maxdepth: 1

    contributing.rst
    configure_git.rst
    customizing_git.rst

.. raw:: html

          </div>
        </div>
      </div>
    </div>
