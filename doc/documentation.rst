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

    .example_details {
        padding-left: 20px;
        margin-bottom: 10px;
    }

    </style>

This is where you can learn about all the things you can do with MNE. It contains **background information** and **tutorials** for taking a deep-dive into the techniques that MNE-python covers. You'll find practical information on how to use these methods with your data, and in many cases some high-level concepts underlying these methods.

There are also **examples**, which contain a short use-case to highlight MNE-functionality and provide inspiration for the many things you can do with this package. You can also find a gallery of these examples in the `examples gallery <auto_examples/index.html>`_.

**See the links below for an introduction to MNE-python, or click one of the sections on this page to see more.**

.. raw:: html

    <div class="panel-group">
      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_intro">Introduction to MNE and Python</a>
          </h4>
        </div>

        <div id="collapse_intro" class="panel-collapse collapse in">
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

    tutorials/philosophy.rst
    manual/cookbook.rst
    whats_new.rst
    python_reference.rst
    auto_examples/index.rst
    generated/commands.rst
    auto_tutorials/plot_configuration.rst
    cited.rst
    faq.rst

**More help**

- :ref:`Cite MNE <cite>`
- `Mailing list <MNE mailing list>`_ for analysis talk
- `GitHub issues <https://github.com/mne-tools/mne-python/issues/>`_ for
  requests and bug reports
- `Gitter <https://gitter.im/mne-tools/mne-python>`_ to chat with devs

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
            <a data-toggle="collapse" href="#collapse_datasets">Data I/O and datasets</a>
          </h4>
        </div>
        <div id="collapse_datasets" class="panel-collapse collapse">
          <div class="panel-body">

**Getting your data into MNE**

.. toctree::
    :maxdepth: 1

    manual/io.rst
    auto_tutorials/plot_creating_data_structures.rst
    auto_tutorials/plot_modifying_data_inplace.rst
    auto_tutorials/plot_ecog.rst
    manual/memory.rst

.. raw:: html

  <details class="example_details">
  <summary><strong>Examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/io/plot_elekta_epochs.rst
    auto_examples/io/plot_objects_from_arrays.rst
    auto_examples/io/plot_read_and_write_raw_data.rst
    auto_examples/io/plot_read_epochs.rst
    auto_examples/io/plot_read_events.rst
    auto_examples/io/plot_read_evoked.rst
    auto_examples/io/plot_read_noise_covariance_matrix.rst

.. raw:: html

    </details>

**Working with public datasets**

.. toctree::
    :maxdepth: 1

    manual/datasets_index.rst
    auto_tutorials/plot_brainstorm_auditory.rst
    auto_tutorials/plot_brainstorm_phantom_ctf.rst
    auto_tutorials/plot_brainstorm_phantom_elekta.rst
    auto_examples/datasets/plot_brainstorm_data.rst
    auto_examples/datasets/plot_megsim_data.rst
    auto_examples/datasets/plot_megsim_data_single_trial.rst
    auto_examples/datasets/plot_spm_faces_dataset.rst

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

  <details class="example_details">
  <summary><strong>Examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/preprocessing/plot_define_target_events.rst
    auto_examples/preprocessing/plot_eog_artifact_histogram.rst
    auto_examples/preprocessing/plot_find_ecg_artifacts.rst
    auto_examples/preprocessing/plot_find_eog_artifacts.rst
    auto_examples/preprocessing/plot_head_positions.rst
    auto_examples/preprocessing/plot_interpolate_bad_channels.rst
    auto_examples/preprocessing/plot_movement_compensation.rst
    auto_examples/preprocessing/plot_rereference_eeg.rst
    auto_examples/preprocessing/plot_resample.rst
    auto_examples/preprocessing/plot_run_ica.rst
    auto_examples/preprocessing/plot_shift_evoked.rst
    auto_examples/preprocessing/plot_virtual_evoked.rst
    auto_examples/preprocessing/plot_xdawn_denoising.rst

.. raw:: html

    </details>

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

  <details class="example_details">
  <summary><strong>Examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/visualization/make_report.rst
    auto_examples/visualization/plot_3d_to_2d.rst
    auto_examples/visualization/plot_channel_epochs_image.rst
    auto_examples/visualization/plot_eeg_on_scalp.rst
    auto_examples/visualization/plot_evoked_topomap.rst
    auto_examples/visualization/plot_evoked_whitening.rst
    auto_examples/visualization/plot_meg_sensors.rst
    auto_examples/visualization/plot_parcellation.rst
    auto_examples/visualization/plot_sensor_noise_level.rst
    auto_examples/visualization/plot_ssp_projs_sensitivity_map.rst
    auto_examples/visualization/plot_topo_compare_conditions.rst
    auto_examples/visualization/plot_topo_customized.rst

.. raw:: html

    </details>
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

**Tutorials**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_introduction.rst
    auto_tutorials/plot_epoching_and_averaging.rst
    auto_tutorials/plot_eeg_erp.rst
    auto_tutorials/plot_sensors_time_frequency.rst
    manual/time_frequency.rst

.. raw:: html

  <details class="example_details">
  <summary><strong>Examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/time_frequency/plot_compute_raw_data_spectrum.rst
    auto_examples/time_frequency/plot_compute_source_psd_epochs.rst
    auto_examples/time_frequency/plot_source_label_time_frequency.rst
    auto_examples/time_frequency/plot_source_power_spectrum.rst
    auto_examples/time_frequency/plot_source_space_time_frequency.rst
    auto_examples/time_frequency/plot_temporal_whitening.rst
    auto_examples/time_frequency/plot_time_frequency_global_field_power.rst
    auto_examples/time_frequency/plot_time_frequency_simulated.rst

.. raw:: html

  </details>

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_source">Source-level analysis</a>
          </h4>
        </div>
        <div id="collapse_source" class="panel-collapse collapse">
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

  <details class="example_details">
  <summary><strong>Forward examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/forward/plot_decimate_head_surface.rst
    auto_examples/forward/plot_forward_sensitivity_maps.rst
    auto_examples/forward/plot_left_cerebellum_volume_source.rst
    auto_examples/forward/plot_read_bem_surfaces.rst
    auto_examples/forward/plot_source_space_morphing.rst

.. raw:: html

    </details>

.. raw:: html

  <details class="example_details">
  <summary><strong>Inverse examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/inverse/plot_compute_mne_inverse_epochs_in_label.rst
    auto_examples/inverse/plot_compute_mne_inverse_raw_in_label.rst
    auto_examples/inverse/plot_compute_mne_inverse_volume.rst
    auto_examples/inverse/plot_covariance_whitening_dspm.rst
    auto_examples/inverse/plot_custom_inverse_solver.rst
    auto_examples/inverse/plot_dics_beamformer.rst
    auto_examples/inverse/plot_dics_source_power.rst
    auto_examples/inverse/plot_gamma_map_inverse.rst
    auto_examples/inverse/plot_label_activation_from_stc.rst
    auto_examples/inverse/plot_label_from_stc.rst
    auto_examples/inverse/plot_label_source_activations.rst
    auto_examples/inverse/plot_lcmv_beamformer.rst
    auto_examples/inverse/plot_lcmv_beamformer_volume.rst
    auto_examples/inverse/plot_mixed_source_space_inverse.rst
    auto_examples/inverse/plot_mixed_norm_inverse.rst
    auto_examples/inverse/plot_mne_crosstalk_function.rst
    auto_examples/inverse/plot_mne_point_spread_function.rst
    auto_examples/inverse/plot_morph_data.rst
    auto_examples/inverse/plot_rap_music.rst
    auto_examples/inverse/plot_read_stc.rst
    auto_examples/inverse/plot_read_inverse.rst
    auto_examples/inverse/plot_read_source_space.rst
    auto_examples/inverse/plot_snr_estimate.rst
    auto_examples/inverse/plot_tf_dics.rst
    auto_examples/inverse/plot_tf_lcmv.rst
    auto_examples/inverse/plot_time_frequency_mixed_norm_inverse.rst

.. raw:: html

    </details>

.. raw:: html

  <details class="example_details">
  <summary><strong>Simulation examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/simulation/plot_simulate_evoked_data.rst
    auto_examples/simulation/plot_simulate_raw_data.rst

.. raw:: html

    </details>

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_stats">Statistics in sensor- and source-space</a>
          </h4>
        </div>
        <div id="collapse_stats" class="panel-collapse collapse">
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

.. raw:: html

  <details class="example_details">
  <summary><strong>Examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/stats/plot_fdr_stats_evoked.rst
    auto_examples/stats/plot_cluster_stats_evoked.rst
    auto_examples/stats/plot_sensor_permutation_test.rst
    auto_examples/stats/plot_sensor_regression.rst
    auto_examples/stats/plot_linear_regression_raw.rst

.. raw:: html

    </details>

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
            <a data-toggle="collapse" href="#collapse_ml">Machine learning (decoding, encoding, MVPA)</a>
          </h4>
        </div>
        <div id="collapse_ml" class="panel-collapse collapse">
          <div class="panel-body">

**Decoding**

.. toctree::
    :maxdepth: 1

    manual/decoding.rst
    auto_tutorials/plot_sensors_decoding.rst

.. raw:: html

  <details class="example_details">
  <summary><strong>Examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/decoding/decoding_rsa.rst
    auto_examples/decoding/plot_decoding_csp_eeg.rst
    auto_examples/decoding/plot_decoding_csp_space.rst
    auto_examples/decoding/plot_decoding_csp_timefreq.rst
    auto_examples/decoding/plot_decoding_spatio_temporal_source.rst
    auto_examples/decoding/plot_decoding_time_generalization_conditions.rst
    auto_examples/decoding/plot_decoding_unsupervised_spatial_filter.rst
    auto_examples/decoding/plot_decoding_xdawn_eeg.rst
    auto_examples/decoding/plot_ems_filtering.rst
    auto_examples/decoding/plot_linear_model_patterns.rst

.. raw:: html

    </details>

**Encoding**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_receptive_field.rst

.. raw:: html

  <details class="example_details">
  <summary><strong>Examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/decoding/plot_receptive_field.rst

.. raw:: html

    </details>

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_conn">Connectivity</a>
          </h4>
        </div>
        <div id="collapse_conn" class="panel-collapse collapse">
          <div class="panel-body">

**Examples**

.. toctree::
    :maxdepth: 1

    auto_examples/connectivity/plot_cwt_sensor_connectivity.rst
    auto_examples/connectivity/plot_mixed_source_space_connectity.rst
    auto_examples/connectivity/plot_mne_inverse_coherence_epochs.rst
    auto_examples/connectivity/plot_mne_inverse_connectivity_spectrum.rst
    auto_examples/connectivity/plot_mne_inverse_label_connectivity.rst
    auto_examples/connectivity/plot_mne_inverse_psi_visual.rst
    auto_examples/connectivity/plot_sensor_connectivity.rst

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_real">Realtime</a>
          </h4>
        </div>
        <div id="collapse_real" class="panel-collapse collapse">
          <div class="panel-body">

**Examples**

.. toctree::
    :maxdepth: 1

    auto_examples/realtime/ftclient_rt_average.rst
    auto_examples/realtime/ftclient_rt_compute_psd.rst
    auto_examples/realtime/plot_compute_rt_average.rst
    auto_examples/realtime/plot_compute_rt_decoder.rst
    auto_examples/realtime/rt_feedback_client.rst
    auto_examples/realtime/rt_feedback_server.rst

.. raw:: html

          </div>
        </div>
      </div>

      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_c">MNE-C and MNE-MATLAB</a>
          </h4>
        </div>
        <div id="collapse_c" class="panel-collapse collapse">
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
            <a data-toggle="collapse" href="#collapse_contributing">Contributing</a>
          </h4>
        </div>

        <div id="collapse_contributing" class="panel-collapse collapse">
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
