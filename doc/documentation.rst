:orphan:

.. include:: known_projects.inc

Documentation
=============

The documentation for MNE-Python is divided into four main sections:

1. The :doc:`auto_tutorials/index` provide narrative explanations, sample code,
   and expected output for the most common MNE-Python analysis tasks. The
   emphasis here is on thorough explanations that get new users up to speed
   quickly, at the expense of covering only a limited number of topics. The
   tutorials are arranged in a fixed order; in theory a user should be able to
   progress through the tutorials without encountering any cases where
   background knowledge is assumed and unexplained.

2. The :doc:`MNE-Python API reference <python_reference>` provides
   documentation for every function and method in the MNE-Python codebase. This
   is the same information that is rendered when running
   ``help(mne.<function_name>)`` in an interactive Python session, or when
   typing ``mne.<function_name>?`` in an IPython session or Jupyter notebook.

3. The :doc:`glossary` provides short definitions of MNE-Python-specific
   vocabulary. The glossary is often a good place to look if you don't
   understand a term used in the API reference for a function.

4. The :doc:`examples gallery <auto_examples/index>` provides working code
   samples demonstrating various analysis and visualization techniques. These
   examples often lack the narrative explanations seen in the tutorials, and do
   not follow any specific order. These examples are a useful way to discover
   new analysis or plotting ideas, or to see how a particular technique you've
   read about can be applied using MNE-Python.

.. note::

   If you haven't already installed Python and MNE-Python, here are the
   :doc:`installation instructions <../getting_started>`.


The rest of this page provides links to resources for :ref:`learning basic
Python programming <learn_python>` (a necessary prerequisite to using any
Python module, and MNE-Python is no exception), as well as some notes on the
:ref:`design philosophy of MNE-Python <design_philosophy>` that may help orient
new users to what MNE-Python does and does not do.


.. _learn_python:

Getting started with Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Python`_ is a modern general-purpose object-oriented high-level programming
language. There are many general introductions to Python online; here are a
few:

- The official `Python tutorial <https://docs.python.org/3/tutorial/index.html>`__
- W3Schools `Python tutorial <https://www.w3schools.com/python/>`__
- Software Carpentry's `Python lesson <http://swcarpentry.github.io/python-novice-inflammation/>`_

Additionally, here are a couple tutorials focused on scientific programming in
Python:

- the `SciPy Lecture Notes <http://scipy-lectures.org/>`_
- `NumPy for MATLAB users <https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html>`_

There are also many video tutorials online, including `videos from the annual
SciPy conferences
<https://www.youtube.com/user/EnthoughtMedia/playlists?shelf_id=1&sort=dd&view=50>`_.
One of those is a `Python introduction for complete beginners
<https://www.youtube.com/watch?v=Xmxy2NU9LOI>`_, but there are many more
lectures on advanced topics available as well.


.. _design_philosophy:

MNE-Python design philosophy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Interactive versus scripted analysis.** MNE-Python has some great interactive
plotting abilities that can help you explore your data, and there are a few
GUI-like interactive plotting commands (like browsing through the raw data and
clicking to mark bad channels, or click-and-dragging to annotate bad temporal
spans). But in general it is not possible to use MNE-Python to mouse-click your
way to a finished, publishable analysis. MNE-Python works best when you
assemble your analysis pipeline into one or more Python scripts. On the plus
side, your scripts act as a record of everything you did in your analysis,
making it easy to tweak your analysis later and/or share it with others
(including your future self).

**Integration with the scientific python stack.** MNE-Python also integrates
well with other standard scientific python libraries. For example, MNE-Python
objects underlyingly store their data in NumPy arrays, making it easy to apply
custom algorithms or pass your data into one of `scikit-learn`_'s machine
learning pipelines. MNE-Python's 2-D plotting functions also return matplotlib
figure objects, and the 3D plotting functions return mayavi scenes, so you can
customize your MNE-Python plots using any of matplotlib or mayavi's plotting
commands. The intent is that MNE-Python will get most neuroscientists 90% of
the way to their desired analysis goal, and other packages can get them over
the finish line.

**Submodule-based organization.** A useful-to-know organizing principle is that
MNE-Python objects and functions are separated into submodules. This can help
you discover related functions if you're using an editor that supports
tab-completion. For example, you can type ``mne.preprocessing.<TAB>`` to see
all the functions in the preprocessing submodule; similarly for visualization
functions (:mod:`mne.viz`), functions for reading and writing data
(:mod:`mne.io`), statistics (:mod:`mne.stats`), etc.  This also helps save
keystrokes — instead of::

    import mne
    mne.preprocessing.eog.peak_finder(...)
    mne.preprocessing.eog.find_eog_events(...)
    mne.preprocessing.eog.create_eog_epochs(...)

you can import submodules directly, and use just the submodule name to access
its functions::

    from mne.preprocessing import eog
    eog.peak_finder(...)
    eog.find_eog_events(...)
    eog.create_eog_epochs(...)

**Internal representation** When importing data, MNE-Python will always convert
measurements to the same standard units. Thus the in-memory representation of
data are always in:

- Volts (eeg, eog, seeg, emg, ecg, bio, ecog)
- Teslas (magnetometers)
- Teslas/meter (gradiometers)
- Amperes*meter (dipole fits)
- Molar (aka mol/L) (fNIRS data: oxyhemoglobin (hbo), deoxyhemoglobin (hbr))
- Arbitrary units (various derived unitless quantities)

**Floating-point precision** MNE-Python performs all computation in memory
using the double-precision 64-bit floating point format. This means that the
data is typecast into float64 format as soon as it is read into memory. The
reason for this is that operations such as filtering and preprocessing are
more accurate when using the 64-bit format. However, for backward
compatibility, MNE-Python writes ``.fif`` files in a 32-bit format by default.
This reduces file size when saving data to disk, but beware that saving
*intermediate results* to disk and re-loading them from disk later may lead to
loss in precision. If you would like to ensure 64-bit precision, there are two
possibilities:

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_io">Data I/O and datasets</a>
          </h4>
        </div>
        <div id="collapse_io" class="panel-collapse collapse">
          <div class="panel-body">

**Getting your data into MNE**

.. toctree::
    :maxdepth: 1

    manual/io.rst
    auto_tutorials/plot_creating_data_structures.rst
    auto_tutorials/plot_metadata_epochs.rst
    auto_tutorials/plot_modifying_data_inplace.rst
    auto_tutorials/plot_ecog.rst
    manual/memory.rst
    manual/migrating.rst

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
    auto_tutorials/plot_phantom_4DBTi.rst
    auto_tutorials/plot_sleep.rst
    auto_examples/datasets/plot_brainstorm_data.rst
    auto_examples/datasets/plot_opm_data.rst
    auto_examples/datasets/plot_megsim_data.rst
    auto_examples/datasets/plot_megsim_data_single_trial.rst
    auto_examples/datasets/spm_faces_dataset.rst

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a id="preprocessing" class="anchor-doc"></a>
            <a data-toggle="collapse" href="#collapse_preprocessing">Preprocessing (filtering, SSP, ICA, Maxwell filtering, ...)</a>
          </h4>
        </div>
        <div id="collapse_preprocessing" class="panel-collapse collapse">
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
    auto_examples/preprocessing/plot_ica_comparison.rst
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
            <a id="visualization" class="anchor-doc"></a>
            <a data-toggle="collapse" href="#collapse_visualization">Visualization</a>
          </h4>
        </div>
        <div id="collapse_visualization" class="panel-collapse collapse">
          <div class="panel-body">

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_visualize_raw.rst
    auto_tutorials/plot_visualize_epochs.rst
    auto_tutorials/plot_visualize_evoked.rst
    auto_tutorials/plot_visualize_stc.rst
    auto_tutorials/plot_whitened.rst
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
    auto_examples/visualization/plot_xhemi.rst

.. raw:: html

    </details>
.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a id="time-freq" class="anchor-doc"></a>
            <a data-toggle="collapse" href="#collapse_tf">Time- and frequency-domain analyses</a>
          </h4>
        </div>
        <div id="collapse_tf" class="panel-collapse collapse">
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
    auto_examples/time_frequency/plot_compute_csd.rst

.. raw:: html

  </details>

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a id="source-analysis" class="anchor-doc"></a>
            <a data-toggle="collapse" href="#collapse_source">Source-level analysis</a>
          </h4>
        </div>
        <div id="collapse_source" class="panel-collapse collapse">
          <div class="panel-body">

**Background**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_background_freesurfer.rst
    manual/source_localization/forward.rst
    manual/source_localization/inverse.rst
    manual/source_localization/morph_stc.rst

**Getting data to source space**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_source_alignment.rst
    auto_tutorials/plot_forward.rst
    auto_tutorials/plot_compute_covariance.rst
    auto_tutorials/plot_mne_dspm_source_localization.rst
    auto_tutorials/plot_mne_solutions.rst
    auto_tutorials/plot_dipole_fit.rst
    auto_tutorials/plot_point_spread.rst
    auto_tutorials/plot_dipole_orientations.rst
    auto_tutorials/plot_dics.rst


.. raw:: html

  <details class="example_details">
  <summary><strong>Forward examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/forward/plot_decimate_head_surface.rst
    auto_examples/forward/plot_forward_sensitivity_maps.rst
    auto_examples/forward/plot_left_cerebellum_volume_source.rst
    auto_examples/forward/plot_source_space_morphing.rst

.. raw:: html

    </details>

.. raw:: html

  <details class="example_details">
  <summary><strong>Inverse examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/datasets/plot_opm_rest_data.rst
    auto_examples/inverse/plot_compute_mne_inverse_epochs_in_label.rst
    auto_examples/inverse/plot_compute_mne_inverse_raw_in_label.rst
    auto_examples/inverse/plot_compute_mne_inverse_volume.rst
    auto_examples/inverse/plot_covariance_whitening_dspm.rst
    auto_examples/inverse/plot_custom_inverse_solver.rst
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
    auto_examples/inverse/plot_morph_surface_stc.rst
    auto_examples/inverse/plot_morph_volume_stc.rst
    auto_examples/inverse/plot_rap_music.rst
    auto_examples/inverse/plot_read_stc.rst
    auto_examples/inverse/plot_read_inverse.rst
    auto_examples/inverse/plot_read_source_space.rst
    auto_examples/inverse/plot_snr_estimate.rst
    auto_examples/inverse/plot_tf_dics.rst
    auto_examples/inverse/plot_tf_lcmv.rst
    auto_examples/inverse/plot_time_frequency_mixed_norm_inverse.rst
    auto_examples/inverse/plot_vector_mne_solution.rst

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
            <a data-toggle="collapse" href="#collapse_statistics">Statistics</a>
          </h4>
        </div>
        <div id="collapse_statistics" class="panel-collapse collapse">
          <div class="panel-body">


**Background**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_background_statistics.rst

**Sensor Space**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_stats_spatio_temporal_cluster_sensors.rst
    auto_tutorials/plot_stats_cluster_1samp_test_time_frequency.rst
    auto_tutorials/plot_stats_cluster_time_frequency.rst
    auto_tutorials/plot_stats_cluster_erp.rst

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
            <a id="machine-learning" class="anchor-doc"></a>
            <a data-toggle="collapse" href="#collapse_ml">Machine learning (decoding, encoding, MVPA)</a>
          </h4>
        </div>
        <div id="collapse_ml" class="panel-collapse collapse">
          <div class="panel-body">

**Decoding**

.. toctree::
    :maxdepth: 1

    auto_tutorials/plot_sensors_decoding.rst

.. raw:: html

  <details class="example_details">
  <summary><strong>Examples</strong></summary>

.. toctree::
    :maxdepth: 1

    auto_examples/decoding/decoding_rsa.rst
    auto_examples/decoding/plot_decoding_csp_eeg.rst
    auto_examples/decoding/plot_decoding_csp_timefreq.rst
    auto_examples/decoding/plot_decoding_spatio_temporal_source.rst
    auto_examples/decoding/plot_decoding_spoc_CMC.rst
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

    auto_examples/decoding/plot_receptive_field_mtrf.rst

.. raw:: html

    </details>

.. raw:: html

          </div>
        </div>
      </div>


      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a id="connectivity" class="anchor-doc"></a>
            <a data-toggle="collapse" href="#collapse_connectivity">Connectivity</a>
          </h4>
        </div>
        <div id="collapse_connectivity" class="panel-collapse collapse">
          <div class="panel-body">

**Examples**

.. toctree::
    :maxdepth: 1

    auto_examples/connectivity/plot_cwt_sensor_connectivity.rst
    auto_examples/connectivity/plot_mixed_source_space_connectivity.rst
    auto_examples/connectivity/plot_mne_inverse_coherence_epochs.rst
    auto_examples/connectivity/plot_mne_inverse_envelope_correlation.rst
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
            <a data-toggle="collapse" href="#collapse_realtime">Realtime</a>
          </h4>
        </div>
        <div id="collapse_realtime" class="panel-collapse collapse">
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
    manual/source_localization/c_forward.rst
    manual/source_localization/c_inverse.rst
    manual/source_localization/c_morph.rst
    manual/appendix/bem_model.rst
    manual/appendix/c_misc.rst
    manual/appendix/c_release_notes.rst
    manual/appendix/c_EULA.rst
    manual/appendix/martinos.rst

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
