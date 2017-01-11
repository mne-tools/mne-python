:orphan:

.. include:: links.inc

.. _get_started:

Get started
============

.. _what_can_you_do:

Installation
------------

To get started with MNE, visit the installation instructions for
the :ref:`MNE<install_python_and_mne_python>` and
:ref:`MNE-C <install_mne_c>`:

.. container:: row

  .. container:: panel panel-default halfpad

    .. container:: panel-heading nosize

      MNE python module

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 2

        install_mne_python

  .. container:: panel panel-default halfpad

    .. container:: panel-heading nosize

      MNE-C

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 2

        install_mne_c

.. container:: row

  .. container:: col-md-8

    .. raw:: html

      <h2>What can you do with MNE?</h2>

    - **Raw data visualization** to visualize recordings
      (see :ref:`general_examples` for more).
    - **Epoching**: Define epochs, baseline correction, handle conditions etc.
    - **Averaging** to get Evoked data.
    - **Compute SSP projectors** to remove ECG and EOG artifacts.
    - **Compute ICA** to remove artifacts or select latent sources.
    - **Maxwell filtering** to remove environmental noise.
    - **Boundary Element Modeling**: single and three-layer BEM model
      creation and solution computation.
    - **Forward modeling**: BEM computation and mesh creation
      (see :ref:`ch_forward`).
    - **Linear inverse solvers** (dSPM, sLORETA, MNE, LCMV, DICS).
    - **Sparse inverse solvers** (L1/L2 mixed norm MxNE, Gamma Map,
      Time-Frequency MxNE, RAP-MUSIC).
    - **Connectivity estimation** in sensor and source space.
    - **Visualization of sensor and source space data**
    - **Time-frequency** analysis with Morlet wavelets (induced power,
      intertrial coherence, phase lock value) also in the source space.
    - **Spectrum estimation** using multi-taper method.
    - **Mixed Source Models** combining cortical and subcortical structures.
    - **Dipole Fitting**
    - **Decoding** multivariate pattern analysis of M/EEG topographies.
    - **Compute contrasts** between conditions, between sensors, across
      subjects etc.
    - **Non-parametric statistics** in time, space and frequency
      (including cluster-level).
    - **Scripting** (batch and parallel computing)


    There is much more beyond these general categories such as:

    - detect heart beat QRS component
    - detect eye blinks and EOG artifacts
    - compute SSP projections to remove ECG or EOG artifacts
    - compute Independent Component Analysis (ICA) to remove artifacts or
      select latent sources
    - estimate noise covariance matrix from Raw and Epochs
    - visualize cross-trial response dynamics using epochs images
    - compute forward solutions
    - estimate power in the source space
    - estimate connectivity in sensor and source space
    - morph stc from one brain to another for group studies
    - compute mass univariate statistics base on custom contrasts
    - visualize source estimates
    - export raw, epochs, and evoked data to other python data analysis
      libraries e.g. pandas
    - Raw movement compensation as you would do with Elekta Maxfilter™
    - and many more things ...


    What you're not supposed to do with MNE:

    - **Brain and head surface segmentation** for use with BEM
      models -- use Freesurfer_.

  .. container:: col-md-4

    .. container:: panel panel-info nosize nopad

      .. container:: panel-heading nosize

        Historical notes

      .. container:: panel-body nosize

        MNE started as tool written in C by Matti Hämäläinen while at MGH in
        Boston.

        - :ref:`MNE-C <c_reference>` is Matti's C code.

        - The MNE python module was built in the Python programming language to
          reimplement all MNE-C’s functionality, offer transparent scripting,
          and extend MNE-C’s functionality considerably (see left). Thus it is
          the primary focus of this documentation.

        - :ref:`ch_matlab` is available mostly to allow reading and writing
          FIF files.

        - :ref:`mne_cpp`  aims to provide modular and open-source tools for
          real-time acquisition, visualization, and analysis. It provides
          a :ref:`separate website <mne_cpp>` for documentation and releases.

        The MNE tools are based on the FIF file format from Neuromag.
        However, MNE can read native CTF, BTI/4D, KIT and various
        EEG formats (see :ref:`IO functions <ch_convert>`).

        If you have been using MNE-C, there is no need to convert your fif
        files to a new system or database -- MNE works nicely with
        the historical fif files.
