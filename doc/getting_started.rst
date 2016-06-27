.. include:: links.inc

.. _getting_started:

Getting started
===============

.. _introduction_to_mne:

**MNE** is an academic software package that aims to provide data analysis
pipelines encompassing all phases of M/EEG data processing.

MNE started as tool written in C by Matti Hämäläinen while at MGH in Boston.
MNE was then extended with the Python programming language to implement
nearly all MNE-C’s functionality, offer transparent scripting, and
:ref:`extend MNE-C’s functionality considerably <what_can_you_do>`.

A basic :ref:`ch_matlab` is also available mostly
to allow reading and write MNE files. The sister :ref:`mne_cpp` project
aims to provide modular and open-source tools for acquisition,
visualization, and analysis.

.. note:: This package is based on the FIF file format from Neuromag. But, it
          can read and convert CTF, BTI/4D, KIT and various EEG formats to
          FIF (see :ref:`IO functions <ch_convert>`).

          If you have been using MNE-C, there is no need to convert your fif
          files to a new system or database -- MNE-Python works nicely with
          the historical fif files.

Installation
------------

To get started with MNE, visit the installation instructions for
:ref:`MNE-Python <install_python_and_mne_python>` and
:ref:`MNE-C <install_mne_c>`:

.. container:: span box

  .. raw:: html

    <h3>MNE-Python</h3>

  .. toctree::
    :maxdepth: 2

    install_mne_python

.. container:: span box

  .. raw:: html

    <h3>MNE-C</h3>

  .. toctree::
    :maxdepth: 2

    install_mne_c


.. _what_can_you_do:

What can you do with MNE using Python?
--------------------------------------

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


Is that all you can do with MNE-Python?
---------------------------------------

Short answer is **No**! You can also do:

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


What you're not supposed to do with MNE-Python
----------------------------------------------

    - **Brain and head surface segmentation** for use with BEM
      models -- use Freesurfer_.
