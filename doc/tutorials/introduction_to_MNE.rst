.. _introduction_to_mne:

Introduction to MNE
===================

This tutorial will give you an introduction to MNE using the Python programming language. You can find several tutorials linked to the gallery for M/EEG data from preprocessing to more sophisticated analysis. These tutorials are however capturing only the most important information, you can then find more details on the :ref:`manual`, and look at :ref:`api_reference` for specific Python function and class usage information.


Background
----------

MNE started as tool written in C by Matti Hamalainen while at MGH in Boston.
Later on, MNE was extended with the Python programming language.
MNE-Python now implements almost all MNE-C’s functionality and offers transparent scripting.
On top of that it extends MNE-C’s functionality considerably (customize events, compute contrasts, group statistics, time-frequency analysis, EEG-sensor space analyses, multivariate statistics, various source localization algorithms etc.) If you have being using the MNE-C commands in the past, there is no need to convert your fif files to a new system or database. MNE-Python works nicely with the historical fif files, yet it also allows you to import data in many other formats (see :ref:`IO functions <ch_convert>`).

**figure/diagram plot**


What can you do with MNE using Python?
--------------------------------------

   - **Raw data visualization** to visualize recordings, can also use
     *mne_browse_raw* for extended functionality (see :ref:`ch_browse`)
   - **Epoching**: Define epochs, baseline correction, handle conditions etc.
   - **Averaging** to get Evoked data
   - **Compute SSP pojectors** to remove ECG and EOG artifacts
   - **Compute ICA** to remove artifacts or select latent sources.
   - **Maxwell filtering** to remove environmental noise.
   - **Boundary Element Modeling**: single and three-layer BEM model
     creation and solution computation.
   - **Forward modeling**: BEM computation and mesh creation
     (see :ref:`ch_forward`)
   - **Linear inverse solvers** (dSPM, sLORETA, MNE, LCMV, DICS)
   - **Sparse inverse solvers** (L1/L2 mixed norm MxNE, Gamma Map,
     Time-Frequency MxNE, RAP-MUSIC)
   - **Connectivity estimation** in sensor and source space
   - **Visualization of sensor and source space data**
   - **Time-frequency** analysis with Morlet wavelets (induced power,
     intertrial coherence, phase lock value) also in the source space
   - **Spectrum estimation** using multi-taper method
   - **Mixed Source Models** combining cortical and subcortical structures
   - **Dipole Fitting**
   - **Decoding** multivariate pattern analyis of M/EEG topographies
   - **Compute contrasts** between conditions, between sensors, across
     subjects etc.
   - **Non-parametric statistics** in time, space and frequency
     (including cluster-level)
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
      models -- use Freesurfer.

.. note:: This package is based on the FIF file format from Neuromag. But, it
          can read and convert CTF, BTI/4D, KIT and various EEG formats to
          FIF (see :ref:`IO functions <ch_convert>`).


Installation of the required materials
---------------------------------------

See :ref:`getting_started` with Python.

.. note:: The default location for the MNE-sample data is
    my-path-to/mne-python/examples. If you downloaded data and an example asks
    you whether to download it again, make sure
    the data reside in the examples directory
    and that you run the script from its current directory.

    From IPython e.g. say::
    
        cd examples/preprocessing
        %run plot_find_ecg_artifacts.py


See :ref:`datasets` for a list of all available datasets and some
advanced configuration options, e.g. to specify a custom
location for storing the datasets.

Want to know more ?
-------------------

Browse out the next :ref:`tutorials` for more details, and :ref:`general_examples`.
