

.. _tut_creating_data_structures:

Getting started
===============

This tutorial will give you an introduction into what is MNE-Python, and how to use it. You can find several tutorials linked to the gallery for M/EEG data from preprocessing to more sophisticated analysis. These tutorials are however capturing only the most important information, you can then find more details on the **Manual**. 


What is MNE-Python
------------------
MNE-Python reimplements most of MNE-C’s (the original MNE command line utils) functionality and offers transparent scripting. On top of that it extends MNE-C’s functionality considerably (customize events, compute contrasts, group statistics, time-frequency analysis, EEG-sensor space analyses , etc.) It uses the same files as standard MNE unix commands: no need to convert your files to a new system or database.

**figure/diagram plot**


What can you do with MNE Python?
--------------------------------

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
     Time-Frequency MxNE)
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
No!

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
    - and many more things ...


What you're not supposed to do with MNE-Python
----------------------------------------------

    - **Brain and head surface segmentation** for use with BEM
      models -- use Freesurfer.
    - **Raw movement compensation** -- use Elekta Maxfilter™

.. note:: This package is based on the FIF file format from Neuromag. It
          can read and convert CTF, BTI/4D, KIT and various EEG formats to
          FIF.


Installation of the required materials
---------------------------------------

See :ref:`getting_started` with Python.

.. note:: The expected location for the MNE-sample data is
    my-path-to/mne-python/examples. If you downloaded data and an example asks
    you whether to download it again, make sure
    the data reside in the examples directory and you run the script from its
    current directory.

    From IPython e.g. say::
    
        cd examples/preprocessing

    %run plot_find_ecg_artifacts.py


Want to know more ?
-------------------

Browse out the next tutorials for more details, and :ref:`examples-index` gallery.
