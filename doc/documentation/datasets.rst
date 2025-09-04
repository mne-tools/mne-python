.. _datasets:

Datasets Overview
#################

.. note:: Contributing datasets to MNE-Python
    :class: sidebar

    Do not hesitate to contact MNE-Python developers on the
    `MNE Forum <https://mne.discourse.group>`_ to discuss the possibility of
    adding more publicly available datasets.

All the dataset fetchers are available in :mod:`mne.datasets`. To download any of the datasets,
use the ``data_path`` (fetches full dataset) or the ``load_data`` (fetches dataset partially) functions.

All fetchers will check the default download location first to see if the dataset
is already on your computer, and only download it if necessary. The default
download location is also configurable; see the documentation of any of the
``data_path`` functions for more information.

.. _sample-dataset:

Sample
======
:func:`mne.datasets.sample.data_path`

These data were acquired with the Neuromag
Vectorview system at MGH/HMS/MIT Athinoula A. Martinos Center Biomedical
Imaging. EEG data from a 60-channel electrode cap was acquired simultaneously with
the MEG. The original MRI data set was acquired with a Siemens 1.5 T
Sonata scanner using an MPRAGE sequence.

.. note:: These data are provided solely for the purpose of getting familiar
          with the MNE software. The data should not be used to evaluate the
          performance of the MEG or MRI system employed.

In this experiment, checkerboard patterns were presented to the subject
into the left and right visual field, interspersed by tones to the
left or right ear. The interval between the stimuli was 750 ms. Occasionally
a smiley face was presented at the center of the visual field.
The subject was asked to press a key with the right index finger
as soon as possible after the appearance of the face.

.. table:: Trigger codes for the sample data set.

    =========  =====  ==========================================
    Name              Contents
    =========  =====  ==========================================
    LA         1      Response to left-ear auditory stimulus
    RA         2      Response to right-ear auditory stimulus
    LV         3      Response to left visual field stimulus
    RV         4      Response to right visual field stimulus
    smiley     5      Response to the smiley face
    button     32     Response triggered by the button press
    =========  =====  ==========================================

Contents of the data set
^^^^^^^^^^^^^^^^^^^^^^^^

The sample data set contains two main directories: ``MEG/sample`` (the MEG/EEG
data) and ``subjects/sample`` (the MRI reconstructions).
In addition to subject ``sample``, the MRI surface reconstructions from another
subject, morph, are provided to demonstrate morphing capabilities.

.. table:: Contents of the MEG/sample directory.

    ========================  =====================================================================
    File                      Contents
    ========================  =====================================================================
    sample/audvis_raw.fif     The raw MEG/EEG data
    audvis.ave                A template script for off-line averaging
    auvis.cov                 A template script for the computation of a noise-covariance matrix
    ========================  =====================================================================

.. table:: Overview of the contents of the subjects/sample directory.

    =======================  ======================================================================
    File / directory         Contents
    =======================  ======================================================================
    bem                      Directory for the forward modelling data
    bem/watershed            BEM surface segmentation data computed with the watershed algorithm
    bem/inner_skull.surf     Inner skull surface for BEM
    bem/outer_skull.surf     Outer skull surface for BEM
    bem/outer_skin.surf      Skin surface for BEM
    sample-head.fif          Skin surface in fif format for mne_analyze visualizations
    surf                     Surface reconstructions
    mri/T1                   The T1-weighted MRI data employed in visualizations
    =======================  ======================================================================

The following preprocessing steps have been already accomplished
in the sample data set:

- The MRI surface reconstructions have
  been computed using the FreeSurfer software.

- The BEM surfaces have been created with the watershed algorithm,
  see :ref:`bem_watershed_algorithm`.

The **sample** dataset is distributed with :ref:`fsaverage` for convenience.

.. _ucl-opm-auditory-dataset:

UCL OPM Auditory
================
:func:`mne.datasets.ucl_opm_auditory.data_path`.

A basic auditory evoked field experiment using an OPM setup from FIL at UCL.
See :footcite:`SeymourEtAl2022` for details.

.. topic:: Examples

    * :ref:`tut-opm-processing`

Brainstorm
==========
Dataset fetchers for three Brainstorm tutorials are available. Users must agree to the
license terms of these datasets before downloading them. These files are recorded in a CTF 275 system
and are provided in native CTF format (.ds files).

Auditory
^^^^^^^^
:func:`mne.datasets.brainstorm.bst_raw.data_path`.

Details about the data can be found at the Brainstorm `auditory dataset tutorial`_.

.. topic:: Examples

    * :ref:`tut-brainstorm-auditory`: Partially replicates the original Brainstorm tutorial.

Resting state
^^^^^^^^^^^^^
:func:`mne.datasets.brainstorm.bst_resting.data_path`

Details can be found at the Brainstorm `resting state dataset tutorial`_.

.. topic:: Examples

    * :ref:`mne-connectivity:ex-envelope-correlation`

Median nerve
^^^^^^^^^^^^
:func:`mne.datasets.brainstorm.bst_raw.data_path`

Details can be found at the Brainstorm `median nerve dataset tutorial`_.

.. topic:: Examples

    * :ref:`ex-brainstorm-raw`

SPM faces
=========
:func:`mne.datasets.spm_face.data_path`

The `SPM faces dataset`_ contains EEG, MEG and fMRI recordings on face perception.

.. topic:: Examples

    * :ref:`ex-spm-faces` Full pipeline including artifact removal, epochs averaging, forward model computation and source reconstruction using dSPM on the contrast: "faces - scrambled".

EEGBCI motor imagery
====================
:func:`mne.datasets.eegbci.load_data`

The EEGBCI dataset is documented in :footcite:`SchalkEtAl2004` and on the
`PhysioNet documentation page <https://physionet.org/content/eegmmidb/1.0.0/>`_.
The data set is available at PhysioNet :footcite:`GoldbergerEtAl2000`.
It contains 64-channel EEG recordings from 109 subjects and 14 runs on each
subject in EDF+ format. The recordings were made using the BCI2000 system.
To load a subject, do::

    from mne.io import concatenate_raws, read_raw_edf
    from mne.datasets import eegbci
    subjects = [1]  # may vary
    runs = [4, 8, 12]  # may vary
    raw_fnames = eegbci.load_data(subjects, runs)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    # concatenate runs from subject
    raw = concatenate_raws(raws)
    # make channel names follow standard conventions
    eegbci.standardize(raw)

.. topic:: Examples

    * :ref:`ex-decoding-csp-eeg`

.. _somato-dataset:

Somatosensory
=============
:func:`mne.datasets.somato.data_path`

This dataset contains somatosensory data with event-related synchronizations
(ERS) and desynchronizations (ERD).

.. topic:: Examples

    * :ref:`tut-sensors-time-freq`
    * :ref:`ex-inverse-source-power`
    * :ref:`ex-time-freq-global-field-power`

Multimodal
==========
:func:`mne.datasets.multimodal.data_path`

This dataset contains a single subject recorded at Otaniemi (Aalto University)
with auditory, visual, and somatosensory stimuli.

.. topic:: Examples

    * :ref:`ex-io-ave-fiff`

.. _fnirs-motor-dataset:

fNIRS motor
===========
:func:`mne.datasets.fnirs_motor.data_path`

This dataset contains a single subject recorded at Macquarie University.
It has optodes placed over the motor cortex. There are three conditions:

- tapping the left thumb to fingers
- tapping the right thumb to fingers
- a control where nothing happens

The tapping lasts 5 seconds, and there are 30 trials of each condition.

.. topic:: Examples

    * :ref:`tut-fnirs-processing`

High frequency SEF
==================
:func:`mne.datasets.hf_sef.data_path()`

This dataset contains somatosensory evoked fields (median nerve stimulation)
with thousands of epochs. It was recorded with an Elekta TRIUX MEG device at
a sampling frequency of 3 kHz. The dataset is suitable for investigating
high-frequency somatosensory responses. Data from two subjects are included
with MRI images in DICOM format and FreeSurfer reconstructions.

.. topic:: Examples

    * :ref:`high-frequency SEF responses <ex-hf-sef-data>`.

Visual 92 object categories
===========================
:func:`mne.datasets.visual_92_categories.data_path`.

This dataset is recorded using a 306-channel Neuromag vectorview system.

Experiment consisted in the visual presentation of 92 images of human, animal
and inanimate objects either natural or artificial :footcite:`CichyEtAl2014`.
Given the high number of conditions this dataset is well adapted to an approach
based on Representational Similarity Analysis (RSA).

.. topic:: Examples

    * :ref:`Representational Similarity Analysis (RSA) <ex-rsa-noplot>`: Partially replicates the results from :footcite:`CichyEtAl2014`.


mTRF Dataset
============
:func:`mne.datasets.mtrf.data_path`.

This dataset contains 128 channel EEG as well as natural speech stimulus features,
which is also available `here <https://sourceforge.net/projects/aespa/files/>`_.

The experiment consisted of subjects listening to natural speech.
The dataset contains several feature representations of the speech stimulus,
suitable for using to fit continuous regression models of neural activity.
More details and a description of the package can be found in
:footcite:`CrosseEtAl2016`.

.. topic:: Examples

    * :ref:`Receptive Field Estimation and Prediction <ex-receptive-field-mtrf>`: Partially replicates the results from :footcite:`CrosseEtAl2016`.


.. _kiloword-dataset:

Kiloword dataset
================
:func:`mne.datasets.kiloword.data_path`.

This dataset consists of averaged EEG data from 75 subjects performing a
lexical decision task on 960 English words :footcite:`DufauEtAl2015`. The words
are richly annotated, and can be used for e.g. multiple regression estimation
of EEG correlates of printed word processing.


KIT phantom dataset
=============================
:func:`mne.datasets.phantom_kit.data_path`.

This dataset was obtained with a phantom on a KIT system at
Macquarie University in Sydney, Australia.

.. topic:: Examples

    * :ref:`tut-phantom-KIT`


4D Neuroimaging / BTi dataset
=============================
:func:`mne.datasets.phantom_4dbti.data_path`.

This dataset was obtained with a phantom on a 4D Neuroimaging / BTi system at
the MEG center in La Timone hospital in Marseille.

.. topic:: Examples

    * :ref:`tut-phantom-4Dbti`

Kernel OPM phantom dataset
==========================
:func:`mne.datasets.phantom_kernel.data_path`.

This dataset was obtained with a Neuromag phantom in a Kernel Flux (720-sensor)
system at ILABS at the University of Washington. Only 7 out of 42 possible modules
were active for testing purposes, yielding 121 channels of data with limited coverage
(mostly occipital and parietal).

.. topic:: Examples

    * :ref:`ex-kernel-opm-phantom`

OPM
===
:func:`mne.datasets.opm.data_path`

OPM data acquired using an Elekta DACQ, simply piping the data into Elekta
magnetometer channels. The FIF files thus appear to come from a TRIUX system
that is only acquiring a small number of magnetometer channels instead of the
whole array.

The OPM ``coil_type`` is custom, requiring a custom ``coil_def.dat``.
The new ``coil_type`` is 9999.

OPM co-registration differs a bit from the typical SQUID-MEG workflow.
No ``-trans.fif`` file is needed for the OPMs, the FIF files include proper
sensor locations in MRI coordinates and no digitization of RPA/LPA/Nasion.
Thus the MEG<->Head coordinate transform is taken to be an identity matrix
(i.e., everything is in MRI coordinates), even though this mis-identifies
the head coordinate frame (which is defined by the relationship of the
LPA, RPA, and Nasion).

Triggers include:

* Median nerve stimulation: trigger value 257.
* Magnetic trigger (in OPM measurement only): trigger value 260.
  1 second before the median nerve stimulation, a magnetic trigger is piped into the MSR.
  This was to be able to check the synchronization between OPMs retrospectively, as each
  sensor runs on an independent clock. Synchronization turned out to be satisfactory.

.. topic:: Examples

    * :ref:`ex-opm-somatosensory`
    * :ref:`ex-opm-resting-state`

The Sleep PolySomnoGraphic Database
===================================
:func:`mne.datasets.sleep_physionet.age.fetch_data`
:func:`mne.datasets.sleep_physionet.temazepam.fetch_data`

The sleep PhysioNet database contains 197 whole-night PolySomnoGraphic sleep
recordings, containing EEG, EOG, chin EMG, and event markers. Some records also
contain respiration and body temperature. Corresponding hypnograms (sleep
patterns) were manually scored by well-trained technicians according to the
Rechtschaffen and Kales manual, and are also available. If you use these
data please cite :footcite:`KempEtAl2000` and :footcite:`GoldbergerEtAl2000`.

.. topic:: Examples

    * :ref:`tut-sleep-stage-classif`

Reference channel noise MEG data set
====================================
:func:`mne.datasets.refmeg_noise.data_path`.

This dataset was obtained with a 4D Neuroimaging / BTi system at
the University Clinic - Erlangen, Germany. There are powerful bursts of
external magnetic noise throughout the recording, which make it a good
example for automatic noise removal techniques.

.. topic:: Examples

    * :ref:`ex-megnoise_processing`

Miscellaneous Datasets
======================
These datasets are used for specific purposes in the documentation and in
general are not useful for separate analyses.

.. _fsaverage:

fsaverage
^^^^^^^^^
:func:`mne.datasets.fetch_fsaverage`

For convenience, we provide a function to separately download and extract the
(or update an existing) fsaverage subject. See also the
:ref:`background information on fsaverage <fsaverage_background>`.

.. topic:: Examples

    :ref:`tut-eeg-fsaverage-source-modeling`

Infant template MRIs
^^^^^^^^^^^^^^^^^^^^
:func:`mne.datasets.fetch_infant_template`

This function will download an infant template MRI from
:footcite:`OReillyEtAl2021` along with MNE-specific files.

ECoG Dataset
^^^^^^^^^^^^
:func:`mne.datasets.misc.data_path`. Data exists at ``/ecog/``.

This dataset contains a sample electrocorticography (ECoG) dataset. It includes
two grids of electrodes and ten shaft electrodes with simulated motor data (actual data
pending availability).

.. topic:: Examples

    * :ref:`ex-electrode-pos-2d`: Demonstrates how to project a 3D electrode location onto a 2D image, a common procedure in ECoG analyses.
    * :ref:`tut-ieeg-localize`: Demonstrates how to use a graphical user interface to locate electrode contacts as well as warp them to a common atlas.

sEEG Dataset
^^^^^^^^^^^^
:func:`mne.datasets.misc.data_path`. Data exists at ``/seeg/``.

This dataset contains a sample stereoelectroencephalography (sEEG) dataset.
It includes 21 shaft electrodes during a two-choice movement task on a keyboard.

.. topic:: Examples

    * :ref:`tut-ieeg-localize`: Demonstrates how to use a graphical user interface to locate electrode contacts as well as warp them to a common atlas.
    * :ref:`tut-working-with-seeg`: Demonstrates ways to plot sEEG anatomy and results.

.. _limo-dataset:

LIMO Dataset
^^^^^^^^^^^^
:func:`mne.datasets.limo.load_data`.

In the original LIMO experiment (see :footcite:`RousseletEtAl2010`), participants
performed a
two-alternative forced choice task, discriminating between two face stimuli.
Subjects discriminated the same two faces during the whole experiment.
The critical manipulation consisted of the level of noise added to the
face-stimuli during the task, making the faces more or less discernible to the
observer.

The presented faces varied across a noise-signal (or phase-coherence) continuum
spanning from 0 to 100% in increasing steps of 10%. In other words, faces with
high phase-coherence (e.g., 90%) were easy to identify, while faces with low
phase-coherence (e.g., 10%) were hard to identify and by extension hard to
discriminate.

.. topic:: Examples

    * :ref:`Single trial linear regression analysis with the LIMO dataset
      <ex-limo-data>`: Explores data from a single subject of the LIMO dataset
      and demonstrates how to fit a single trial linear regression using the
      information contained in the metadata of the individual datasets.

.. _erp-core-dataset:

ERP CORE Dataset
^^^^^^^^^^^^^^^^
:func:`mne.datasets.erp_core.data_path`

The original `ERP CORE dataset`_ :footcite:`Kappenman2021` contains data from
40 participants who completed 6 EEG experiments, carefully crafted to evoke
7 well-known event-related potential (ERP) components.

Currently, the MNE-Python ERP CORE dataset only provides data from one
participant (subject ``001``) of the Flankers paradigm, which elicits the
lateralized readiness potential (LRP) and error-related negativity (ERN). The
data provided is **not** the original data from the ERP CORE dataset, but
rather a slightly modified version, designed to demonstrate the Epochs metadata
functionality. For example, we already set the references and montage
correctly, and stored events as Annotations. Data is provided in ``FIFF``
format.

.. topic:: Examples

    * :ref:`tut-autogenerate-metadata`: Learn how to auto-generate
      `~mne.Epochs` metadata, and visualize the error-related negativity (ERN)
      ERP component.

.. _ssvep-dataset:

SSVEP
=====
:func:`mne.datasets.ssvep.data_path`

This is a simple example dataset with frequency tagged visual stimulation:
N=2 participants observed checkerboards patterns inverting with a constant
frequency of either 12.0 Hz of 15.0 Hz. 10 trials of 20.0 s length each.
32 channels wet EEG was recorded.

Data format: BrainVision .eeg/.vhdr/.vmrk files organized according to BIDS
standard.

.. topic:: Examples

    * :ref:`tut-ssvep`

.. _eyelink-dataset:

EYELINK
=======
:func:`mne.datasets.eyelink.data_path`

Two small example datasets of eye-tracking data from SR Research EyeLink.

EEG-Eyetracking
^^^^^^^^^^^^^^^
:func:`mne.datasets.eyelink.data_path`. Data exists at ``/eeg-et/``.

Contains both EEG (EGI) and eye-tracking (ASCII format) data recorded from a
pupillary light reflex experiment, stored in separate files. 1 participant fixated
on the screen while short light flashes appeared. Event onsets were recorded by a
photodiode attached to the screen and were sent to both the EEG and eye-tracking
systems.

.. topic:: Examples

    * :ref:`tut-eyetrack`

Freeviewing
^^^^^^^^^^^
:func:`mne.datasets.eyelink.data_path`. Data exists at ``/freeviewing/``.

Contains eye-tracking data (ASCII format) from 1 participant who was free-viewing a
video of a natural scene. In some videos, the natural scene was pixelated such that
the people in the scene were unrecognizable.

.. topic:: Examples

    * :ref:`tut-eyetrack-heatmap`

References
==========

.. footbibliography::


.. LINKS

.. _auditory dataset tutorial: https://neuroimage.usc.edu/brainstorm/DatasetAuditory
.. _resting state dataset tutorial: https://neuroimage.usc.edu/brainstorm/DatasetResting
.. _median nerve dataset tutorial: https://neuroimage.usc.edu/brainstorm/DatasetMedianNerveCtf
.. _SPM faces dataset: https://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/
.. _ERP-CORE dataset: https://erpinfo.org/erp-core
