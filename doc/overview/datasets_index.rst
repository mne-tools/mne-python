.. _datasets:

Datasets Overview
#################

.. sidebar:: Contributing datasets to MNE-Python

    Do not hesitate to contact MNE-Python developers on the
    `MNE mailing list <http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis>`_
    to discuss the possibility of adding more publicly available datasets.

All the dataset fetchers are available in :mod:`mne.datasets`. To download any of the datasets,
use the ``data_path`` (fetches full dataset) or the ``load_data`` (fetches dataset partially) functions.

All fetchers will check the default download location first to see if the dataset
is already on your computer, and only download it if necessary. The default
download location is also configurable; see the documentation of any of the
``data_path`` functions for more information.

.. contents:: Available datasets
   :local:
   :depth: 2


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

    * :ref:`ex-envelope-correlation`

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

The EEGBCI dataset is documented in [2]_. The data set is available at PhysioNet [3]_.
The dataset contains 64-channel EEG recordings from 109 subjects and 14 runs on each subject in EDF+ format.
The recordings were made using the BCI2000 system. To load a subject, do::

    from mne.io import concatenate_raws, read_raw_edf
    from mne.datasets import eegbci
    raw_fnames = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

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
and inanimate objects either natural or artificial [4]_. Given the high number
of conditions this dataset is well adapted to an approach based on
Representational Similarity Analysis (RSA).

.. topic:: Examples

    * :ref:`Representational Similarity Analysis (RSA) <ex-rsa-noplot>`: Partially replicates the results from Cichy et al. (2014).


mTRF Dataset
============
:func:`mne.datasets.mtrf.data_path`.

This dataset contains 128 channel EEG as well as natural speech stimulus features,
which is also available `here <https://sourceforge.net/projects/aespa/files/>`_.

The experiment consisted of subjects listening to natural speech.
The dataset contains several feature representations of the speech stimulus,
suitable for using to fit continuous regression models of neural activity.
More details and a description of the package can be found in [5]_.

.. topic:: Examples

    * :ref:`Receptive Field Estimation and Prediction <ex-receptive-field-mtrf>`: Partially replicates the results from Crosse et al. (2016).


.. _kiloword-dataset:

Kiloword dataset
================
:func:`mne.datasets.kiloword.data_path`.

This dataset consists of averaged EEG data from 75 subjects performing a lexical decision
task on 960 English words [6]_. The words are richly annotated, and can be used for e.g.
multiple regression estimation of EEG correlates of printed word processing.


4D Neuroimaging / BTi dataset
=============================
:func:`mne.datasets.phantom_4dbti.data_path`.

This dataset was obtained with a phantom on a 4D Neuroimaging / BTi system at the MEG
center in La Timone hospital in Marseille.

.. topic:: Examples

    * :ref:`tut_phantom_4Dbti`

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
data please cite [7]_ and [8]_.

.. topic:: Examples

    * :ref:`tut-sleep-stage-classif`

Miscellaneous Datasets
======================
These datasets are used for specific purposes in the documentation and in
general are not useful for separate analyses.

.. _fsaverage:

fsaverage
^^^^^^^^^
:func:`mne.datasets.fetch_fsaverage`

For convenience, we provide a function to separately download and extract the
(or update an existing) fsaverage subject.

.. topic:: Examples

    :ref:`tut-eeg-fsaverage-source-modeling`


ECoG Dataset
^^^^^^^^^^^^
:func:`mne.datasets.misc.data_path`. Data exists at ``/ecog/sample_ecog.mat``.

This dataset contains a sample Electrocorticography (ECoG) dataset. It includes
a single grid of electrodes placed over the temporal lobe during an auditory
listening task. This dataset is primarily used to demonstrate visualization
functions in MNE and does not contain useful metadata for analysis.

.. topic:: Examples

    * :ref:`How to convert 3D electrode positions to a 2D image.
      <ex-electrode-pos-2d>`: Demonstrates
      how to project a 3D electrode location onto a 2D image, a common procedure
      in electrocorticography.

.. _limo-dataset:

LIMO Dataset
^^^^^^^^^^^^
:func:`mne.datasets.limo.load_data`.

In the original LIMO experiment (see [9]_), participants performed a
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

References
==========

.. [2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE TBME 51(6):1034-1043

.. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220

.. [4] Cichy, R. M., Pantazis, D., & Oliva, A. Resolving human object recognition in space and time. Nature Neuroscience (2014): 17(3), 455-462

.. [5] Crosse, M. J., Di Liberto, G. M., Bednar, A., & Lalor, E. C. The Multivariate Temporal Response Function (mTRF) Toolbox: A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli. Frontiers in Human Neuroscience (2016): 10.

.. [6] Dufau, S., Grainger, J., Midgley, KJ., Holcomb, PJ. A thousand words are worth a picture: Snapshots of printed-word processing in an event-related potential megastudy. Psychological science, 2015

.. [7] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000). https://ieeexplore.ieee.org/document/867928

.. [8] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/cgi/content/full/101/23/e215]; 2000 (June 13).

.. [9] Rousselet, G. A., Gaspar, C. M., Pernet, C. R., Husk, J. S., Bennett, P. J., & Sekuler, A. B. (2010). Healthy aging delays scalp EEG sensitivity to noise in a face discrimination task. Frontiers in psychology, 1, 19. https://doi.org/10.3389/fpsyg.2010.00019

.. _auditory dataset tutorial: https://neuroimage.usc.edu/brainstorm/DatasetAuditory
.. _resting state dataset tutorial: https://neuroimage.usc.edu/brainstorm/DatasetResting
.. _median nerve dataset tutorial: https://neuroimage.usc.edu/brainstorm/DatasetMedianNerveCtf
.. _SPM faces dataset: https://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/
