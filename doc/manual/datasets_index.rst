.. _datasets:

Datasets
########

All the dataset fetchers are available in :mod:`mne.datasets`. To download any of the datasets,
use the ``data_path`` (fetches full dataset) or the ``load_data`` (fetches dataset partially) functions.

All fetchers will check the default download location first to see if the dataset
is already on your computer, and only download it if necessary. The default
download location is also configurable; see the documentation of any of the
``data_path`` functions for more information.

.. sidebar:: Contributing datasets in MNE-Python

    Do not hesitate to contact MNE-Python developers on the
    `MNE mailing list <http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis>`_
    to discuss the possibility to add more publicly available datasets.

.. contents:: Available datasets
   :local:
   :depth: 2


.. _sample-dataset:

Sample
======
:func:`mne.datasets.sample.data_path`

:ref:`ch_sample_data` is recorded using a 306-channel Neuromag vectorview system.

In this experiment, checkerboard patterns were presented to the subject
into the left and right visual field, interspersed by tones to the
left or right ear. The interval between the stimuli was 750 ms. Occasionally
a smiley face was presented at the center of the visual field.
The subject was asked to press a key with the right index finger
as soon as possible after the appearance of the face.

In MNE-Python, the **sample** dataset is distributed with :ref:`fsaverage` for
convenience.

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

MEGSIM
======
:func:`mne.datasets.megsim.load_data`

This dataset contains experimental and simulated MEG data. To load data from this dataset, do::

    from mne.io import Raw
    from mne.datasets.megsim import load_data
    raw_fnames = load_data(condition='visual', data_format='raw', data_type='experimental', verbose=True)
    raw = Raw(raw_fnames[0])

Detailed description of the dataset can be found in the related publication [1]_.

.. topic:: Examples

    * :ref:`ex-megsim`

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
  sensor runs on an indepent clock. Synchronization turned out to be satisfactory

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

References
==========

.. [1] Aine CJ, Sanfratello L, Ranken D, Best E, MacArthur JA, Wallace T, Gilliam K, Donahue CH, Montano R, Bryant JE, Scott A, Stephen JM (2012) MEG-SIM: A Web Portal for Testing MEG Analysis Methods using Realistic Simulated and Empirical Data. Neuroinform 10:141-158

.. [2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE TBME 51(6):1034-1043

.. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220

.. [4] Cichy, R. M., Pantazis, D., & Oliva, A. Resolving human object recognition in space and time. Nature Neuroscience (2014): 17(3), 455-462

.. [5] Crosse, M. J., Di Liberto, G. M., Bednar, A., & Lalor, E. C. The Multivariate Temporal Response Function (mTRF) Toolbox: A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli. Frontiers in Human Neuroscience (2016): 10.

.. [6] Dufau, S., Grainger, J., Midgley, KJ., Holcomb, PJ. A thousand words are worth a picture: Snapshots of printed-word processing in an event-related potential megastudy. Psychological science, 2015

.. [7] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000). https://ieeexplore.ieee.org/document/867928

.. [8] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/cgi/content/full/101/23/e215]; 2000 (June 13).


.. _auditory dataset tutorial: https://neuroimage.usc.edu/brainstorm/DatasetAuditory
.. _resting state dataset tutorial: https://neuroimage.usc.edu/brainstorm/DatasetResting
.. _median nerve dataset tutorial: https://neuroimage.usc.edu/brainstorm/DatasetMedianNerveCtf
.. _SPM faces dataset: https://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/
