.. _datasets:

Datasets
########

.. contents:: Contents
   :local:
   :depth: 2

All the dataset fetchers are available in :mod:`mne.datasets`. To download any of the datasets,
use the ``data_path`` (fetches full dataset) or the ``load_data`` (fetches dataset partially) functions.

Sample
======
:func:`mne.datasets.sample.data_path()`

:ref:`ch_sample_data` is recorded using a 306-channel Neuromag vectorview system.

In this experiment, checkerboard patterns were presented to the subject
into the left and right visual field, interspersed by tones to the
left or right ear. The interval between the stimuli was 750 ms. Occasionally
a smiley face was presented at the center of the visual field.
The subject was asked to press a key with the right index finger
as soon as possible after the appearance of the face.

Once the ``data_path`` is known, its contents can be examined using :ref:`IO functions <ch_convert>`.

Brainstorm
==========
Dataset fetchers for three Brainstorm tutorials are available. Users must agree to the
license terms of these datasets before downloading them. These files are recorded in a CTF 275 system.
The data is converted to `fif` format before being made available to MNE users. However, MNE-Python now supports
IO for the `ctf` format as well in addition to the C converter utilities. Please consult the :ref:`IO section <ch_convert>` for details.

Auditory
^^^^^^^^
:func:`mne.datasets.brainstorm.bst_raw.data_path()`.

Details about the data can be found at the Brainstorm `auditory dataset tutorial`_.

.. topic:: Examples

    * :ref:`Brainstorm auditory dataset tutorial <sphx_glr_auto_examples_datasets_plot_brainstorm_data.py>`: Partially replicates the original Brainstorm tutorial.

Resting state
^^^^^^^^^^^^^
:func:`mne.datasets.brainstorm.bst_resting.data_path()`

Details can be found at the Brainstorm `resting state dataset tutorial`_.

Median nerve
^^^^^^^^^^^^
:func:`mne.datasets.brainstorm.bst_raw.data_path()`

Details can be found at the Brainstorm `median nerve dataset tutorial`_.

MEGSIM
======
:func:`mne.datasets.megsim.load_data()`

This dataset contains experimental and simulated MEG data. To load data from this dataset, do::

    from mne.io import Raw
    from mne.datasets.megsim import load_data
    raw_fnames = load_data(condition='visual', data_format='raw', data_type='experimental', verbose=True)
    raw = Raw(raw_fnames[0])

Detailed description of the dataset can be found in the related publication [1]_.

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_datasets_plot_megsim_data.py`

SPM faces
=========
:func:`mne.datasets.spm_face.data_path()`

The `SPM faces dataset`_ contains EEG, MEG and fMRI recordings on face perception.

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_datasets_plot_spm_faces_dataset.py` Full pipeline including artifact removal, epochs averaging, forward model computation and source reconstruction using dSPM on the contrast: "faces - scrambled".

EEGBCI motor imagery
====================
:func:`mne.datasets.eegbci.load_data()`

The EEGBCI dataset is documented in [2]_. The data set is available at PhysioNet [3]_.
The dataset contains 64-channel EEG recordings from 109 subjects and 14 runs on each subject in EDF+ format.
The recordings were made using the BCI2000 system. To load a subject, do::

    from mne.io import concatenate_raws, read_raw_edf
    from mne.datasets import eegbci
    raw_fnames = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_decoding_plot_decoding_csp_eeg.py`

Do not hesitate to contact MNE-Python developers on the
`MNE mailing list <http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis>`_
to discuss the possibility to add more publicly available datasets.

.. _auditory dataset tutorial: http://neuroimage.usc.edu/brainstorm/DatasetAuditory
.. _resting state dataset tutorial: http://neuroimage.usc.edu/brainstorm/DatasetResting
.. _median nerve dataset tutorial: http://neuroimage.usc.edu/brainstorm/DatasetMedianNerveCtf
.. _SPM faces dataset: http://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/

Somatosensory
=============
:func:`mne.datasets.somato.data_path()`

This dataset contains somatosensory data with event-related synchronizations
(ERS) and desynchronizations (ERD).

.. topic:: Examples

    * :ref:`sphx_glr_auto_tutorials_plot_sensors_time_frequency.py`

Multimodal
==========
:func:`mne.datasets.multimodal.data_path()`

This dataset contains a single subject recorded at Otaniemi (Aalto University)
with auditory, visual, and somatosensory stimuli.

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_io_plot_elekta_epochs.py`


Visual 92 object categories
===========================
:func:`mne.datasets.visual_92_categories.data_path()`.

This dataset is recorded using a 306-channel Neuromag vectorview system.

Experiment consisted in the visual presentation of 92 images of human, animal
and inanimate objects either natural or artificial [4]_. Given the high number
of conditions this dataset is well adapted to an approach based on
Representational Similarity Analysis (RSA).

.. topic:: Examples

    * :ref:`Representational Similarity Analysis (RSA) <sphx_glr_auto_examples_decoding_decoding_rsa.py>`: Partially replicates the results from Cichy et al. (2014).


mTRF Dataset
============
:func:`mne.datasets.mtrf.data_path()`.

This dataset contains 128 channel EEG as well as natural speech stimulus features,
which is also available `here <https://sourceforge.net/projects/aespa/files/>`_.

The experiment consisted of subjects listening to natural speech.
The dataset contains several feature representations of the speech stimulus,
suitable for using to fit continuous regression models of neural activity.
More details and a description of the package can be found in [5]_.

.. topic:: Examples

    * :ref:`Receptive Field Estimation and Prediction <sphx_glr_auto_examples_decoding_plot_receptive_field.py>`: Partially replicates the results from Crosse et al. (2016).

References
==========

.. [1] Aine CJ, Sanfratello L, Ranken D, Best E, MacArthur JA, Wallace T, Gilliam K, Donahue CH, Montano R, Bryant JE, Scott A, Stephen JM (2012) MEG-SIM: A Web Portal for Testing MEG Analysis Methods using Realistic Simulated and Empirical Data. Neuroinform 10:141-158

.. [2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE TBME 51(6):1034-1043

.. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220

.. [4] Cichy, R. M., Pantazis, D., & Oliva, A. "Resolving human object recognition in space and time." Nature neuroscience (2014): 17(3), 455-462

.. [5] Crosse, M. J., Di Liberto, G. M., Bednar, A., & Lalor, E. C. The Multivariate Temporal Response Function (mTRF) Toolbox: A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli. Frontiers in Human Neuroscience (2016): 10.
