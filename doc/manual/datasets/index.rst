.. _datasets:

Datasets
########

.. contents:: Contents
   :local:
   :depth: 2

All the dataset fetchers are available in :mod:`mne.datasets`. To download any of the datasets,
use the ``data_path`` (fetches full dataset) or the ``load_data`` (fetches dataset partially) functions.

.. include:: ./sample.rst
.. include:: ./brainstorm.rst
.. include:: ./megsim.rst
.. include:: ./spm_faces.rst
.. include:: ./eegbci_motor_imagery.rst
.. include:: ./somatosensory.rst
.. include:: ./multimodal.rst
.. include:: ./high_frequency_sef.rst
.. include:: ./visual_92_object_categories.rst
.. include:: ./mtrf_dataset.rst
.. include:: ./misc.rst
.. include:: ./kiloword_dataset.rst
.. include:: ./4d_neuroimaging_bti_dataset.rst
.. include:: ./opm.rst

Do not hesitate to contact MNE-Python developers on the
`MNE mailing list <http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis>`_
to discuss the possibility to add more publicly available datasets.

.. _auditory dataset tutorial: http://neuroimage.usc.edu/brainstorm/DatasetAuditory
.. _resting state dataset tutorial: http://neuroimage.usc.edu/brainstorm/DatasetResting
.. _median nerve dataset tutorial: http://neuroimage.usc.edu/brainstorm/DatasetMedianNerveCtf
.. _SPM faces dataset: http://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/

References
==========

.. [1] Aine CJ, Sanfratello L, Ranken D, Best E, MacArthur JA, Wallace T, Gilliam K, Donahue CH, Montano R, Bryant JE, Scott A, Stephen JM (2012) MEG-SIM: A Web Portal for Testing MEG Analysis Methods using Realistic Simulated and Empirical Data. Neuroinform 10:141-158

.. [2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE TBME 51(6):1034-1043

.. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220

.. [4] Cichy, R. M., Pantazis, D., & Oliva, A. Resolving human object recognition in space and time. Nature Neuroscience (2014): 17(3), 455-462

.. [5] Crosse, M. J., Di Liberto, G. M., Bednar, A., & Lalor, E. C. The Multivariate Temporal Response Function (mTRF) Toolbox: A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli. Frontiers in Human Neuroscience (2016): 10.

.. [6] Dufau, S., Grainger, J., Midgley, KJ., Holcomb, PJ. A thousand words are worth a picture: Snapshots of printed-word processing in an event-related potential megastudy. Psychological science, 2015
