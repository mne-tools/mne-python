.. _maxwell:

Maxwell filtering
#################

.. contents:: Contents
   :local:
   :depth: 2

Maxwell filtering in mne-python can be used to suppress sources of external
intereference and compensate for subject head movements. 

About Maxwell filtering
-----------------------
The principles behind Maxwell filtering are covered in relevant references
[1]_ [2]_ as well as the Elekta MaxFilter manual; see those materials
for basic principles and algorithm descriptions.

In mne-python Maxwell filtering of raw data can be done using the
:func:`mne.preprocessing.maxwell_filter` function. 

.. warning:: Automatic bad channel detection is not currently implemented.
             It is critical to mark bad channels before running Maxwell
             filtering, so data should be inspected and marked accordingly
             prior to running this algorithm.

Our Maxwell filtering algorithm currently provides multiple features,
including:

    * Bad channel reconstruction
    * Cross-talk cancellation
    * Fine calibration correction
    * tSSS
    * Coordinate frame translation
    * Regularization of internal components using information theory
    * Raw movement compensation
      (using head positions estimated by MaxFilter)
    * cHPI subtraction (see :func:`mne.chpi.filter_chpi`)
    * Handling of 3D (in addition to 1D) fine calibration files
    * Epoch-based movement compensation as described in [1]_ through
      :func:`mne.epochs.average_movements`
    * **Experimental** processing of data from (un-compensated)
      non-Elekta systems

Movement compensation
---------------------
When subject head movements are recorded continuously using continuous HPI
(cHPI) and subjects are expected to move during the recording (e.g., when
recording data in children), movement compensation can be performed to
correct for head movements. Movement compensation can be performed two ways:

1. Raw movement compensation: :func:`mne.preprocessing.maxwell_filter` using
   the ``pos`` argument.

2. Evoked movement compensation: :func:`mne.epochs.average_movements`.

Each of these requires time-varying estimates of head positions, which can
currently be obtained from MaxFilter using the ``-headpos`` and ``-hp``
arguments (see the MaxFilter manual for details). The resulting
MaxFilter-style head position information can be read using
:func:`mne.chpi.read_head_pos` and passed to mne-python's movement
compensation algorithms.

References
----------
.. [1] Taulu S. and Kajola M. "Presentation of electromagnetic
       multichannel data: The signal space separation method,"
       Journal of Applied Physics, vol. 97, pp. 124905 1-10, 2005.

       http://lib.tkk.fi/Diss/2008/isbn9789512295654/article2.pdf

.. [2] Taulu S. and Simola J. "Spatiotemporal signal space separation
       method for rejecting nearby interference in MEG measurements,"
       Physics in Medicine and Biology, vol. 51, pp. 1759-1768, 2006.

       http://lib.tkk.fi/Diss/2008/isbn9789512295654/article3.pdf
