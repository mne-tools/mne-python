

.. _setup_martinos:

============================
Setup at the Martinos Center
============================

.. contents:: Contents
   :local:
   :depth: 2


This Appendix contains information specific to the Martinos
Center setup.

.. _user_environment_martinos:

User environment
################

In the Martinos Center computer network, the 2.7 version
of MNE is located at /usr/pubsw/packages/mne/stable. To use this
version, follow :ref:`user_environment` substituting /usr/pubsw/packages/mne/stable
for <*MNE*> and /usr/pubsw/packages/matlab/current
for <*Matlab*> . For most users,
the default shell is tcsh.

.. note:: A new version of MNE is build every night from    the latest sources. This version is located at /usr/pubsw/packages/mne/nightly.

.. _BABGFDJG:

Using Neuromag software
#######################

Software overview
=================

The complete set of Neuromag software is available on the
LINUX workstations. The programs can be accessed from the command
line, see :ref:`BABFIEHC`. The corresponding manuals, located
at ``$NEUROMAG_ROOT/manuals`` are listed in :ref:`BABCJJGF`.

.. _BABFIEHC:

.. table:: Principal Neuromag software modules.

    ===========  =================================
    Module       Description
    ===========  =================================
    xfit         Source modelling
    xplotter     Data plotting
    graph        General purpose data processor
    mrilab       MEG-MRI integration
    seglab       MRI segmentation
    cliplab      Graphics clipboard
    ===========  =================================

.. _BABCJJGF:

.. table:: List of Neuromag software manuals.

    ===========  =========================================
    Module       pdf
    ===========  =========================================
    xfit         XFit.pdf
    xplotter     Xplotter.pdf
    graph        GraphUsersGuide.pdf GraphReference.pdf
    mrilab       Mrilab.pdf
    seglab       Seglab.pdf
    cliplab      Cliplab.pdf
    ===========  =========================================

To access the Neuromag software on the LINUX workstations
in the Martinos Center, say (in tcsh or csh)

``source /space/orsay/8/megdev/Neuromag-LINUX/neuromag_setup_csh``

or in POSIX shell

``. /space/orsay/8/megdev/Neuromag-LINUX/neuromag_setup_sh``

Using MRIlab for coordinate system alignment
============================================

The MEG-MRI coordinate system alignment can be also accomplished with
the Neuromag tool MRIlab, part of the standard software on Neuromag
MEG systems.

In MRIlab, the following steps are necessary for the coordinate
system alignment:

- Load the MRI description file ``COR.fif`` from ``subjects/sample/mri/T1-neuromag/sets`` through File/Open .

- Open the landmark setting dialog from Windows/Landmarks .

- Click on one of the coordinate setting fields on the Nasion line.
  Click Goto . Select the crosshair
  tool and move the crosshair to the nasion. Click Get .

- Proceed similarly for the left and right auricular points.
  Your instructor will help you with the selection of the correct
  points.

- Click OK to set the alignment

- Load the digitization data from the file ``sample_audvis_raw.fif`` or ``sample_audvis-ave.fif`` (the
  on-line evoked-response average file) in ``MEG/sample`` through File/Import/Isotrak data . Click Make points to
  show all the digitization data on the MRI slices.

- Check that the alignment is correct by looking at the locations
  of the digitized points are reasonable. Adjust the landmark locations
  using the Landmarks dialog, if
  necessary.

- Save the aligned file to the file suggested in the dialog
  coming up from File/Save .
