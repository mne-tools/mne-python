

.. _setup_martinos:

============================
Setup at the Martinos Center
============================

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

Mature software
###############

This Section contains documentation for software components,
which are still available in the MNE software but have been replaced
by new programs.

.. _BABDABHI:

mne_compute_mne
===============

This chapter contains information about the options accepted
by the program mne_compute_mne ,
which is gradually becoming obsolete. All of its functions will
be eventually included to mne_make_movie ,
see :ref:`CBBECEDE`. At this time, mne_compute_mne is
still needed to produce time-collapsed w files unless you are willing
to write a Matlab script of your own for this purpose.

mne_compute_mne accepts
the following command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---inv <*name*>**

    Load the inverse operator decomposition from here.

**\---meas <*name*>**

    Load the MEG or EEG data from this file.

**\---set <*number*>**

    The data set (condition) number to load. The list of data sets can
    be seen, *e.g.*, in mne_analyze , mne_browse_raw ,
    and xplotter .

**\---bmin <*time/ms*>**

    Specifies the starting time of the baseline. In order to activate
    baseline correction, both ``--bmin`` and ``--bmax`` options
    must be present.

**\---bmax <*time/ms*>**

    Specifies the finishing time of the baseline.

**\---nave <*value*>**

    Specifies the number of averaged epochs in the input data. If the input
    data file is one produced by mne_process_raw or mne_browse_raw ,
    the number of averages is correct in the file. However, if subtractions
    or some more complicated combinations of simple averages are produced, *e.g.*,
    by using the xplotter software, the
    number of averages should be manually adjusted. This is accomplished
    either by employing this flag or by adjusting the number of averages
    in the data file with help of mne_change_nave .

**\---snr <*value*>**

    An estimate for the amplitude SNR. The regularization parameter will
    be set as :math:`\lambda = ^1/_{\text{SNR}}`. If the SNR option is
    absent, the regularization parameter will be estimated from the
    data. The regularization parameter will be then time dependent.

**\---snronly**

    Only estimate SNR and output the result into a file called SNR. Each
    line of the file contains three values: the time point in ms, the estimated
    SNR + 1, and the regularization parameter estimated from the data
    at this time point.

**\---abs**

    Calculate the absolute value of the current and the dSPM for fixed-orientation
    data.

**\---spm**

    Calculate the dSPM instead of the expected current value.

**\---chi2**

    Calculate an approximate :math:`\chi_2^3` statistic
    instead of the *F* statistic. This is simply
    accomplished by multiplying the *F* statistic
    by three.

**\---sqrtF**

    Take the square root of the :math:`\chi_2^3` or *F* statistic
    before outputting the stc file.

**\---collapse**

    Make all frames in the stc file (or the wfile) identical. The value
    at each source location is the maximum value of the output quantity
    at this location over the analysis period. This option is convenient
    for determining the correct thresholds for the rendering of the
    final brain-activity movies.

**\---collapse1**

    Make all frames in the stc file (or the wfile) identical. The value
    at each source location is the :math:`L_1` norm
    of the output quantity at this location over the analysis period.

**\---collapse2**

    Make all frames in the stc file (or the wfile) identical. The value
    at each source location is the :math:`L_2` norm
    of the output quantity at this location over the analysis period.

**\---SIcurrents**

    Output true current values in SI units (Am). By default, the currents are
    scaled so that the maximum current value is set to 50 (Am).

**\---out <*name*>**

    Specifies the output file name. This is the 'stem' of
    the output file name. The actual name is derived by removing anything up
    to and including the last period from the end of <*name*> .
    According to the hemisphere, ``-lh`` or ``-rh`` is
    then appended. Finally, ``.stc`` or ``.w`` is added,
    depending on the output file type.

**\---wfiles**

    Use binary w-files in the output whenever possible. The noise-normalization
    factors can be always output in this format.  The current estimates
    and dSPMs can be output as wfiles if one of the collapse options
    is selected.

**\---pred <*name*>**

    Save the predicted data into this file. This is a fif file containing
    the predicted data waveforms, see :ref:`CHDCACDC`.

**\---outputnorm <*name*>**

    Output noise-normalization factors to this file.

**\---invnorm**

    Output inverse noise-normalization factors to the file defined by
    the ``--outputnorm`` option.

**\---dip <*name*>**

    Specifies a dipole distribution snapshot file. This is a file containing the
    current distribution at a time specified with the ``--diptime`` option.
    The file format is the ASCII dip file format produced by the Neuromag
    source modelling software (xfit). Therefore, the file can be loaded
    to the Neuromag MRIlab MRI viewer to display the actual current
    distribution. This option is only effective if the ``--spm`` option
    is absent.

**\---diptime <*time/ms*>**

    Time for the dipole snapshot, see ``--dip`` option above.

**\---label <*name*>**

    Label to process. The label files are produced by tksurfer and specify
    regions of interests (ROIs). A label file name should end with ``-lh.label`` for
    left-hemisphere ROIs and with ``-rh.label`` for right-hemisphere
    ones. The corresponding output files are tagged with ``-lh-`` <*data type* ``.amp`` and ``-rh-`` <*data type* ``.amp`` , respectively. <*data type*> equals ``MNE`` for expected current
    data and ``spm`` for dSPM data. Each line of the output
    file contains the waveform of the output quantity at one of the
    source locations falling inside the ROI.

.. note:: The ``--tmin`` and ``--tmax`` options    which existed in previous versions of mne_compute_mne have    been removed. mne_compute_mne can now    process only the entire averaged epoch.
