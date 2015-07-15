

.. _ch_convert:

===========================
Data reading and conversion
===========================

Overview
########

Here we describe the data reading and conversion utilities included
with the MNE software.

.. _BEHIAADG:

Importing data from other MEG/EEG systems
#########################################

This part describes the utilities to convert data from
other MEG/EEG systems into the fif format.


Importing MEG data
==================

This section describes reading and converting of various MEG data formats.

Importing 4-D Neuroimaging / BTI data
-------------------------------------

The python toolbox includes a function to read and convert 4D / BTI data.


Background
^^^^^^^^^^

The newest version of 4-D Magnes software included the possibility
to export data in fif. Please consult the documentation of the Magnes
system for details of this export utility. However, the exported
fif file does not include information about the compensation channels
and the weights to be applied to realize software gradient compensation.
To augment the Magnes fif files with the necessary information,
the MNE software includes the utilities mne_insert_4D_comp , mne_create_comp_data ,
and mne_add_to_meas_info.


.. note:: Including the compensation channel data is recommended but not mandatory.
If the data are saved in the Magnes system are already compensated,
there will be a small error in the forward calculations whose significance
has not been evaluated carefully at this time.

.. _BEHDEBCH:

Importing CTF data
------------------

The C command line tools include a utility mne_ctf2fiff ,
based on the BrainStorm Matlab code by Richard Leahy, John Mosher,
and Sylvain Baillet, to convert data in CTF ds directory to fif
format.

The command-line options of mne_ctf2fiff are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---verbose**

    Produce a verbose listing of the conversion process to stdout.

**\---ds <*directory*>**

    Read the data from this directory

**\---omit <*filename*>**

    Read the names of channels to be omitted from this text file. Enter one
    channel name per line. The names should match exactly with those
    listed in the CTF data structures. By default, all channels are included.

**\---fif <*filename*>**

    The name of the output file. If the length of the raw data exceeds
    the 2-GByte fif file limit, several output files will be produced.
    These additional 'extension' files will be tagged
    with ``_001.fif`` , ``_002.fif`` , etc.

**\---evoked**

    Produce and evoked-response fif file instead of a raw data file.
    Each trial in the CTF data file is included as a separate category
    (condition). The maximum number of samples in each trial is limited
    to 25000.

**\---infoonly**

    Write only the measurement info to the output file, do not include data.

During conversion, the following files are consulted from
the ds directory:

** <*name*> .res4**

    This file contains most of the header information pertaining the acquisition.

** <*name*> .hc**

    This file contains the HPI coil locations in sensor and head coordinates.

** <*name*> .meg4**

    This file contains the actual MEG data. If the data are split across several
    files due to the 2-GByte file size restriction, the 'extension' files
    are called <*name*> ``.`` <*number*> ``_meg4`` .

** <*name*> .eeg**

    This is an optional input file containing the EEG electrode locations. More
    details are given below.

If the <*name*> ``.eeg`` file,
produced from the Polhemus data file with CTF software, is present,
it is assumed to contain lines with the format:

 <*number*> <*name*> <*x/cm*> <*y/cm*> <*z/cm*>

The field <*number*> is
a sequential number to be assigned to the converted data point in
the fif file. <*name*> is either
a name of an EEG channel, one of ``left`` , ``right`` ,
or ``nasion`` to indicate a fiducial landmark, or any word
which is not a name of any channel in the data. If <*name*> is
a name of an EEG channel available in the data, the location is
included in the Polhemus data as an EEG electrode locations and
inserted as the location of the EEG electrode. If the name is one
of the fiducial landmark names, the point is included in the Polhemus
data as a fiducial landmark. Otherwise, the point is included as
an additional head surface points.

The standard ``eeg`` file produced by CTF software
does not contain the fiducial locations. If desired, they can be
manually copied from the ``pos`` file which was the source
of the ``eeg`` file.

.. note:: In newer CTF data the EEG position information    maybe present in the ``res4`` file. If the ``eeg`` file    is present, the positions given there take precedence over the information    in the ``res4`` file.

.. note:: mne_ctf2fiff converts    both epoch mode and continuous raw data file into raw data fif files.    It is not advisable to use epoch mode files with time gaps between    the epochs because the data will be discontinuous in the resulting    fif file with jumps at the junctions between epochs. These discontinuities    produce artefacts if the raw data is filtered in mne_browse_raw , mne_process_raw ,    or graph .

.. note:: The conversion process includes a transformation    from the CTF head coordinate system convention to that used in the    Neuromag systems.

.. _BEHBABFA:

Importing CTF Polhemus data
===========================

The CTF MEG systems store the Polhemus digitization data
in text files. The utility mne_ctf_dig2fiff was
created to convert these data files into the fif and hpts formats.

The input data to mne_ctf_dig2fiff is
a text file, which contains the coordinates of the digitization
points in centimeters. The first line should contain a single number
which is the number of points listed in the file. Each of the following
lines contains a sequential number of the point, followed by the
three coordinates. mne_ctf_dig2fiff ignores
any text following the :math:`z` coordinate
on each line. If the ``--numfids`` option is specified,
the first three points indicate the three fiducial locations (1
= nasion, 2 = left auricular point, 3 = right auricular point).
Otherwise, the input file must end with three lines beginning with ``left`` , ``right`` ,
or ``nasion`` to indicate the locations of the fiducial
landmarks, respectively.

.. note:: The sequential numbers should be unique within    a file. I particular, the numbers 1, 2, and 3 must not be appear    more than once if the ``--numfids`` options is used.

The command-line options for mne_ctf_dig2fiff are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---dig <*name*>**

    Specifies the input data file in CTF output format.

**\---numfids**

    Fiducial locations are numbered instead of labeled, see above.

**\---hpts <*name*>**

    Specifies the output hpts file. The format of this text file is
    described in :ref:`CJADJEBH`.

**\---fif <*name*>**

    Specifies the output fif file.

.. _BEHDDFBI:

Applying software gradient compensation
=======================================

Since the software gradient compensation employed in CTF
systems is a reversible operation, it is possible to change the
compensation status of CTF data in the data files as desired. This
section contains information about the technical details of the
compensation procedure and a description of mne_compensate_data ,
which is a utility to change the software gradient compensation
state in evoked-response data files.

The fif files containing CTF data converted using the utility mne_ctf2fiff contain
several compensation matrices which are employed to suppress external disturbances
with help of the reference channel data. The reference sensors are
located further away from the brain than the helmet sensors and
are thus measuring mainly the external disturbances rather than magnetic
fields originating in the brain. Most often, a compensation matrix
corresponding to a scheme nicknamed *Third-order gradient
compensation* is employed.

Let us assume that the data contain :math:`n_1` MEG
sensor channels, :math:`n_2` reference sensor
channels, and :math:`n_3` other channels.
The data from all channels can be concatenated into a single vector

.. math::    x = [x_1^T x_2^T x_3^T]^T\ ,

where :math:`x_1`, :math:`x_2`,
and :math:`x_3` are the data vectors corresponding
to the MEG sensor channels, reference sensor channels, and other
channels, respectively. The data before and after compensation,
denoted here by :math:`x_{(0)}` and :math:`x_{(k)}`, respectively,
are related by

.. math::    x_{(k)} = M_{(k)} x_{(0)}\ ,

where the composite compensation matrix is

.. math::    M_{(k)} = \begin{bmatrix}
		I_{n_1} & C_{(k)} & 0 \\
		0 & I_{n_2} & 0 \\
		0 & 0 & I_{n_3}
		\end{bmatrix}\ .

In the above, :math:`C_{(k)}` is a :math:`n_1` by :math:`n_2` compensation
data matrix corresponding to compensation "grade" :math:`k`.
It is easy to see that

.. math::    M_{(k)}^{-1} = \begin{bmatrix}
		I_{n_1} & -C_{(k)} & 0 \\
		0 & I_{n_2} & 0 \\
		0 & 0 & I_{n_3}
		\end{bmatrix}\ .

To convert from compensation grade :math:`k` to :math:`p` one
can simply multiply the inverse of one compensate compensation matrix
by another and apply the product to the data:

.. math::    x_{(k)} = M_{(k)} M_{(p)}^{-1} x_{(p)}\ .

This operation is performed by mne_compensate_data ,
which has the following command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---in <*name*>**

    Specifies the input data file.

**\---out <*name*>**

    Specifies the output data file.

**\---grad <*number*>**

    Specifies the desired compensation grade in the output file. The value
    can be 1, 2, 3, or 101. The values starting from 101 will be used
    for 4D Magnes compensation matrices.

.. note:: Only average data is included in the output.    Evoked-response data files produced with mne_browse_raw or mne_process_raw may    include standard errors of mean, which can not be re-compensated    using the above method and are thus omitted.

.. note:: Raw data cannot be compensated using mne_compensate_data .    For this purpose, load the data to mne_browse_raw or mne_process_raw , specify    the desired compensation grade, and save a new raw data file.

.. _BEHGDDBH:

Importing Magnes compensation channel data
==========================================

At present, it is not possible to include reference channel
data to fif files containing 4D Magnes data directly using the conversion
utilities available for the Magnes systems. However, it is possible
to export the compensation channel signals in text format and merge
them with the MEG helmet channel data using mne_insert_4D_comp .
This utility has the following command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---in <*name*>**

    Specifies the input fif file containing the helmet sensor data.

**\---out <*name*>**

    Specifies the output fif file which will contain both the helmet
    sensor data and the compensation channel data.

**\---ref <*name*>**

    Specifies a text file containing the reference sensor data.

Each line of the reference sensor data file contains the
following information:

**epoch #**

    is
    always one,

**time/s**

    time point of this sample,

**data/T**

    the reference channel data
    values.

The standard locations of the MEG (helmet) and compensation
sensors in a Magnes WH3600 system are listed in ``$MNE_ROOT/share/mne/Magnes_WH3600.pos`` . mne_insert_4D_comp matches
the helmet sensor positions in this file with those present in the
input data file and transforms the standard compensation channel
locations accordingly to be included in the output. Since a standard
position file is only provided for Magnes WH600, mne_insert_4D_comp only
works for that type of a system.

The fif files exported from the Magnes systems may contain
slightly smaller number of samples than originally acquired because
the total number of samples may not be evenly divisible with a reasonable
number of samples which will be used as the fif raw data file buffer
size. Therefore, the reference channel data may contain more samples
than the fif file. The superfluous samples will be omitted from
the end.

.. _BEHBIIFF:

Creating software gradient compensation data
============================================

The utility mne_create_comp_data was
written to create software gradient compensation weight data for
4D Magnes fif files. This utility takes a text file containing the
compensation data as input and writes the corresponding fif file
as output. This file can be merged into the fif file containing
4D Magnes data with the utility mne_add_to_meas_info .

The command line options of mne_create_comp_data are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---in <*name*>**

    Specifies the input text file containing the compensation data.

**\---kind <*value*>**

    The compensation type to be stored in the output file with the data. This
    value defaults to 101 for the Magnes compensation and does not need
    to be changed.

**\---out <*name*>**

    Specifies the output fif file containing the compensation channel weight
    matrix :math:`C_{(k)}`, see :ref:`BEHDDFBI`.

The format of the text-format compensation data file is:

 <*number of MEG helmet channels*> <*number of compensation channels included*>
 <*cname_1*> <*cname_2*> ...
 <*name_1*> <*weights*>
 <*name_2*> <*weights*> ...

In the above <*name_k*> denote
names of MEG helmet channels and <*cname_k*>
those of the compensation channels, respectively. If the channel
names contain spaces, they must be surrounded by quotes, for example, ``"MEG 0111"`` .

.. _BEHBJGGF:

Importing KIT MEG system data
=============================

The utility mne_kit2fiff was
created in collaboration with Alec Maranz and Asaf Bachrach to import
their MEG data acquired with the 160-channel KIT MEG system to MNE
software.

To import the data, the following input files are mandatory:

- The Polhemus data file (elp file)
  containing the locations of the fiducials and the head-position
  indicator (HPI) coils. These data are usually given in the CTF/4D
  head coordinate system. However, mne_kit2fiff does
  not rely on this assumption. This file can be exported directly from
  the KIT system.

- A file containing the locations of the HPI coils in the MEG
  device coordinate system. These data are used together with the elp file
  to establish the coordinate transformation between the head and
  device coordinate systems. This file can be produced easily by manually
  editing one of the files exported by the KIT system.

- A sensor data file (sns file)
  containing the locations and orientations of the sensors. This file
  can be exported directly from the KIT system.

.. note:: The output fif file will use the Neuromag head    coordinate system convention, see :ref:`BJEBIBAI`. A coordinate    transformation between the CTF/4D head coordinates and the Neuromag    head coordinates is included. This transformation can be read with    MNE Matlab Toolbox routines, see :ref:`ch_matlab`.

The following input files are optional:

- A head shape data file (hsp file)
  containing locations of additional points from the head surface.
  These points must be given in the same coordinate system as that
  used for the elp file and the
  fiducial locations must be within 1 mm from those in the elp file.

- A raw data file containing the raw data values, sample by
  sample, as text. If this file is not specified, the output fif file
  will only contain the measurement info block.

By default mne_kit2fiff includes
the first 157 channels, assumed to be the MEG channels, in the output
file. The compensation channel data are not converted by default
but can be added, together with other channels, with the ``--type`` .
The channels from 160 onwards are designated as miscellaneous input
channels (MISC 001, MISC 002, etc.). The channel names and types
of these channels can be afterwards changed with the mne_rename_channels utility,
see :ref:`CHDCFEAJ`. In addition, it is possible to synthesize
the digital trigger channel (STI 014) from available analog
trigger channel data, see the ``--stim`` option, below.
The synthesized trigger channel data value at sample :math:`k` will
be:

.. math::    s(k) = \sum_{p = 1}^n {t_p(k) 2^{p - 1}}\ ,

where :math:`t_p(k)` are the thresholded
from the input channel data d_p(k):

.. math::    t_p(k) = \Bigg\{ \begin{array}{l}
		 0 \text{  if  } d_p(k) \leq t\\
		 1 \text{  if  } d_p(k) > t
	     \end{array}\ .

The threshold value :math:`t` can
be adjusted with the ``--stimthresh`` option, see below.

mne_kit2fiff accepts
the following command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---elp <*filename*>**

    The name of the file containing the locations of the fiducials and
    the HPI coils. This option is mandatory.

**\---hsp <*filename*>**

    The name of the file containing the locations of the fiducials and additional
    points on the head surface. This file is optional.

**\---sns <*filename*>**

    The name of file containing the sensor locations and orientations. This
    option is mandatory.

**\---hpi <*filename*>**

    The name of a text file containing the locations of the HPI coils
    in the MEG device coordinate frame, given in millimeters. The order of
    the coils in this file does not have to be the same as that in the elp file.
    This option is mandatory.

**\---raw <*filename*>**

    Specifies the name of the raw data file. If this file is not specified, the
    output fif file will only contain the measurement info block.

**\---sfreq <*value/Hz*>**

    The sampling frequency of the data. If this option is not specified, the
    sampling frequency defaults to 1000 Hz.

**\---lowpass <*value/Hz*>**

    The lowpass filter corner frequency used in the data acquisition.
    If not specified, this value defaults to 200 Hz.

**\---highpass <*value/Hz*>**

    The highpass filter corner frequency used in the data acquisition.
    If not specified, this value defaults to 0 Hz (DC recording).

**\---out <*filename*>**

    Specifies the name of the output fif format data file. If this file
    is not specified, no output is produced but the elp , hpi ,
    and hsp files are processed normally.

**\---stim <*chs*>**

    Specifies a colon-separated list of numbers of channels to be used
    to synthesize a digital trigger channel. These numbers refer to
    the scanning order channels as listed in the sns file,
    starting from one. The digital trigger channel will be the last
    channel in the file. If this option is absent, the output file will
    not contain a trigger channel.

**\---stimthresh <*value*>**

    The threshold value used when synthesizing the digital trigger channel,
    see above. Defaults to 1.0.

**\---add <*chs*>**

    Specifies a colon-separated list of numbers of channels to include between
    the 157 default MEG channels and the digital trigger channel. These
    numbers refer to the scanning order channels as listed in the sns file,
    starting from one.

.. note:: The mne_kit2fiff utility    has not been extensively tested yet.

.. _BABHDBBD:

Importing EEG data saved in the EDF, EDF+, or BDF format
========================================================

Overview
--------

The mne_edf2fiff allows
conversion of EEG data from EDF, EDF+, and BDF formats to the fif
format. Documentation for these three input formats can be found
at:

**EDF:**

    http://www.edfplus.info/specs/edf.html

**EDF+:**

    http://www.edfplus.info/specs/edfplus.html

**BDF:**

    http://www.biosemi.com/faq/file_format.htm

EDF (European Data Format) and EDF+ are 16-bit formats while
BDF is a 24-bit variant of this format used by the EEG systems manufactured
by a company called BioSemi.

None of these formats support electrode location information
and  head shape digitization information. Therefore, this information
has to be provided separately. Presently hpts and elp file formats
are supported to include digitization data. For information on these
formats, see :ref:`CJADJEBH` and http://www.sourcesignal.com/formats_probe.html.
Note that it is mandatory to have the three fiducial locations (nasion
and the two auricular points) included in the digitization data.
Using the locations of the fiducial points the digitization data
are converted to the MEG head coordinate system employed in the
MNE software, see :ref:`BJEBIBAI`. In the comparison of the
channel names only the initial segment up to the first '-' (dash)
in the EDF/EDF+/BDF channel name is significant.

The EDF+ files may contain an annotation channel which can
be used to store trigger information. The Time-stamped Annotation
Lists (TALs) on the annotation  data can be converted to a trigger
channel (STI 014) using an annotation map file which associates
an annotation label with a number on the trigger channel. The TALs
can be listed with the ``--listtal`` option,
see below.

.. warning:: The data samples in a BDF file    are represented in a 3-byte (24-bit) format. Since 3-byte raw data    buffers are not presently supported in the fif format    these data will be changed to 4-byte integers in the conversion.    Since the maximum size of a fif file is 2 GBytes, the maximum size of    a BDF file to be converted is approximately 1.5 GBytes

.. warning:: The EDF/EDF+/BDF formats support channel    dependent sampling rates. This feature is not supported by mne_edf2fiff .    However, the annotation channel in the EDF+ format can have a different    sampling rate. The annotation channel data is not included in the    fif files output.

Using mne_edf2fiff
------------------

The command-line options of mne_edf2fiff are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---edf <*filename*>**

    Specifies the name of the raw data file to process.

**\---tal <*filename*>**

    List the time-stamped annotation list (TAL) data from an EDF+ file here.
    This output is useful to assist in creating the annotation map file,
    see the ``--annotmap`` option, below.
    This output file is an event file compatible with mne_browse_raw and mne_process_raw ,
    see :ref:`ch_browse`. In addition, in the mapping between TAL
    labels and trigger numbers provided by the ``--annotmap`` option is
    employed to assign trigger numbers in the event file produced. In
    the absence of the ``--annotmap`` option default trigger number 1024
    is used.

**\---annotmap <*filename*>**

    Specify a file which maps the labels of the TALs to numbers on a trigger
    channel (STI 014) which will be added to the output file if this
    option is present. This annotation map file
    may contain comment lines starting with the '%' or '#' characters.
    The data lines contain a label-number pair, separated by a colon.
    For example, a line 'Trigger-1:9' means that each
    annotation labeled with the text 'Trigger-1' will
    be translated to the number 9 on the trigger channel.

**\---elp <*filename*>**

    Specifies the name of the an electrode location file. This file
    is in the "probe" file format used by the *Source
    Signal Imaging, Inc.* software. For description of the
    format, see http://www.sourcesignal.com/formats_probe.html. Note
    that some other software packages may produce electrode-position
    files with the elp ending not
    conforming to the above specification. As discussed above, the fiducial
    marker locations, optional in the "probe" file
    format specification are mandatory for mne_edf2fiff .
    When this option is encountered on the command line any previously
    specified hpts file will be ignored.

**\---hpts <*filename*>**

    Specifies the name of an electrode position file in  the hpts format discussed
    in :ref:`CJADJEBH`. The mandatory entries are the fiducial marker
    locations and the EEG electrode locations. It is recommended that
    electrode (channel) names instead of numbers are used to label the
    EEG electrode locations. When this option is encountered on the
    command line any previously specified elp file
    will be ignored.

**\---meters**

    Assumes that the digitization data in an hpts file
    is given in meters instead of millimeters.

**\---fif <*filename*>**

    Specifies the name of the fif file to be output.

Post-conversion tasks
---------------------

This section outlines additional steps to be taken to use
the EDF/EDF+/BDF file is converted to the fif format in MNE:

- Some of the channels may not have a
  digitized electrode location associated with them. If these channels
  are used for EOG or EMG measurements, their channel types should
  be changed to the correct ones using the mne_rename_channels utility,
  see :ref:`CHDCFEAJ`. EEG channels which do not have a location
  associated with them should be assigned to be MISC channels.

- After the channel types are correctly defined, a topographical
  layout file can be created for mne_browse_raw and mne_analyze using
  the mne_make_eeg_layout utility,
  see :ref:`CHDDGDJA`.

- The trigger channel name in BDF files is "Status".
  This must be specified with the ``--digtrig`` option or with help of
  the MNE_TRIGGER_CH_NAME environment variable when mne_browse_raw or mne_process_raw is
  invoked, see :ref:`BABBGJEA`.

- Only the two least significant bytes on the "Status" channel
  of BDF files are significant as trigger information the ``--digtrigmask``
  0xff option MNE_TRIGGER_CH_MASK environment variable should be used
  to specify this to mne_browse_raw and mne_process_raw ,
  see :ref:`BABBGJEA`.

.. _BEHDGAIJ:

Importing EEG data saved in the Tufts University format
=======================================================

The utility mne_tufts2fiff was
created in collaboration with Phillip Holcomb and Annette Schmid
from Tufts University to import their EEG data to the MNE software.

The Tufts EEG data is included in three files:

- The raw data file containing the acquired
  EEG data. The name of this file ends with the suffix ``.raw`` .

- The calibration raw data file. This file contains known calibration
  signals and is required to bring the data to physical units. The
  name of this file ends with the suffix ``c.raw`` .

- The electrode location information file. The name of this
  file ends with the suffix ``.elp`` .

The utility mne_tufts2fiff has
the following command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---raw <*filename*>**

    Specifies the name of the raw data file to process.

**\---cal <*filename*>**

    The name of the calibration data file. If calibration data are missing, the
    calibration coefficients will be set to unity.

**\---elp <*filename*>**

    The name of the electrode location file. If this file is missing,
    the electrode locations will be unspecified. This file is in the "probe" file
    format used by the *Source Signal Imaging, Inc.* software.
    For description of the format, see http://www.sourcesignal.com/formats_probe.html.
    The fiducial marker locations, optional in the "probe" file
    format specification are mandatory for mne_tufts2fiff . Note
    that some other software packages may produce electrode-position
    files with the elp ending not
    conforming to the above specification.

.. note::

    The conversion process includes a transformation from the Tufts head coordinate system convention to that used in    the Neuromag systems.

.. note::

    The fiducial landmark locations, optional in the probe file format, must be present for mne_tufts2fiff .

.. _BEHCCCDC:

Importing BrainVision EEG data
==============================

The utility mne_brain_vision2fiff was
created to import BrainVision EEG data. This utility also helps
to import the eXimia (Nexstim) TMS-compatible EEG system data to
the MNE software. The utility uses an optional fif file containing
the head digitization data to allow source modeling. The MNE Matlab
toolbox contains the function fiff_write_dig_file to
write a digitization file based on digitization data available in
another format, see :ref:`ch_matlab`.

.. note::

    mne_brain_vision2fiff reads events from the ``vmrk`` file referenced in the
    ``vhdr`` file, but it only includes events whose "Type" is ``Stimulus`` and
    whose "description" is given by ``S<number>``. All other events are ignored.


The command-line options of mne_brain_vision2fiff are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---header <*name*>**

    The name of the BrainVision header file. The extension of this file
    is ``vhdr`` . The header file typically refers to a marker
    file (``vmrk`` ) which is automatically processed and a
    digital trigger channel (STI 014) is formed from the marker information.
    The ``vmrk`` file is ignored if the ``--eximia`` option
    is present.

**\---dig <*name*>**

    The name of the fif file containing the digitization data.

**\---orignames**

    Use the original EEG channel labels. If this option is absent the EEG
    channels will be automatically renamed to EEG 001, EEG 002, *etc.*

**\---eximia**

    Interpret this as an eXimia data file. The first three channels
    will be thresholded and interpreted as trigger channels. The composite
    digital trigger channel will be composed in the same way as in the mne_kit2fiff utility,
    see :ref:`BEHBJGGF`, above. In addition, the fourth channel
    will be assigned as an EOG channel. This option is normally used
    by the mne_eximia2fiff script,
    see :ref:`BEHGCEHH`.

**\---split <*size/MB*>**

    Split the output data into several files which are no more than <*size*> MB.
    By default, the output is split into files which are just below
    2 GB so that the fif file maximum size is not exceeded.

**\---out <*filename*>**

    Specifies the name of the output fif format data file. If <*filename*> ends
    with ``.fif`` or ``_raw.fif`` , these endings are
    deleted. After these modifications, ``_raw.fif`` is inserted
    after the remaining part of the file name. If the file is split
    into multiple parts, the additional parts will be called
    <*name*> ``-`` <*number*> ``_raw.fif`` .

.. _BEHGCEHH:

Converting eXimia EEG data
==========================

EEG data from the Nexstim eXimia system can be converted
to the fif format with help of the mne_eximia2fiff script.
It creates a BrainVision ``vhdr`` file and calls mne_brain_vision2fiff.
Usage:

``mne_eximia2fiff`` [``--dig`` dfile ] [``--orignames`` ] file1 file2 ...

where file1 file2 ...
are eXimia ``nxe`` files and the ``--orignames`` option
is passed on to mne_brain_vision2fiff .
If you want to convert all data files in a directory, say

``mne_eximia2fiff *.nxe``

The optional file specified with the ``--dig`` option is assumed
to contain digitizer data from the recording in the Nexstim format.
The resulting fif data file will contain these data converted to
the fif format as well as the coordinate transformation between
the eXimia digitizer and MNE head coordinate systems.

.. note:: This script converts raw data files only.

.. _BABCJEAD:

Converting digitization data
############################

The mne_convert_dig_data utility
converts Polhemus digitization data between different file formats.
The input formats are:

**fif**

    The
    standard format used in MNE. The digitization data are typically
    present in the measurement files.

**hpts**

    A text format which is a translation
    of the fif format data, see :ref:`CJADJEBH` below.

**elp**

    A text format produced by the *Source
    Signal Imaging, Inc.* software. For description of this "probe" format,
    see http://www.sourcesignal.com/formats_probe.html.

The data can be output in fif and hpts formats.
Only the last command-line option specifying an input file will
be honored. Zero or more output file options can be present on the
command line.

.. note:: The elp and hpts input    files may contain textual EEG electrode labels. They will not be    copied to the fif format output.

The command-line options of mne_convert_dig_data are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---fif <*name*>**

    Specifies the name of an input fif file.

**\---hpts <*name*>**

    Specifies the name of an input hpts file.

**\---elp <*name*>**

    Specifies the name of an input elp file.

**\---fifout <*name*>**

    Specifies the name of an output fif file.

**\---hptsout <*name*>**

    Specifies the name of an output hpts file.

**\---headcoord**

    The fif and hpts input
    files are assumed to contain data in the  MNE head coordinate system,
    see :ref:`BJEBIBAI`. With this option present, the data are
    transformed to the MNE head coordinate system with help of the fiducial
    locations in the data. Use this option if this is not the case or
    if you are unsure about the definition of the coordinate system
    of the fif and hpts input
    data. This option is implied with elp input
    files. If this option is present, the fif format output file will contain
    the transformation between the original digitizer data coordinates
    the MNE head coordinate system.

.. _CJADJEBH:

The hpts format
===============

The hpts format digitzer
data file may contain comment lines starting with the pound sign
(#) and data lines of the form:

 <*category*> <*identifier*> <*x/mm*> <*y/mm*> <*z/mm*>

where

** <*category*>**

    defines the type of points. Allowed categories are: hpi , cardinal (fiducial ),eeg ,
    and extra corresponding to head-position
    indicator coil locations, cardinal landmarks, EEG electrode locations,
    and additional head surface points, respectively. Note that tkmedit does not
    recognize the fiducial as an
    alias for cardinal .

** <*identifier*>**

    identifies the point. The identifiers are usually sequential numbers. For
    cardinal landmarks, 1 = left auricular point, 2 = nasion, and 3
    = right auricular point. For EEG electrodes, identifier = 0 signifies
    the reference electrode. Some programs (not tkmedit )
    accept electrode labels as identifiers in the eeg category.

** <*x/mm*> , <*y/mm*> , <*z/mm*>**

    Location of the point, usually in the MEG head coordinate system, see :ref:`BJEBIBAI`.
    Some programs have options to accept coordinates in meters instead
    of millimeters. With ``--meters`` option, mne_transform_points lists
    the coordinates in meters.

.. _BEHDEJEC:

Converting volumetric data into an MRI overlay
##############################################

With help of the mne_volume_source_space utility
(:ref:`BJEFEHJI`) it is possible to create a source space which
is defined within a volume rather than a surface. If the ``--mri`` option
was used in mne_volume_source_space , the
source space file contains an interpolator matrix which performs
a trilinear interpolation into the voxel space of the MRI volume
specified.

At present, the MNE software does not include facilities
to compute volumetric source estimates. However, it is possible
to calculate forward solutions in the volumetric grid and use the
MNE Matlab toolbox to read the forward solution. It is then possible
to compute, *e.g.*, volumetric beamformer solutions
in Matlab and output the results into w or stc files.
The purpose of the mne_volume_data2mri is
to produce MRI overlay data compatible with FreeSurfer MRI viewers
(in the mgh or mgz formats) from this type of w or stc files.

mne_volume_data2mri accepts
the following command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---src <*filename*>**

    The name of the volumetric source space file created with mne_volume_source_space .
    The source space must have been created with the ``--mri`` option,
    which adds the appropriate sparse trilinear interpolator matrix
    to the source space.

**\---w <*filename*>**

    The name of a w file to convert
    into an MRI overlay.

**\---stc <*filename*>**

    The name of the stc file to convert
    into an MRI overlay. If this file has many time frames, the output
    file may be huge. Note: If both ``-w`` and ``--stc`` are
    specified, ``-w`` takes precedence.

**\---scale <*number*>**

    Multiply the stc or w by
    this scaling constant before producing the overlay.

**\---out <*filename*>**

    Specifies the name of the output MRI overlay file. The name must end
    with either ``.mgh`` or ``.mgz`` identifying the
    uncompressed and compressed FreeSurfer MRI formats, respectively.

.. _BEHBHIDH:

Listing source space data
#########################

The utility mne_list_source_space outputs
the source space information into text files suitable for loading
into the Neuromag MRIlab software.

The command-line options are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---src <*name*>**

    The source space to be listed. This can be either the output from mne_make_source_space
    (`*src.fif`), output from the forward calculation (`*fwd.fif`), or
    the output from the inverse operator decomposition (`*inv.fif`).

**\---mri <*name*>**

    A file containing the transformation between the head and MRI coordinates
    is specified with this option. This file can be either a Neuromag
    MRI description file, the output from the forward calculation (`*fwd.fif`),
    or the output from the inverse operator decomposition (`*inv.fif`).
    If this file is included, the output will be in head coordinates.
    Otherwise the source space will be listed in MRI coordinates.

**\---dip <*name*>**

    Specifies the 'stem' for the Neuromag text format
    dipole files to be output. Two files will be produced: <*stem*> -lh.dip
    and <*stem*> -rh.dip. These correspond
    to the left and right hemisphere part of the source space, respectively.
    This source space data can be imported to MRIlab through the File/Import/Dipoles menu
    item.

**\---pnt <*name*>**

    Specifies the 'stem' for Neuromag text format
    point files to be output. Two files will be produced: <*stem*> -lh.pnt
    and <*stem*> -rh.pnt. These correspond
    to the left and right hemisphere part of the source space, respectively.
    This source space data can be imported to MRIlab through the File/Import/Strings menu
    item.

**\---exclude <*name*>**

    Exclude the source space points defined by the given FreeSurfer 'label' file
    from the output. The name of the file should end with ``-lh.label``
    if it refers to the left hemisphere and with ``-rh.label`` if
    it lists points in the right hemisphere, respectively.

**\---include <*name*>**

    Include only the source space points defined by the given FreeSurfer 'label' file
    to the output. The file naming convention is the same as described
    above under the ``--exclude`` option. Are 'include' labels are
    processed before the 'exclude' labels.

**\---all**

    Include all nodes in the output files instead of only those active
    in the source space. Note that the output files will be huge if
    this option is active.

.. _BEHBBEHJ:

Listing BEM mesh data
#####################

The utility mne_list_bem outputs
the BEM meshes in text format. The default output data contains
the *x*, *y*, and *z* coordinates
of the vertices, listed in millimeters, one vertex per line.

The command-line options are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---bem <*name*>**

    The BEM file to be listed. The file name normally ends with -bem.fif or -bem-sol.fif .

**\---out <*name*>**

    The output file name.

**\---id <*number*>**

    Identify the surface to be listed. The surfaces are numbered starting with
    the innermost surface. Thus, for a three-layer model the surface numbers
    are: 4 = scalp, 3 = outer skull, 1 = inner skull
    Default value is 4.

**\---gdipoli**

    List the surfaces in the format required by Thom Oostendorp's
    gdipoli program. This is also the default input format for mne_surf2bem .

**\---meters**

    List the surface coordinates in meters instead of millimeters.

**\---surf**

    Write the output in the binary FreeSurfer format.

**\---xfit**

    Write a file compatible with xfit. This is the same effect as using
    the options ``--gdipoli`` and ``--meters`` together.

.. _BEHDIAJG:

Converting surface data between different formats
#################################################

The utility mne_convert_surface converts
surface data files between different formats.

.. note:: The MNE Matlab toolbox functions enable    reading of FreeSurfer surface files directly. Therefore, the ``--mat``   option has been removed. The dfs file format conversion functionality    has been moved here from mne_convert_dfs .    Consequently, mne_convert_dfs has    been removed from MNE software.

.. _BABEABAA:

command-line options
====================

mne_convert_surface accepts
the following command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---fif <*name*>**

    Specifies a fif format input file. The first surface (source space)
    from this file will be read.

**\---tri <*name*>**

    Specifies a text format input file. The format of this file is described in :ref:`BEHDEFCD`.

**\---meters**

    The unit of measure for the vertex locations in a text input files
    is meters instead of the default millimeters. This option does not
    have any effect on the interpretation of the FreeSurfer surface
    files specified with the ``--surf`` option.

**\---swap**

    Swap the ordering or the triangle vertices. The standard convention in
    the MNE software is to have the vertices in text format files ordered
    so that the vector cross product of the vectors from vertex 1 to
    2 and 1 to 3 gives the direction of the outward surface normal. This
    is also called the counterclockwise ordering. If your text input file
    does not comply with this right-hand rule, use the ``--swap`` option.
    This option does not have any effect on the interpretation of the FreeSurfer surface
    files specified with the ``--surf`` option.

**\---surf <*name*>**

    Specifies a FreeSurfer format
    input file.

**\---dfs <*name*>**

    Specifies the name of a dfs file to be converted. The surfaces produced
    by BrainSuite are in the dfs format.

**\---mghmri <*name*>**

    Specifies a mgh/mgz format MRI data file which will be used to define
    the coordinate transformation to be applied to the data read from
    a dfs file to bring it to the FreeSurfer MRI
    coordinates, *i.e.*, the coordinate system of
    the MRI stack in the file. In addition, this option can be used
    to insert "volume geometry" information to the FreeSurfer
    surface file output (``--surfout`` option). If the input file already
    contains the volume geometry information, --replacegeom is needed
    to override the input volume geometry and to proceed to writing
    the data.

**\---replacegeom**

    Replaces existing volume geometry information. Used in conjunction
    with the ``--mghmri`` option described above.

**\---fifmri <*name*>**

    Specifies a fif format MRI destription file which will be used to define
    the coordinate transformation to be applied to the data read from
    a dfs file to bring it to the same coordinate system as the MRI stack
    in the file.

**\---trans <*name*>**

    Specifies the name of a text file which contains the coordinate
    transformation to be applied to the data read from the dfs file
    to bring it to the MRI coordinates, see below. This option is rarely
    needed.

**\---flip**

    By default, the dfs surface nodes are assumed to be in a right-anterior-superior
    (RAS) coordinate system with its origin at the left-posterior-inferior
    (LPI) corner of the MRI stack. Sometimes the dfs file has left and
    right flipped. This option reverses this flip, *i.e.*,
    assumes the surface coordinate system is left-anterior-superior
    (LAS) with its origin in the right-posterior-inferior (RPI) corner
    of the MRI stack.

**\---shift <*value/mm*>**

    Shift the surface vertices to the direction of the surface normals
    by this amount before saving the surface.

**\---surfout <*name*>**

    Specifies a FreeSurfer format output file.

**\---fifout <*name*>**

    Specifies a fif format output file.

**\---triout <*name*>**

    Specifies an ASCII output file that will contain the surface data
    in the triangle file format desribed in :ref:`BEHDEFCD`.

**\---pntout <*name*>**

    Specifies a ASCII output file which will contain the vertex numbers only.

**\---metersout**

    With this option the ASCII output will list the vertex coordinates
    in meters instead of millimeters.

**\---swapout**

    Defines the vertex ordering of ASCII triangle files to be output.
    For details, see ``--swap`` option, above.

**\---smfout <*name*>**

    Specifies a smf (Simple Model Format) output file. For details of this
    format, see http://people.sc.fsu.edu/~jburkardt/data/smf/smf.txt.

.. note:: Multiple output options can be specified to    produce outputs in several different formats with a single invocation    of mne_convert_surface .

The coordinate transformation file specified with the ``--trans`` should contain
a 4 x 4 coordinate transformation matrix:

.. math::    T = \begin{bmatrix}
		R_{11} & R_{12} & R_{13} & x_0 \\
		R_{13} & R_{13} & R_{13} & y_0 \\
		R_{13} & R_{13} & R_{13} & z_0 \\
		0 & 0 & 0 & 1
		\end{bmatrix}

defined so that if the augmented location vectors in the
dfs file and MRI coordinate systems are denoted by :math:`r_{dfs} = [x_{dfs} y_{dfs} z_{dfs} 1]^T` and :math:`r_{MRI} = [x_{MRI} y_{MRI} z_{MRI} 1]^T`,
respectively,

.. math::    r_{MRI} = Tr_{dfs}

.. _BABBHHHE:

Converting MRI data into the fif format
#######################################

The utility mne_make_cor_set creates
a fif format MRI description
file optionally including the MRI data using FreeSurfer MRI volume
data as input. The command-line options are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---dir <*directory*>**

    Specifies a directory containing the MRI volume in COR format. Any
    previous ``--mgh`` options are cancelled when this option
    is encountered.

**\---withdata**

    Include the pixel data to the output file. This option is implied
    with the ``--mgh`` option.

**\---mgh <*name*>**

    An MRI volume volume file in mgh or mgz format.
    The ``--withdata`` option is implied with this type of
    input. Furthermore, the :math:`T_3` transformation,
    the Talairach transformation :math:`T_4` from
    the talairach.xfm file referred to in the MRI volume, and the the
    fixed transforms :math:`T_-` and :math:`T_+` will
    added to the output file. For definition of the coordinate transformations,
    see :ref:`CHDEDFIB`.

**\---talairach <*name*>**

    Take the Talairach transform from this file instead of the one specified
    in mgh/mgz files.

**\---out <*name*>**

    Specifies the output file, which is a fif-format MRI description
    file.

.. _BABBIFIJ:

Collecting coordinate transformations into one file
###################################################

The utility mne_collect_transforms collects
coordinate transform information from various sources and saves
them into a single fif file. The coordinate transformations used
by MNE software are summarized in Figure 5.1. The output
of mne_collect_transforms may
include all transforms referred to therein except for the sensor
coordinate system transformations :math:`T_{s_1} \dotso T_{s_n}`.
The command-line options are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---meas <*name*>**

    Specifies a measurement data file which provides :math:`T_1`.
    A forward solution or an inverse operator file can also be specified
    as implied by Table 5.1.

**\---mri <*name*>**

    Specifies an MRI description or a standalone coordinate transformation
    file produced by mne_analyze which
    provides :math:`T_2`. If the ``--mgh`` option
    is not present mne_collect_transforms also
    tries to find :math:`T_3`, :math:`T_4`, :math:`T_-`,
    and :math:`T_+` from this file.

**\---mgh <*name*>**

    An MRI volume volume file in mgh or mgz format.
    This file provides :math:`T_3`. The transformation :math:`T_4` will
    be read from the talairach.xfm file referred to in the MRI volume.
    The fixed transforms :math:`T_-` and :math:`T_+` will
    also be created.

**\---out <*name*>**

    Specifies the output file. If this option is not present, the collected transformations
    will be output on screen but not saved.

.. _BEHCHGHD:

Converting an ncov covariance matrix file to fiff
#################################################

The ncov file format was used to store the noise-covariance
matrix file. The MNE software requires that the covariance matrix
files are in fif format. The utility mne_convert_ncov converts
ncov files to fif format.

The command-line options are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---ncov <*name*>**

    The ncov file to be converted.

**\---meas <*name*>**

    A fif format measurement file used to assign channel names to the noise-covariance
    matrix elements. This file should have precisely the same channel
    order within MEG and EEG as the ncov file. Typically, both the ncov
    file and the measurement file are created by the now mature off-line
    averager, meg_average .

.. _BEHCDBHG:

Converting a lisp covariance matrix to fiff
###########################################

The utility mne_convert_lspcov converts a LISP-format noise-covariance file,
produced by the Neuromag signal processor, graph into fif format.

The command-line options are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---lspcov <*name*>**

    The LISP noise-covariance matrix file to be converted.

**\---meas <*name*>**

    A fif format measurement file used to assign channel names to the noise-covariance
    matrix elements. This file should have precisely the same channel
    order within MEG and EEG as the LISP-format covariance matrix file.

**\---out <*name*>**

    The name of a fif format output file. The file name should end with
    -cov.fif.text format output file. No information about the channel names
    is included. The covariance matrix file is listed row by row. This
    file can be loaded to MATLAB, for example

**\---outasc <*name*>**

    The name of a text format output file. No information about the channel
    names is included. The covariance matrix file is listed row by row.
    This file can be loaded to MATLAB, for example

.. _BEHCCEBJ:

The MNE data file conversion tool
#################################

This utility, called mne_convert_mne_data ,
allows the conversion of various fif files related to the MNE computations
to other formats. The two principal purposes of this utility are
to facilitate development of new analysis approaches with Matlab
and conversion of the forward model and noise covariance matrix
data into evoked-response type fif files, which can be accessed
and displayed with the Neuromag source modelling software.

.. note:: Most of the functions of mne_convert_mne_data are    now covered by the MNE Matlab toolbox covered in :ref:`ch_matlab`.    This toolbox is recommended to avoid creating additional files occupying    disk space.

.. _BEHCICCF:

Command-line options
====================

The command-line options recognize
by mne_convert_mne_data are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---fwd <*name*>**

    Specity the name of the forward solution file to be converted. Channels
    specified with the ``--bad`` option will be excluded from
    the file.

**\---fixed**

    Convert the forward solution to the fixed-orientation mode before outputting
    the converted file. With this option only the field patterns corresponding
    to a dipole aligned with the estimated cortex surface normal are
    output.

**\---surfsrc**

    When outputting a free-orientation forward model (three orthogonal dipole
    components present) rotate the dipole coordinate system at each
    source node so that the two tangential dipole components are output
    first, followed by the field corresponding to the dipole aligned
    with the estimated cortex surface normal. The orientation of the
    first two dipole components in the tangential plane is arbitrarily selected
    to create an orthogonal coordinate system.

**\---noiseonly**

    When creating a 'measurement' fif file, do not
    output a forward model file, just the noise-covariance matrix.

**\---senscov <*name*>**

    Specifies the fif file containing a sensor covariance matrix to
    be included with the output. If no other input files are specified
    only the covariance matrix is output

**\---srccov <*name*>**

    Specifies the fif file containing the source covariance matrix to
    be included with the output. Only diagonal source covariance files
    can be handled at the moment.

**\---bad <*name*>**

    Specifies the name of the file containing the names of the channels to
    be omitted, one channel name per line. This does not affect the output
    of the inverse operator since the channels have been already selected
    when the file was created.

**\---fif**

    Output the forward model and the noise-covariance matrix into 'measurement' fif
    files. The forward model files are tagged with <*modalities*> ``-meas-fwd.fif`` and
    the noise-covariance matrix files with <*modalities*> ``-meas-cov.fif`` .
    Here, modalities is ``-meg`` if MEG is included, ``-eeg`` if
    EEG is included, and ``-meg-eeg`` if both types of signals
    are present. The inclusion of modalities is controlled by the ``--meg`` and ``--eeg`` options.

**\---mat**

    Output the data into MATLAB mat files. This is the default. The
    forward model files are tagged with <*modalities*> ``-fwd.mat`` forward model
    and noise-covariance matrix output, with ``-inv.mat`` for inverse
    operator output, and with ``-inv-meas.mat`` for combined inverse
    operator and measurement data output, respectively. The meaning
    of <*modalities*> is the same
    as in the fif output, described above.

**\---tag <*name*>**

    By default, all variables in the matlab output files start with
    ``mne\_``. This option allows to change this prefix to <*name*> _.

**\---meg**

    Include MEG channels from the forward solution and noise-covariance
    matrix.

**\---eeg**

    Include EEG channels from the forward solution and noise-covariance
    matrix.

**\---inv <*name*>**

    Output the inverse operator data from the specified file into a
    mat file. The source and noise covariance matrices as well as active channels
    have been previously selected when the inverse operator was created
    with mne_inverse_operator . Thus
    the options ``--meg`` , ``--eeg`` , ``--senscov`` , ``--srccov`` , ``--noiseonly`` ,
    and ``--bad`` do not affect the output of the inverse operator.

**\---meas <*name*>**

    Specifies the file containing measurement data to be output together with
    the inverse operator. The channels corresponding to the inverse operator
    are automatically selected from the file if ``--inv`` .
    option is present. Otherwise, the channel selection given with ``--sel`` option will
    be taken into account.

**\---set <*number*>**

    Select the data set to be output from the measurement file.

**\---bmin <*value/ms*>**

    Specifies the baseline minimum value setting for the measurement signal
    output.

**\---bmax <*value/ms*>**

    Specifies the baseline maximum value setting for the measurement signal
    output.

.. note:: The ``--tmin`` and ``--tmax`` options    which existed in previous versions of mne_converted_mne_data have    been removed. If output of measurement data is requested, the entire    averaged epoch is now included.

Guide to combining options
==========================

The combination of options is quite complicated. The :ref:`BEHDCIII` should be
helpful to determine the combination of options appropriate for your needs.


.. tabularcolumns:: |p{0.38\linewidth}|p{0.1\linewidth}|p{0.2\linewidth}|p{0.3\linewidth}|
.. _BEHDCIII:
.. table:: Guide to combining mne_convert_mne_data options.

    +-------------------------------------+---------+--------------------------+-----------------------+
    | Desired output                      | Format  | Required options         | Optional options      |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | forward model                       | fif     |   \---fwd <*name*>       | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       | \---surfsrc           |
    |                                     |         |   \---meg and/or \---eeg |                       |
    |                                     |         |   \---fif                |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | forward model                       | mat     |   \---fwd <*name*>       | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       | \---surfsrc           |
    |                                     |         |   \---meg and/or --eeg   |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | forward model and sensor covariance | mat     |   \---fwd <*name*>       | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       | \---surfsrc           |
    |                                     |         |   \---senscov <*name*>   |                       |
    |                                     |         |   \---meg and/or --eeg   |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | sensor covariance                   | fif     |   \---fwd <*name*>       | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       |                       |
    |                                     |         |   \---senscov <*name*>   |                       |
    |                                     |         |   \---noiseonly          |                       |
    |                                     |         |   \---fif                |                       |
    |                                     |         |   \---meg and/or --eeg   |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | sensor covariance                   | mat     |   \---senscov <*name*>   | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | sensor covariance eigenvalues       | text    |   \---senscov <*name*>   | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       |                       |
    |                                     |         |   \---eig                |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | evoked MEG/EEG data                 | mat     |   \---meas <*name*>      | \---sel <*name*>      |
    |                                     |         |   \---out <*name*>       | \---set <*number*>    |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | evoked MEG/EEG data forward model   | mat     |   \---meas <*name*>      | \---bad <*name*>      |
    |                                     |         |   \---fwd <*name*>       | \---set <*number*>    |
    |                                     |         |   \---out <*name*>       |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | inverse operator data               | mat     |   \---inv <*name*>       |                       |
    |                                     |         |   \---out <*name*>       |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | inverse operator data evoked        | mat     |   \–--inv <*name*>       |                       |
    | MEG/EEG data                        |         |   \–--meas <*name*>      |                       |
    |                                     |         |   \–--out <*name*>       |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+

Matlab data structures
======================

The Matlab output provided by mne_convert_mne_data is
organized in structures, listed in :ref:`BEHCICCA`. The fields
occurring in these structures are listed in :ref:`BABCBIGF`.

The symbols employed in variable size descriptions are:

**nloc**

    Number
    of source locations

**nsource**

    Number
    of sources. For fixed orientation sources nsource = nloc whereas nsource = 3*nloc for
    free orientation sources

**nchan**

    Number
    of measurement channels.

**ntime**

    Number
    of time points in the measurement data.

.. _BEHCICCA:
.. table:: Matlab structures produced by mne_convert_mne_data.

    ===============  =======================================
    Structure        Contents
    ===============  =======================================
    <*tag*> _meas      Measured data
    <*tag*> _inv       The inverse operator decomposition
    <*tag*> _fwd       The forward solution
    <*tag*> _noise     A standalone noise-covariance matrix
    ===============  =======================================

The prefix given with the ``--tag`` option is indicated <*tag*> , see :ref:`BEHCICCF`. Its default value is MNE.


.. tabularcolumns:: |p{0.14\linewidth}|p{0.13\linewidth}|p{0.73\linewidth}|
.. _BABCBIGF:
.. table:: The fields of Matlab structures.


    +-----------------------+-----------------+------------------------------------------------------------+
    | Variable              | Size            | Description                                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | fwd                   | nsource x nchan | The forward solution, one source on each row. For free     |
    |                       |                 | orientation sources, the fields of the three orthogonal    |
    |                       |                 | dipoles for each location are listed consecutively.        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | names ch_names        | nchan (string)  | String array containing the names of the channels included |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_types              | nchan x 2       | The column lists the types of the channels (1 = MEG,       |
    |                       |                 | 2 = EEG). The second column lists the coil types, see      |
    |                       |                 | :ref:`BGBBHGEC` and :ref:`CHDBDFJE`. For EEG electrodes,   |
    |                       |                 | this value equals one.                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_pos                | nchan x 3       | The location information for each channel. The first three |
    |                       |                 | values specify the origin of the sensor coordinate system  |
    |                       |                 | or the location of the electrode. For MEG channels, the    |
    |                       |                 | following nine number specify the *x*, *y*, and            |
    |                       |                 | *z*-direction unit vectors of the sensor coordinate system.|
    |                       |                 | For EEG electrodes the first unit vector specifies the     |
    |                       |                 | location of the reference electrode. If the reference is   |
    |                       |                 | not specified this value is all zeroes. The remaining unit |
    |                       |                 | vectors are irrelevant for EEG electrodes.                 |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_lognos             | nchan x 1       | Logical channel numbers as listed in the fiff file         |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_units              | nchan x 2       | Units and unit multipliers as listed in the fif file. The  |
    |                       |                 | unit of the data is listed in the first column (T = 112,   |
    |                       |                 | T/m = 201, V = 107). At present, the second column will be |
    |                       |                 | always zero, *i.e.*, no unit multiplier.                   |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_cals               | nchan x 2       | Even if the data comes from the conversion already         |
    |                       |                 | calibrated, the original calibration factors are included. |
    |                       |                 | The first column is the range member of the fif data       |
    |                       |                 | structures and while the second is the cal member. To get  |
    |                       |                 | calibrated values in the units given in ch_units from the  |
    |                       |                 | raw data, the data must be multiplied with the product of  |
    |                       |                 | range and cal.                                             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | sfreq                 | 1               | The sampling frequency in Hz.                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | lowpass               | 1               | Lowpass filter frequency (Hz)                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | highpass              | 1               | Highpass filter frequency (Hz)                             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | source_loc            | nloc x 3        | The source locations given in the coordinate frame         |
    |                       |                 | indicated by the coord_frame member.                       |
    +-----------------------+-----------------+------------------------------------------------------------+
    | source_ori            | nsource x 3     | The source orientations                                    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | source_selection      | nsource x 2     | Indication of the sources selected from the complete source|
    |                       |                 | spaces. Each row contains the number of the source in the  |
    |                       |                 | complete source space (starting with 0) and the source     |
    |                       |                 | space number (1 or 2). These numbers refer to the order the|
    |                       |                 | two hemispheres where listed when mne_make_source_space was|
    |                       |                 | invoked. mne_setup_source_space lists the left hemisphere  |
    |                       |                 | first.                                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | coord_frame           | string          | Name of the coordinate frame employed in the forward       |
    |                       |                 | calculations. Possible values are 'head' and 'mri'.        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | mri_head_trans        | 4 x 4           | The coordinate frame transformation from mri the MEG 'head'|
    |                       |                 | coordinates.                                               |
    +-----------------------+-----------------+------------------------------------------------------------+
    | meg_head_trans        | 4 x 4           | The coordinate frame transformation from the MEG device    |
    |                       |                 | coordinates to the MEG head coordinates                    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | noise_cov             | nchan x nchan   | The noise covariance matrix                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | source_cov            | nsource         | The elements of the diagonal source covariance matrix.     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | sing                  | nchan           | The singular values of                                     |
    |                       |                 | :math:`A = C_0^{-^1/_2} G R^C = U \Lambda V^T`             |
    |                       |                 | with :math:`R` selected so that                            |
    |                       |                 | :math:`\text{trace}(AA^T) / \text{trace}(I) = 1`           |
    |                       |                 | as discussed in :ref:`CHDDHAGE`                            |
    +-----------------------+-----------------+------------------------------------------------------------+
    | eigen_fields          | nchan x nchan   | The rows of this matrix are the left singular vectors of   |
    |                       |                 | :math:`A`, i.e., the columns of :math:`U`, see above.      |
    +-----------------------+-----------------+------------------------------------------------------------+
    | eigen_leads           | nchan x nsource | The rows of this matrix are the right singular vectors of  |
    |                       |                 | :math:`A`, i.e., the columns of :math:`V`, see above.      |
    +-----------------------+-----------------+------------------------------------------------------------+
    | noise_eigenval        | nchan           | In terms of :ref:`CHDDHAGE`, eigenvalues of :math:`C_0`,   |
    |                       |                 | i.e., not scaled with number of averages.                  |
    +-----------------------+-----------------+------------------------------------------------------------+
    | noise_eigenvec        | nchan           | Eigenvectors of the noise covariance matrix. In terms of   |
    |                       |                 | :ref:`CHDDHAGE`, :math:`U_C^T`.                            |
    +-----------------------+-----------------+------------------------------------------------------------+
    | data                  | nchan x ntime   | The measured data. One row contains the data at one time   |
    |                       |                 | point.                                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | times                 | ntime           | The time points in the above matrix in seconds             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nave                  | 1               | Number of averages as listed in the data file.             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | meas_times            | ntime           | The time points in seconds.                                |
    +-----------------------+-----------------+------------------------------------------------------------+

.. _convert_to_matlab:

Converting raw data to Matlab format
####################################

The utility mne_raw2mat converts
all or selected channels from a raw data file to a Matlab mat file.
In addition, this utility can provide information about the raw
data file so that the raw data can be read directly from the original
fif file using Matlab file I/O routines.

.. note:: The MNE Matlab toolbox described in :ref:`ch_matlab` provides    direct access to raw fif files without a need for conversion to    mat file format first. Therefore, it is recommended that you use    the Matlab toolbox rather than  mne_raw2mat which    creates large files occupying disk space unnecessarily.

Command-line options
====================

mne_raw2mat accepts the
following command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---raw <*name*>**

    Specifies the name of the raw data fif file to convert.

**\---mat <*name*>**

    Specifies the name of the destination Matlab file.

**\---info**

    With this option present, only information about the raw data file
    is included. The raw data itself is omitted.

**\---sel <*name*>**

    Specifies a text file which contains the names of the channels to include
    in the output file, one channel name per line. If the ``--info`` option
    is specified, ``--sel`` does not have any effect.

**\---tag <*tag*>**

    By default, all Matlab variables included in the output file start
    with ``mne\_``. This option changes the prefix to <*tag*> _.

Matlab data structures
======================

The Matlab files output by mne_raw2mat can
contain two data structures, <*tag*>_raw and <*tag*>_raw_info .
If ``--info`` option is specifed, the file contains the
latter structure only.

The <*tag*>_raw structure
contains only one field, data which
is a matrix containing the raw data. Each row of this matrix constitutes
the data from one channel in the original file. The data type of
this matrix is the same of the original data (2-byte signed integer,
4-byte signed integer, or single-precision float).

The fields of the <*tag*>_raw_info structure
are listed in :ref:`BEHFDCIH`. Further explanation of the bufs field
is provided in :ref:`BEHJEIHJ`.


.. tabularcolumns:: |p{0.2\linewidth}|p{0.15\linewidth}|p{0.6\linewidth}|
.. _BEHFDCIH:
.. table:: The fields of the raw data info structure.

    +-----------------------+-----------------+------------------------------------------------------------+
    | Variable              | Size            | Description                                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | orig_file             | string          | The name of the original fif file specified with the       |
    |                       |                 | ``--raw`` option.                                          |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nchan                 | 1               | Number of channels.                                        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nsamp                 | 1               | Total number of samples                                    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | bufs                  | nbuf x 4        | This field is present if ``--info`` option was specified on|
    |                       |                 | the command line. For details, see :ref:`BEHJEIHJ`.        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | sfreq                 | 1               | The sampling frequency in Hz.                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | lowpass               | 1               | Lowpass filter frequency (Hz)                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | highpass              | 1               | Highpass filter frequency (Hz)                             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_names              | nchan (string)  | String array containing the names of the channels included |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_types              | nchan x 2       | The column lists the types of the channesl (1 = MEG, 2 =   |
    |                       |                 | EEG). The second column lists the coil types, see          |
    |                       |                 | :ref:`BGBBHGEC` and :ref:`CHDBDFJE`. For EEG electrodes,   |
    |                       |                 | this value equals one.                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_lognos             | nchan x 1       | Logical channel numbers as listed in the fiff file         |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_units              | nchan x 2       | Units and unit multipliers as listed in the fif file.      |
    |                       |                 | The unit of the data is listed in the first column         |
    |                       |                 | (T = 112, T/m = 201, V = 107). At present, the second      |
    |                       |                 | column will be always zero, *i.e.*, no unit multiplier.    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_pos                | nchan x 12      | The location information for each channel. The first three |
    |                       |                 | values specify the origin of the sensor coordinate system  |
    |                       |                 | or the location of the electrode. For MEG channels, the    |
    |                       |                 | following nine number specify the *x*, *y*, and            |
    |                       |                 | *z*-direction unit vectors of the sensor coordinate system.|
    |                       |                 | For EEG electrodes the first vector after the electrode    |
    |                       |                 | location specifies the location of the reference electrode.|
    |                       |                 | If the reference is not specified this value is all zeroes.|
    |                       |                 | The remaining unit vectors are irrelevant for EEG          |
    |                       |                 | electrodes.                                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_cals               | nchan x 2       | The raw data output by mne_raw2mat is uncalibrated.        |
    |                       |                 | The first column is the range member of the fiff data      |
    |                       |                 | structures and while the second is the cal member. To get  |
    |                       |                 | calibrared data values in the units given in ch_units from |
    |                       |                 | the raw data, the data must be multiplied with the product |
    |                       |                 | of range and cal .                                         |
    +-----------------------+-----------------+------------------------------------------------------------+
    | meg_head_trans        | 4 x 4           | The coordinate frame transformation from the MEG device    |
    |                       |                 | coordinates to the MEG head coordinates.                   |
    +-----------------------+-----------------+------------------------------------------------------------+


.. tabularcolumns:: |p{0.1\linewidth}|p{0.6\linewidth}|
.. _BEHJEIHJ:
.. table:: The bufs member of the raw data info structure.

    +-----------------------+-------------------------------------------------------------------------+
    | Column                | Contents                                                                |
    +-----------------------+-------------------------------------------------------------------------+
    | 1                     | The raw data type (2 or 16 = 2-byte signed integer, 3 = 4-byte signed   |
    |                       | integer, 4 = single-precision float). All data in the fif file are      |
    |                       | written in the big-endian byte order. The raw data are stored sample by |
    |                       | sample.                                                                 |
    +-----------------------+-------------------------------------------------------------------------+
    | 2                     | Byte location of this buffer in the original fif file.                  |
    +-----------------------+-------------------------------------------------------------------------+
    | 3                     | First sample of this buffer. Since raw data storing can be switched on  |
    |                       | and off during the acquisition, there might be gaps between the end of  |
    |                       | one buffer and the beginning of the next.                               |
    +-----------------------+-------------------------------------------------------------------------+
    | 4                     | Number of samples in the buffer.                                        |
    +-----------------------+-------------------------------------------------------------------------+

.. _BEHFIDCB:

Converting epochs to Matlab format
##################################

The utility mne_epochs2mat converts
epoch data including all or selected channels from a raw data file
to a simple binary file with an associated description file in Matlab
mat file format. With help of the description file, a matlab program
can easily read the epoch data from the simple binary file. Signal
space projection and bandpass filtering can be optionally applied
to the raw data prior to saving the epochs.

.. note:: The MNE Matlab toolbox described in :ref:`ch_matlab` provides direct    access to raw fif files without conversion with mne_epochs2mat first.    Therefore, it is recommended that you use the Matlab toolbox rather than mne_epochs2mat which    creates large files occupying disk space unnecessarily. An exception    to this is the case where you apply a filter to the data and save    the band-pass filtered epochs.

Command-line options
====================

mne_epochs2mat accepts
the following command-line options are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---raw <*name*>**

    Specifies the name of the raw data fif file to use as input.

**\---mat <*name*>**

    Specifies the name of the destination file. Anything following the last
    period in the file name will be removed before composing the output
    file name. The binary epoch file will be called <*trimmed name*> ``.epochs`` and
    the corresponding Matlab description file will be <*trimmed name*> ``_desc.mat`` .

**\---tag <*tag*>**

    By default, all Matlab variables included in the description file
    start with ``mne\_``. This option changes the prefix to <*tag*> _.

**\---events <*name*>**

    The file containing the event definitions. This can be a text or
    fif format file produced by mne_process_raw or mne_browse_raw ,
    see :ref:`CACBCEGC`. With help of this file it is possible
    to select virtually any data segment from the raw data file. If
    this option is missing, the digital trigger channel in the raw data
    file or a fif format event file produced automatically by mne_process_raw or mne_browse_raw is
    consulted for event information.

**\---event <*name*>**

    Event number identifying the epochs of interest.

**\---tmin <*time/ms*>**

    The starting point of the epoch with respect to the event of interest.

**\---tmax <*time/ms*>**

    The endpoint of the epoch with respect to the event of interest.

**\---sel <*name*>**

    Specifies a text file which contains the names of the channels to include
    in the output file, one channel name per line. If the ``--inv`` option
    is specified, ``--sel`` is ignored. If neither ``--inv`` nor ``--sel`` is
    present, all MEG and EEG channels are included. The digital trigger
    channel can be included with the ``--includetrig`` option, described
    below.

**\---inv <*name*>**

    Specifies an inverse operator, which will be employed in two ways. First,
    the channels included to output will be those included in the inverse
    operator. Second, any signal-space projection operator present in
    the inverse operator file will be applied to the data. This option
    cancels the effect of ``--sel`` and ``--proj`` options.

**\---digtrig <*name*>**

    Name of the composite digital trigger channel. The default value
    is 'STI 014'. Underscores in the channel name
    will be replaced by spaces.

**\---digtrigmask <*number*>**

    Mask to be applied to the trigger channel values before considering them.
    This option is useful if one wants to set some bits in a don't care
    state. For example, some finger response pads keep the trigger lines
    high if not in use, *i.e.*, a finger is not in
    place. Yet, it is convenient to keep these devices permanently connected
    to the acquisition system. The number can be given in decimal or
    hexadecimal format (beginning with 0x or 0X). For example, the value
    255 (0xFF) means that only the lowest order byte (usually trigger
    lines 1 - 8 or bits 0 - 7) will be considered.

**\---includetrig**

    Add the digital trigger channel to the list of channels to output.
    This option should not be used if the trigger channel is already
    included in the selection specified with the ``--sel`` option.

**\---filtersize <*size*>**

    Adjust the length of the FFT to be applied in filtering. The number will
    be rounded up to the next power of two. If the size is :math:`N`,
    the corresponding length of time is :math:`^N/_{f_s}`,
    where :math:`f_s` is the sampling frequency
    of your data. The filtering procedure includes overlapping tapers
    of length :math:`^N/_2` so that the total FFT
    length will actually be :math:`2N`. The default
    value is 4096.

**\---highpass <*value/Hz*>**

    Highpass filter frequency limit. If this is too low with respect
    to the selected FFT length and data file sampling frequency, the
    data will not be highpass filtered. You can experiment with the
    interactive version to find the lowest applicable filter for your
    data. This value can be adjusted in the interactive version of the
    program. The default is 0, i.e., no highpass filter in effect.

**\---highpassw <*value/Hz*>**

    The width of the transition band of the highpass filter. The default
    is 6 frequency bins, where one bin is :math:`^{f_s}/_{(2N)}`.

**\---lowpass <*value/Hz*>**

    Lowpass filter frequency limit. This value can be adjusted in the interactive
    version of the program. The default is 40 Hz.

**\---lowpassw <*value/Hz*>**

    The width of the transition band of the lowpass filter. This value
    can be adjusted in the interactive version of the program. The default
    is 5 Hz.

**\---filteroff**

    Do not filter the data.

**\---proj <*name*>**

    Include signal-space projection (SSP) information from this file.
    If the ``--inv`` option is present, ``--proj`` has
    no effect.

.. note:: Baseline has not been subtracted from the epochs. This has to be done in subsequent processing with Matlab if so desired.

.. note:: Strictly speaking, trigger mask value zero would mean that all trigger inputs are ignored. However, for convenience,    setting the mask to zero or not setting it at all has the same effect    as 0xFFFFFFFF, *i.e.*, all bits set.

.. note:: The digital trigger channel can also be set with the MNE_TRIGGER_CH_NAME environment variable. Underscores in the variable    value will *not* be replaced with spaces by mne_browse_raw or mne_process_raw .    Using the ``--digtrig`` option supersedes the MNE_TRIGGER_CH_NAME    environment variable.

.. note:: The digital trigger channel mask can also be    set with the MNE_TRIGGER_CH_MASK environment variable. Using the ``--digtrigmask`` option    supersedes the MNE_TRIGGER_CH_MASK environment variable.

The binary epoch data file
==========================

mne_epochs2mat saves the
epoch data extracted from the raw data file is a simple binary file.
The data are stored as big-endian single-precision floating point
numbers. Assuming that each of the total of :math:`p` epochs
contains :math:`n` channels and :math:`m` time
points, the data :math:`s_{jkl}` are ordered
as

.. math::    s_{111} \dotso s_{1n1} s_{211} \dotso s_{mn1} \dotso s_{mnp}\ ,

where the first index stands for the time point, the second
for the channel, and the third for the epoch number, respectively.
The data are not calibrated, i.e., the calibration factors present
in the Matlab description file have to be applied to get to physical
units as described below.

.. note:: The maximum size of an epoch data file is 2 Gbytes, *i.e.*, 0.5 Gsamples.

Matlab data structures
======================

The Matlab description files output by mne_epochs2mat contain
a data structure <*tag*>_epoch_info .
The fields of the this structure are listed in :ref:`BEHFDCIH`.
Further explanation of the epochs member
is provided in :ref:`BEHHAGHE`.


.. tabularcolumns:: |p{0.15\linewidth}|p{0.15\linewidth}|p{0.6\linewidth}|
.. _BEHIFJIJ:
.. table:: The fields of the raw data info structure.

    +-----------------------+-----------------+------------------------------------------------------------+
    | Variable              | Size            | Description                                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | orig_file             | string          | The name of the original fif file specified with the       |
    |                       |                 | ``--raw`` option.                                          |
    +-----------------------+-----------------+------------------------------------------------------------+
    | epoch_file            | string          | The name of the epoch data file produced by mne_epocs2mat. |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nchan                 | 1               | Number of channels.                                        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nepoch                | 1               | Total number of epochs.                                    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | epochs                | nepoch x 5      | Description of the content of the epoch data file,         |
    |                       |                 | see :ref:`BEHHAGHE`.                                       |
    +-----------------------+-----------------+------------------------------------------------------------+
    | sfreq                 | 1               | The sampling frequency in Hz.                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | lowpass               | 1               | Lowpass filter frequency (Hz)                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | highpass              | 1               | Highpass filter frequency (Hz)                             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_names              | nchan (string)  | String array containing the names of the channels included |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_types              | nchan x 2       | The column lists the types of the channels (1 = MEG, 2 =   |
    |                       |                 | EEG). The second column lists the coil types, see          |
    |                       |                 | :ref:`BGBBHGEC` and :ref:`CHDBDFJE`. For EEG electrodes,   |
    |                       |                 | this value equals one.                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_lognos             | nchan x 1       | Logical channel numbers as listed in the fiff file         |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_units              | nchan x 2       | Units and unit multipliers as listed in the fif file.      |
    |                       |                 | The unit of the data is listed in the first column         |
    |                       |                 | (T = 112, T/m = 201, V = 107). At present, the second      |
    |                       |                 | column will be always zero, *i.e.*, no unit multiplier.    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_pos                | nchan x 12      | The location information for each channel. The first three |
    |                       |                 | values specify the origin of the sensor coordinate system  |
    |                       |                 | or the location of the electrode. For MEG channels, the    |
    |                       |                 | following nine number specify the *x*, *y*, and            |
    |                       |                 | *z*-direction unit vectors of the sensor coordinate        |
    |                       |                 | system. For EEG electrodes the first vector after the      |
    |                       |                 | electrode location specifies the location of the reference |
    |                       |                 | electrode. If the reference is not specified this value is |
    |                       |                 | all zeroes. The remaining unit vectors are irrelevant for  |
    |                       |                 | EEG electrodes.                                            |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_cals               | nchan x 2       | The raw data output by mne_raw2mat are not calibrated.     |
    |                       |                 | The first column is the range member of the fiff data      |
    |                       |                 | structures and while the second is the cal member. To      |
    |                       |                 | get calibrated data values in the units given in           |
    |                       |                 | ch_units from the raw data, the data must be multiplied    |
    |                       |                 | with the product of range and cal .                        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | meg_head_trans        | 4 x 4           | The coordinate frame transformation from the MEG device    |
    |                       |                 | coordinates to the MEG head coordinates.                   |
    +-----------------------+-----------------+------------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.6\linewidth}|
.. _BEHHAGHE:
.. table:: The epochs member of the raw data info structure.

    +---------------+------------------------------------------------------------------+
    | Column        | Contents                                                         |
    +---------------+------------------------------------------------------------------+
    | 1             | The raw data type (2 or 16 = 2-byte signed integer, 3 = 4-byte   |
    |               | signed integer, 4 = single-precision float). The epoch data are  |
    |               | written using the big-endian byte order. The data are stored     |
    |               | sample by sample.                                                |
    +---------------+------------------------------------------------------------------+
    | 2             | Byte location of this epoch in the binary epoch file.            |
    +---------------+------------------------------------------------------------------+
    | 3             | First sample of this epoch in the original raw data file.        |
    +---------------+------------------------------------------------------------------+
    | 4             | First sample of the epoch with respect to the event.             |
    +---------------+------------------------------------------------------------------+
    | 5             | Number of samples in the epoch.                                  |
    +---------------+------------------------------------------------------------------+

.. note:: For source modelling purposes, it is recommended    that the MNE Matlab toolbox, see :ref:`ch_matlab` is employed    to read the measurement info instead of using the channel information    in the raw data info structure described in :ref:`BEHIFJIJ`.
