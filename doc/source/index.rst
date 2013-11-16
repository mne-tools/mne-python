========
MNE Home
========

MNE is a software package for processing magnetoencephalography
(MEG) and electroencephalography (EEG) data. 

The MNE software computes cortically-constrained L2 minimum-norm
current estimates and associated dynamic statistical parametric maps
from MEG and EEG data, optionally constrained by fMRI. 

This software includes MEG and EEG preprocessing tools, interactive
and batch-mode modules for the forward and inverse calculations, as
well as various data conditioning and data conversion utilities. These
tools are provided as compiled C code for the LINUX and Mac OSX
operating systems.

In addition to the compiled C code tools, MNE Software includes a
Matlab toolbox which facilitates access to the fif (functional image
file) format data files employed in our software and enables
development of custom analysis tools based on the intermediate results
computed with the MNE tools. 

The third and newest component of MNE is MNE-Python which implements
all the functionality of the MNE Matlab tools in Python and extends
the capabilities of the MNE Matlab tools to, e.g., frequency-domain
and time-frequency analyses and non-parametric statistics. This
component of MNE is presently evolving quickly and thanks to the
adopted open development environment user contributions can be easily
incorporated.

The Matlab and Python components of MNE are provided under the
simplified BSD license.

  * `Download <http://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php>`_ MNE
  * Read the :ref:`manual`.
  * Get started with :ref:`mne_python`
  * :ref:`command_line_tutorial`
  * Join the MNE `mailing list <http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis>`_
  * `Help/Feature Request/Bug Report <mailto:mne_support@nmr.mgh.harvard.edu>`_
  * :ref:`ch_reading`

.. toctree::
   :maxdepth: 2

   manual
   mne-python
   cite

