MNE: MEG & EEG data analysis package
====================================

.. raw:: html

   <div class="container-fluid">
   <div class="row">
   <div class="col-md-8">
   <br>

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

The MNE development is supported by National Institute of Biomedical Imaging and Bioengineering 
grants 5R01EB009048 and P41EB015896 (Center for Functional Neuroimaging Technologies) as well as 
NSF awards 0958669 and 1042134.

The Matlab and Python components of MNE are provided under the
simplified BSD license.

.. raw:: html

   </div>
   <div class="col-md-4">
   <h2>Documentation</h2>

.. toctree::
   :maxdepth: 1

   getting_started.rst
   tutorials
   examples
   manual
   faq.rst
   whats_new.rst
   python_reference.rst
   advanced_setup.rst
   mne-cpp
   cite
   :ref:`ch_reading`

.. raw:: html

   <h2>Community</h2>

* Join the MNE `mailing list <http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis>`_
* `Help/Feature Request/Bug Report <mailto:mne_support@nmr.mgh.harvard.edu>`_
* :ref:`contributing`

.. raw:: html

   <h2>Versions</h2>

   <ul>
      <li><a href=http://martinos.org/mne/stable>Stable</a></li>
      <li><a href=http://martinos.org/mne/dev>Development</a></li>
   </ul>

.. raw:: html

   </div>
   </div>
   </div>

.. raw:: html

    <div>
    <div style="width: 40%; float: left; padding: 20px;">
        <a class="twitter-timeline" href="https://twitter.com/mne_python" data-widget-id="317730454184804352">Tweets by @mne_python</a>
    </div>
    <div style="width: 40%; float: left; padding: 20px;">
        <script type="text/javascript" src="http://www.ohloh.net/p/586838/widgets/project_basic_stats.js"></script>
    </div>
    </div>
