.. -*- mode: rst -*-


|Travis|_ |Appveyor|_ |Circle|_ |Codecov|_ |Zenodo|_

|MNE|_

.. |Travis| image:: https://api.travis-ci.org/mne-tools/mne-python.png?branch=master
.. _Travis: https://travis-ci.org/mne-tools/mne-python

.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/7isroetnxsp7hgxv/branch/master?svg=true
.. _Appveyor: https://ci.appveyor.com/project/mne-tools/mne-python/branch/master

.. |Circle| image:: https://circleci.com/gh/mne-tools/mne-python.svg?style=svg
.. _Circle: https://circleci.com/gh/mne-tools/mne-python

.. |Codecov| image:: https://codecov.io/gh/mne-tools/mne-python/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/mne-tools/mne-python

.. |Zenodo| image:: https://zenodo.org/badge/5822/mne-tools/mne-python.svg
.. _Zenodo: https://zenodo.org/badge/latestdoi/5822/mne-tools/mne-python

.. |MNE| image:: https://martinos.org/mne/stable/_static/mne_logo.png
.. _MNE: https://martinos.org/mne

MNE-Python
==========

`MNE-Python software`_ is an open-source Python package for exploring,
visualizing, and analyzing human neurophysiological data such as MEG, EEG, sEEG,
ECoG, and more. It includes modules for data input/output, preprocessing,
visualization, source estimation, time-frequency analysis, connectivity analysis,
machine learning, and statistics.


Documentation
^^^^^^^^^^^^^

`MNE documentation`_ for MNE-Python is available online.


Get the latest code
^^^^^^^^^^^^^^^^^^^

To get the latest code using `git <https://git-scm.com/>`_, open a terminal and type:

.. code-block:: bash

    git clone git://github.com/mne-tools/mne-python.git

Alternatively, you can also download a
`zip file of the latest development version <https://github.com/mne-tools/mne-python/archive/master.zip>`_.


Installing MNE-Python
^^^^^^^^^^^^^^^^^^^^^

To install the latest stable version of MNE-Python, you can use `pip <https://pip.pypa.io/en/stable/>`_ in a terminal:

.. code-block:: bash

    pip install -U mne

For more complete instructions and more advanced installation methods (e.g. for
the latest development version), see the `getting started page`_.


Dependencies
^^^^^^^^^^^^

The minimum required dependencies to run MNE-Python are:

- Python >= 2.7
- NumPy >= 1.8
- SciPy >= 0.12
- Matplotlib >= 1.3

For full functionality, some functions require:

- Scikit-learn >= 0.15 (>= 0.18 recommended)
- NiBabel >= 2.1.0
- Pandas >= 0.13
- Picard >= 0.3

To use `NVIDIA CUDA <https://developer.nvidia.com/cuda-zone>`_ for resampling
and FFT FIR filtering, you will also need to install the NVIDIA CUDA SDK,
pycuda, and scikit-cuda (see the `getting started page`_
for more information).


Contributing to MNE-Python
^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see the documentation on the MNE-Python homepage:

https://martinos.org/mne/contributing.html


Mailing list
^^^^^^^^^^^^

http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis


Licensing
^^^^^^^^^

MNE-Python is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2011-2017, authors of MNE-Python.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of MNE-Python authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**


.. _MNE-Python software: https://martinos.org/mne
.. _MNE documentation: http://martinos.org/mne/documentation.html
.. _getting started page: https://martinos.org/mne/getting_started.html
