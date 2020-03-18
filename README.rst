.. -*- mode: rst -*-


|Travis|_ |Azure|_ |Circle|_ |Codecov|_ |Zenodo|_

|MNE|_

.. |Travis| image:: https://api.travis-ci.org/mne-tools/mne-python.png?branch=master
.. _Travis: https://travis-ci.org/mne-tools/mne-python/branches

.. |Azure| image:: https://dev.azure.com/mne-tools/mne-python/_apis/build/status/mne-tools.mne-python?branchName=master
.. _Azure: https://dev.azure.com/mne-tools/mne-python/_build/latest?definitionId=1&branchName=master

.. |Circle| image:: https://circleci.com/gh/mne-tools/mne-python.svg?style=svg
.. _Circle: https://circleci.com/gh/mne-tools/mne-python

.. |Codecov| image:: https://codecov.io/gh/mne-tools/mne-python/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/mne-tools/mne-python

.. |Zenodo| image:: https://zenodo.org/badge/5822/mne-tools/mne-python.svg
.. _Zenodo: https://zenodo.org/badge/latestdoi/5822/mne-tools/mne-python

.. |MNE| image:: https://mne.tools/stable/_static/mne_logo.png
.. _MNE: https://mne.tools/dev/

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


Installing MNE-Python
^^^^^^^^^^^^^^^^^^^^^

To install the latest stable version of MNE-Python, you can use pip_ in a terminal:

.. code-block:: bash

    pip install -U mne

**Note** that MNE-Python 0.17 was the last release to support Python 2.
MNE-Python 0.18 only works under Python 3, and MNE-Python 0.19 requires
Python 3.5 or higher.

For more complete instructions and more advanced installation methods (e.g. for
the latest development version), see the `installation guide`_.


Get the latest code
^^^^^^^^^^^^^^^^^^^

To install the latest version of the code using pip_ open a terminal and type:

.. code-block:: bash

    pip install -U https://api.github.com/repos/mne-tools/mne-python/zipball/master

To get the latest code using `git <https://git-scm.com/>`__, open a terminal and type:

.. code-block:: bash

    git clone git://github.com/mne-tools/mne-python.git

Alternatively, you can also download a
`zip file of the latest development version <https://github.com/mne-tools/mne-python/archive/master.zip>`__.


Dependencies
^^^^^^^^^^^^

The minimum required dependencies to run MNE-Python are:

- Python >= 3.5
- NumPy >= 1.13.3
- SciPy >= 1.0.0

For full functionality, some functions require:

- Matplotlib >= 2.1
- Mayavi >= 4.6
- PySurfer >= 0.8
- Scikit-learn >= 0.19.1
- Numba >= 0.40
- NiBabel >= 2.1.0
- Pandas >= 0.21
- Picard >= 0.3
- CuPy >= 4.0 (for NVIDIA CUDA acceleration)
- DIPY >= 0.10.1
- PyVista >= 0.23.1

Contributing to MNE-Python
^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see the documentation on the MNE-Python homepage:

https://mne.tools/dev/install/contributing.html


Mailing list
^^^^^^^^^^^^

http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis


Licensing
^^^^^^^^^

MNE-Python is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2011-2019, authors of MNE-Python.
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


.. _MNE-Python software: https://mne.tools/dev/
.. _MNE documentation: https://mne.tools/dev/overview/index.html
.. _installation guide: https://mne.tools/dev/install/index.html
.. _pip: https://pip.pypa.io/en/stable/
