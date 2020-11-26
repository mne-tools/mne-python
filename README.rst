.. -*- mode: rst -*-

|GH-Linux|_ |GH-macOS|_ |Azure|_ |Circle|_ |Codecov|_ |PyPI|_ |conda-forge|_ |Zenodo|_

|MNE|_

.. |GH-Linux| image:: https://github.com/mne-tools/mne-python/workflows/linux%20/%20conda/badge.svg?branch=master
.. _GH-Linux: https://github.com/mne-tools/mne-python/actions?query=branch:master+event:push

.. |GH-macOS| image:: https://github.com/mne-tools/mne-python/workflows/macos%20/%20conda/badge.svg?branch=master
.. _GH-macOS: https://github.com/mne-tools/mne-python/actions?query=branch:master+event:push

.. |Azure| image:: https://dev.azure.com/mne-tools/mne-python/_apis/build/status/mne-tools.mne-python?branchName=master
.. _Azure: https://dev.azure.com/mne-tools/mne-python/_build/latest?definitionId=1&branchName=master

.. |Circle| image:: https://circleci.com/gh/mne-tools/mne-python.svg?style=shield
.. _Circle: https://circleci.com/gh/mne-tools/mne-python

.. |Codecov| image:: https://codecov.io/gh/mne-tools/mne-python/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/mne-tools/mne-python

.. |PyPI| image:: https://img.shields.io/pypi/dm/mne.svg?label=PyPI%20downloads
.. _PyPI: https://pypi.org/project/mne/

.. |conda-forge| image:: https://img.shields.io/conda/dn/conda-forge/mne.svg?label=Conda%20downloads
.. _conda-forge: https://anaconda.org/conda-forge/mne

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.592483.svg
.. _Zenodo: https://doi.org/10.5281/zenodo.592483

.. |MNE| image:: https://mne.tools/stable/_static/mne_logo.svg
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

- MNE-Python 0.17 was the last release to support Python 2.7
- MNE-Python 0.18 requires Python 3.5 or higher
- MNE-Python 0.21 requires Python 3.6 or higher

For more complete instructions and more advanced installation methods (e.g. for
the latest development version), see the `installation guide`_.


Get the latest code
^^^^^^^^^^^^^^^^^^^

To install the latest version of the code using pip_ open a terminal and type:

.. code-block:: bash

    pip install -U https://github.com/mne-tools/mne-python/archive/master.zip

To get the latest code using `git <https://git-scm.com/>`__, open a terminal and type:

.. code-block:: bash

    git clone git://github.com/mne-tools/mne-python.git

Alternatively, you can also download a
`zip file of the latest development version <https://github.com/mne-tools/mne-python/archive/master.zip>`__.


Dependencies
^^^^^^^^^^^^

The minimum required dependencies to run MNE-Python are:

- Python >= 3.6
- NumPy >= 1.15.4
- SciPy >= 1.1.0

For full functionality, some functions require:

- Matplotlib >= 3.0.3
- Scikit-learn >= 0.20.2
- Numba >= 0.40
- NiBabel >= 2.1.0
- Pandas >= 0.23.4
- Picard >= 0.3
- CuPy >= 4.0 (for NVIDIA CUDA acceleration)
- DIPY >= 0.10.1
- Imageio >= 2.6.1
- PyVista >= 0.24
- pyvistaqt >= 0.2.0
- mffpy >= 0.5.7

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
