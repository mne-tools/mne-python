.. -*- mode: rst -*-

|PyPI|_ |conda-forge|_ |Zenodo|_ |Discourse|_ |Codecov|_

|MNE|_

.. |PyPI| image:: https://img.shields.io/pypi/dm/mne.svg?label=PyPI
.. _PyPI: https://pypi.org/project/mne/

.. |conda-forge| image:: https://img.shields.io/conda/dn/conda-forge/mne.svg?label=Conda
.. _conda-forge: https://anaconda.org/conda-forge/mne

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.592483.svg
.. _Zenodo: https://doi.org/10.5281/zenodo.592483

.. |Discourse| image:: https://img.shields.io/discourse/status?label=Community&server=https%3A%2F%2Fmne.discourse.group%2F
.. _Discourse: https://mne.discourse.group/

.. |Codecov| image:: https://img.shields.io/codecov/c/github/mne-tools/mne-python?label=Coverage
.. _Codecov: https://codecov.io/gh/mne-tools/mne-python

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


Forum
^^^^^^

Our user forum is https://mne.discourse.group and is the best place to ask
questions about MNE-Python usage or about the contribution process. It also
includes job opportunities and other announcements.


Installing MNE-Python
^^^^^^^^^^^^^^^^^^^^^

To install the latest stable version of MNE-Python, you can use pip_ in a terminal:

.. code-block:: console

    $ pip install --upgrade mne

- MNE-Python 0.17 was the last release to support Python 2.7
- MNE-Python 0.18 requires Python 3.5 or higher
- MNE-Python 0.21 requires Python 3.6 or higher
- MNE-Python 0.24 requires Python 3.7 or higher
- MNE-Python 1.4 requires Python 3.8 or higher

For more complete instructions and more advanced installation methods (e.g. for
the latest development version), see the `installation guide`_.


Get the latest code
^^^^^^^^^^^^^^^^^^^

To install the latest version of the code using pip_ open a terminal and type:

.. code-block:: console

    $ pip install --upgrade git+https://github.com/mne-tools/mne-python@main

To get the latest code using `git <https://git-scm.com/>`__, open a terminal and type:

.. code-block:: console

    $ git clone https://github.com/mne-tools/mne-python.git


Dependencies
^^^^^^^^^^^^

The minimum required dependencies to run MNE-Python are:

- Python >= 3.8
- NumPy >= 1.20.2
- SciPy >= 1.6.3
- Matplotlib >= 3.4.0
- pooch >= 1.5
- tqdm
- Jinja2
- decorator

For full functionality, some functions require:

- Scikit-learn >= 0.24.2
- joblib >= 0.15 (for parallelization control)
- mne-qt-browser >= 0.1 (for fast raw data visualization)
- Qt5 >= 5.12 via one of the following bindings (for fast raw data visualization and interactive 3D visualization):

  - PyQt6 >= 6.0
  - PySide6 >= 6.0
  - PyQt5 >= 5.12
  - PySide2 >= 5.12

- Numba >= 0.53.1
- NiBabel >= 3.2.1
- OpenMEEG >= 2.5.6
- Pandas >= 1.2.4
- Picard >= 0.3
- CuPy >= 9.0.0 (for NVIDIA CUDA acceleration)
- DIPY >= 1.4.0
- Imageio >= 2.8.0
- PyVista >= 0.32 (for 3D visualization)
- pyvistaqt >= 0.4 (for 3D visualization)
- mffpy >= 0.5.7
- h5py
- h5io
- pymatreader

Contributing to MNE-Python
^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see the documentation on the MNE-Python homepage:

https://mne.tools/dev/install/contributing.html


Licensing
^^^^^^^^^

MNE-Python is **BSD-licenced** (BSD-3-Clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2011-2022, authors of MNE-Python.
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
