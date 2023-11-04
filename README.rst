.. -*- mode: rst -*-

|MNE|_

.. |PyPI| image:: https://img.shields.io/pypi/dm/mne.svg?label=PyPI
.. _PyPI: https://pypi.org/project/mne/

.. |conda-forge| image:: https://img.shields.io/conda/dn/conda-forge/mne.svg?label=Conda
.. _conda-forge: https://anaconda.org/conda-forge/mne

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.592483.svg
.. _Zenodo: https://doi.org/10.5281/zenodo.592483

.. |Discourse| image:: https://img.shields.io/discourse/status?label=Forum&server=https%3A%2F%2Fmne.discourse.group%2F
.. _Discourse: https://mne.discourse.group/

.. |Codecov| image:: https://img.shields.io/codecov/c/github/mne-tools/mne-python?label=Coverage
.. _Codecov: https://codecov.io/gh/mne-tools/mne-python

.. |Bandit| image:: https://img.shields.io/badge/security-bandit-yellow.svg
.. _Bandit: https://github.com/PyCQA/bandit

.. |OpenSSF| image:: https://www.bestpractices.dev/projects/7783/badge
.. _OpenSSF: https://www.bestpractices.dev/projects/7783

.. |MNE| image:: https://mne.tools/stable/_static/mne_logo_small.svg
.. _MNE: https://mne.tools/dev/


MNE-Python
==========

`MNE-Python`_ is an open-source Python package for exploring,
visualizing, and analyzing human neurophysiological data such as MEG, EEG, sEEG,
ECoG, and more. It includes modules for data input/output, preprocessing,
visualization, source estimation, time-frequency analysis, connectivity analysis,
machine learning, statistics, and more.


Documentation
^^^^^^^^^^^^^

`Documentation`_ for MNE-Python encompasses installation instructions, tutorials,
and examples for a wide variety of topics, contributing guidelines, and an API
reference.


Forum
^^^^^^

The `user forum`_ is the best place to ask questions about MNE-Python usage or
the contribution process. The forum also features job opportunities and other
announcements.

If you find a bug or have an idea for new a new feature that should be added to
MNE-Python, please use the
`issue tracker <https://github.com/mne-tools/mne-python/issues/new/choose>`__ of
our GitHub repository.


Installation
^^^^^^^^^^^^

To install the latest stable version of MNE-Python, use pip_ in a terminal:

.. code-block:: console

    $ pip install --upgrade mne

The current MNE-Python release requires Python 3.8 or higher. MNE-Python 0.17
was the last release to support Python 2.7.

For more complete instructions, including our standalone installers and more
advanced installation methods, please refer to the `installation guide`_.


Get the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the latest development version of MNE-Python using pip_, open a
terminal and type:

.. code-block:: console

    $ pip install --upgrade git+https://github.com/mne-tools/mne-python@main

To clone the repository with `git <https://git-scm.com/>`__, open a terminal
and type:

.. code-block:: console

    $ git clone https://github.com/mne-tools/mne-python.git


Dependencies
^^^^^^^^^^^^

The minimum required dependencies to run MNE-Python are:

- Python ≥ 3.8
- NumPy ≥ 1.21.2
- SciPy ≥ 1.7.1
- Matplotlib ≥ 3.5.0
- Pooch ≥ 1.5
- tqdm
- Jinja2
- decorator
- lazy_loader

For full functionality, some functions require:

- scikit-learn ≥ 1.0
- Joblib ≥ 0.15 (for parallelization)
- mne-qt-browser ≥ 0.1 (for fast raw data visualization)
- Qt5 ≥ 5.12 via one of the following bindings (for fast raw data visualization and interactive 3D visualization):

  - PyQt6 ≥ 6.0
  - PySide6 ≥ 6.0
  - PyQt5 ≥ 5.12
  - PySide2 ≥ 5.12

- Numba ≥ 0.54.0
- NiBabel ≥ 3.2.1
- OpenMEEG ≥ 2.5.6
- pandas ≥ 1.3.2
- python-picard ≥ 0.3
- CuPy ≥ 9.0.0 (for NVIDIA CUDA acceleration)
- DIPY ≥ 1.4.0
- imageio ≥ 2.8.0
- PyVista ≥ 0.32 (for 3D visualization)
- PyVistaQt ≥ 0.4 (for 3D visualization)
- mffpy ≥ 0.5.7
- h5py
- h5io
- pymatreader


Contributing
^^^^^^^^^^^^

Please see the instructions on our documentation website:

https://mne.tools/dev/install/contributing.html


About
^^^^^

======= ======================
CI      |Codecov|_ |Bandit|_
Package |PyPI|_ |conda-forge|_
Docs    |Discourse|_
Meta    |Zenodo|_ |OpenSSF|_
======= ======================


License
^^^^^^^

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


.. _MNE-Python: https://mne.tools/dev/
.. _Documentation: https://mne.tools/dev/overview/index.html
.. _user forum: https://mne.discourse.group
.. _installation guide: https://mne.tools/dev/install/index.html
.. _pip: https://pip.pypa.io/en/stable/
