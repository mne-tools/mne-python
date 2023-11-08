.. -*- mode: rst -*-

|MNE|_

MNE-Python
==========

MNE-Python is an open-source Python package for exploring,
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

If you find a bug or have an idea for a new feature that should be added to
MNE-Python, please use the
`issue tracker <https://github.com/mne-tools/mne-python/issues/new/choose>`__ of
our GitHub repository.


Installation
^^^^^^^^^^^^

To install the latest stable version of MNE-Python with minimal dependencies
only, use pip_ in a terminal:

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

- `Python <https://www.python.org>`__ ≥ 3.8
- `NumPy <https://numpy.org>`__ ≥ 1.21.2
- `SciPy <https://scipy.org>`__ ≥ 1.7.1
- `Matplotlib <https://matplotlib.org>`__ ≥ 3.5.0
- `Pooch <https://www.fatiando.org/pooch/latest/>`__ ≥ 1.5
- `tqdm <https://tqdm.github.io>`__
- `Jinja2 <https://palletsprojects.com/p/jinja/>`__
- `decorator <https://github.com/micheles/decorator>`__
- `lazy_loader <https://pypi.org/project/lazy_loader/>`__

For full functionality, some functions require:

- `scikit-learn <https://scikit-learn.org/stable/>`__ ≥ 1.0
- `Joblib <https://joblib.readthedocs.io/en/latest/index.html>`__ ≥ 0.15 (for parallelization)
- `mne-qt-browser <https://github.com/mne-tools/mne-qt-browser>`__ ≥ 0.1 (for fast raw data visualization)
- `Qt <https://www.qt.io>`__ ≥ 5.12 via one of the following bindings (for fast raw data visualization and interactive 3D visualization):

  - `PyQt6 <https://www.riverbankcomputing.com/software/pyqt/>`__ ≥ 6.0
  - `PySide6 <https://doc.qt.io/qtforpython-6/>`__ ≥ 6.0
  - `PyQt5 <https://www.riverbankcomputing.com/software/pyqt/>`__ ≥ 5.12
  - `PySide2 <https://doc.qt.io/qtforpython-6/gettingstarted/porting_from2.html>`__ ≥ 5.12

- `Numba <https://numba.pydata.org>`__ ≥ 0.54.0
- `NiBabel <https://nipy.org/nibabel/>`__ ≥ 3.2.1
- `OpenMEEG <https://openmeeg.github.io>`__ ≥ 2.5.6
- `pandas <https://pandas.pydata.org>`__ ≥ 1.3.2
- `Picard <https://pierreablin.github.io/picard/>`__ ≥ 0.3
- `CuPy <https://cupy.dev>`__ ≥ 9.0.0 (for NVIDIA CUDA acceleration)
- `DIPY <https://dipy.org>`__ ≥ 1.4.0
- `imageio <https://imageio.readthedocs.io/en/stable/>`__ ≥ 2.8.0
- `PyVista <https://pyvista.org>`__ ≥ 0.32 (for 3D visualization)
- `PyVistaQt <https://qtdocs.pyvista.org>`__ ≥ 0.4 (for 3D visualization)
- `mffpy <https://github.com/BEL-Public/mffpy>`__ ≥ 0.5.7
- `h5py <https://www.h5py.org>`__
- `h5io <https://github.com/h5io/h5io>`__
- `pymatreader <https://pymatreader.readthedocs.io/en/latest/>`__


Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines <https://mne.tools/dev/development/contributing.html>`__ on our documentation website.


About
^^^^^

+---------+------------+----------------+
| CI      | |Codecov|_ | |Bandit|_      |
+---------+------------+----------------+
| Package | |PyPI|_    | |conda-forge|_ |
+---------+------------+----------------+
| Docs    | |Docs|_    | |Discourse|_   |
+---------+------------+----------------+
| Meta    | |Zenodo|_  | |OpenSSF|_     |
+---------+------------+----------------+


License
^^^^^^^

MNE-Python is **BSD-licensed** (BSD-3-Clause):

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


.. _Documentation: https://mne.tools/dev/
.. _user forum: https://mne.discourse.group
.. _installation guide: https://mne.tools/dev/install/index.html
.. _pip: https://pip.pypa.io/en/stable/

.. |PyPI| image:: https://img.shields.io/pypi/dm/mne.svg?label=PyPI
.. _PyPI: https://pypi.org/project/mne/

.. |conda-forge| image:: https://img.shields.io/conda/dn/conda-forge/mne.svg?label=Conda
.. _conda-forge: https://anaconda.org/conda-forge/mne

.. |Docs| image:: https://img.shields.io/badge/Docs-online-green?label=Documentation
.. _Docs: https://mne.tools/dev/

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.592483.svg
.. _Zenodo: https://doi.org/10.5281/zenodo.592483

.. |Discourse| image:: https://img.shields.io/discourse/status?label=Forum&server=https%3A%2F%2Fmne.discourse.group%2F
.. _Discourse: https://mne.discourse.group/

.. |Codecov| image:: https://img.shields.io/codecov/c/github/mne-tools/mne-python?label=Coverage
.. _Codecov: https://codecov.io/gh/mne-tools/mne-python

.. |Bandit| image:: https://img.shields.io/badge/Security-Bandit-yellow.svg
.. _Bandit: https://github.com/PyCQA/bandit

.. |OpenSSF| image:: https://www.bestpractices.dev/projects/7783/badge
.. _OpenSSF: https://www.bestpractices.dev/projects/7783

.. |MNE| image:: https://mne.tools/stable/_static/mne_logo_gray.svg
.. _MNE: https://mne.tools/dev/
