.. include:: links.inc

.. _install_python_and_mne_python:

Install Python and MNE-Python
-----------------------------

.. contents:: Steps
   :local:
   :depth: 1

.. note:: Users who work at a facility with a site-wide install of
          MNE (e.g. Martinos center) are encouraged to contact
          their technical staff about how to access and use MNE,
          as the instructions might differ.

.. _install_interpreter:

1. Install a Python interpreter and dependencies
################################################

There are multiple options available for getting a suitable Python interpreter
running on your system. However, for a fast and up to date scientific Python
environment that resolves all dependencies, **we recommend the
Anaconda Python distribution**.

Python has two major versions currently available, 2.7+ and 3.3+. Currently
the 3D visualization dependencies `Mayavi`_ and `PySurfer`_ only run out of the
box on Python 2.7, so **we recommend using Python 2.7**. You can get
Anaconda 2.7 for Windows, OSX, and Linux download and installation instructions
`from the ContinuumIO site <http://docs.continuum.io/anaconda/install>`_.

Once everything is set up, you should be able to check the version
of ``conda`` and ``python`` that is installed:

.. code-block:: bash

    $ conda --version
    conda 4.2.14
    $ which python
    /home/agramfort/anaconda/bin/python
    $ python --version
    Python 2.7.12 :: Continuum Analytics, Inc.

.. note:: If your installation doesn't look something like this,
          *something went wrong* and you should try to fix it. Try looking
          through the Anaconda documentation or Googling for Anaconda
          install tips (StackExchange results are often helpful).

You can then do this to resolve the MNE dependencies:

.. code-block:: bash

    $ conda install scipy matplotlib scikit-learn mayavi ipython-notebook
    $ pip install PySurfer

Now that you have a working Python environment you can install MNE.

Users who would like a MATLAB-like interface should consider using Spyder_,
which can easily be installed ``$ conda install spyder``.

.. _install_mne_python:

2. Install the MNE module
#########################

There are a many options for installing MNE, but two of the most
useful and common are:

1. **Use the stable release version of MNE.** It can be installed as:

   .. code-block:: bash

       $ pip install mne --upgrade

   We tend to release about once every six months, and this
   command can be used to update the install after each release.

.. _installing_master:

2. **Use the development master version of MNE.** If you want to
   be able to update your version between releases for
   bugfixes or new features, this will set you up for frequent updates:

   .. code-block:: bash

       $ git clone git://github.com/mne-tools/mne-python.git
       $ cd mne-python
       $ python setup.py develop

   A feature of ``python setup.py develop`` is that any changes made to
   the files (e.g., by updating to latest ``master``) will be reflected in
   ``mne`` as soon as you restart your Python interpreter. So to update to
   the latest version of the ``master`` development branch, you can do:

   .. code-block:: bash

       $ git pull origin master

   and MNE will be updated to have the latest changes.

If you plan to contribute to MNE, please read how to :ref:`contribute_to_mne`.

3. Check your installation
##########################

To check that everything went fine, in ipython, type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go!

You can launch a web browser to the documentation with::


    >>> mne.open_docs()  # doctest: +SKIP

Along the way, make frequent use of the :ref:`api_reference` and
:ref:`documentation` to understand the capabilities of MNE.

For advanced topics like how to get NVIDIA :ref:`CUDA` support working for ~10x
faster filtering and resampling, or if you're having trouble, visit
:ref:`advanced_setup`.
