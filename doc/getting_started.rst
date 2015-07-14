.. _getting_started:

Getting Started
===============

.. XXX do a Getting for both C and Python

* `Download <http://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php>`_ MNE


This page will help you get started with mne-python. If you are at the
Martinos Center, please see this section :ref:`inside_martinos`. If you
would like to use a custom installation of python (or have specific
questions about integrating special tools like IPython notebooks), please
see this section :ref:`detailed_notes`.

New to the Python programming language?
---------------------------------------
This is a very good place to get started: http://scipy-lectures.github.io.

Installing the Python interpreter
---------------------------------

For a fast and up to date scientific Python environment that resolves all
dependencies, we recommend the Anaconda Python distribution:

https://store.continuum.io/cshop/anaconda/

Anaconda is free for academic purposes.

To test that everything works properly, open up IPython::

    ipython --pylab qt

Now that you have a working Python environment you can install MNE.

mne-python installation
-----------------------
Most users should start with the "stable" version of mne-python, which can
be installed this way:

    pip install mne --upgrade

For the newest features (and potentially more bugs), you can instead install
the development version by:

    pip install -e git+https://github.com/mne-tools/mne-python#egg=mne-dev

If you plan to contribute to the project, please follow the git instructions: 
:ref:`contributing`

Checking your installation
--------------------------

To check that everything went fine, in ipython, type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go!
Consider reading the :ref:`detailed_notes` for more advanced options and
speed-related enhancements.

mne-python basics
-----------------
mne-python uses its own custom objects to store M/EEG data. A full description
of these objects is available in the :ref:`api_reference` section.
A typical mne-python workflow is as follows:

.. image:: _static/mne-python_flow.svg

For example, here we use a simplistic pipeline to go from raw data to brain
source time courses in under 30 lines of code. Note that the only step that
requires manual coregistration is the creation of the head-to-mri transform
file :code:`sample_audvis_raw-trans.fif`:

    >>> # import the necesary packages and functions
    >>> from os import path as op
    >>> import mne
    >>> from mne.datasets.sample import data_path

    >>> # structural information (made in Freesurfer and MNE-C)
    >>> subjects_dir = op.join(data_path(), 'subjects')
    >>> bem_dir = op.join(subjects_dir, 'sample', 'bem')
    >>> mri = op.join(data_path(), 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
    >>> src = op.join(bem_dir, 'sample-oct-6-src.fif')
    >>> bem = op.join(bem_dir, 'sample-5120-5120-5120-bem-sol.fif')

    >>> # raw data / experiment information
    >>> event_id, tmin, tmax = 1, -0.2, 0.5
    >>> raw_fname = op.join(data_path(), 'MEG', 'sample', 'sample_audvis_raw.fif')

    >>> # process data
    >>> raw = mne.io.Raw(raw_fname, preload=True)  # Load raw data # doctest: +SKIP
    >>> raw.filter(None, 40)  # Low-pass filter # doctest: +SKIP
    >>> events = mne.find_events(raw, stim_channel='STI 014')  # Extract events # doctest: +SKIP
    >>> epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,  # doctest: +SKIP
                            reject=dict(eeg=80e-6, eog=150e-6))  # Create Epochs # doctest: +SKIP
    >>> evoked = epochs.average()  # Average to create Evoked # doctest: +SKIP
    >>> cov = mne.compute_covariance(epochs, tmax=0)  # Calculate baseline covariance # doctest: +SKIP
    >>> forward = mne.make_forward_solution(evoked.info, mri, src, bem, mindist=5.0)  # doctest: +SKIP
    >>> inverse = mne.minimum_norm.make_inverse_operator(evoked.info, forward, cov)  # doctest: +SKIP
    >>> stc = mne.minimum_norm.apply_inverse(evoked, inverse,  # doctest: +SKIP
                                             lambda2=1. / 9.)  # Source estimates # doctest: +SKIP

Check out :ref:`intro_tutorial` for a more complete explanation
of these steps. Also check out the :ref:`examples-index` for many more
examples showing some of the more advanced features of mne-python.
