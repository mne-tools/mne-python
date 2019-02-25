:orphan:

.. include:: git_links.inc

Documentation
=============

The documentation for MNE-Python is divided into four main sections:

1. The :doc:`auto_tutorials/index` provide narrative explanations, sample code,
   and expected output for the most common MNE-Python analysis tasks. The
   emphasis here is on thorough explanations that get new users up to speed
   quickly, at the expense of covering only a limited number of topics. The
   tutorials are arranged in a fixed order; in theory a user should be able to
   progress through the tutorials without encountering any cases where
   background knowledge is assumed and unexplained.

2. The :doc:`MNE-Python API reference <python_reference>` provides
   documentation for every function and method in the MNE-Python codebase. This
   is the same information that is rendered when running
   ``help(mne.<function_name>)`` in an interactive Python session, or when
   typing ``mne.<function_name>?`` in an IPython session or Jupyter notebook.

3. The :doc:`glossary` provides short definitions of MNE-Python-specific
   vocabulary. The glossary is often a good place to look if you don't
   understand a term used in the API reference for a function.

4. The :doc:`examples gallery <auto_examples/index>` provides working code
   samples demonstrating various analysis and visualization techniques. These
   examples often lack the narrative explanations seen in the tutorials, and do
   not follow any specific order. These examples are a useful way to discover
   new analysis or plotting ideas, or to see how a particular technique you've
   read about can be applied using MNE-Python.

.. note::

   If you haven't already installed Python and MNE-Python, here are the
   :doc:`installation instructions <../getting_started>`.


The rest of this page provides:

- links to resources for :ref:`learning basic Python programming
  <learn_python>` (a necessary prerequisite to using any Python module, and
  MNE-Python is no exception)

- some notes on the :ref:`design philosophy of MNE-Python <design_philosophy>`
  that may help orient new users to what MNE-Python does and does not do

- a :ref:`flowchart` of the conceptual flow of data through MNE-Python


.. _learn_python:

Getting started with Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Python`_ is a modern general-purpose object-oriented high-level programming
language. There are many general introductions to Python online; here are a
few:

- The official `Python tutorial <https://docs.python.org/3/tutorial/index.html>`__
- W3Schools `Python tutorial <https://www.w3schools.com/python/>`__
- Software Carpentry's `Python lesson <http://swcarpentry.github.io/python-novice-inflammation/>`_

Additionally, here are a couple tutorials focused on scientific programming in
Python:

- the `SciPy Lecture Notes <http://scipy-lectures.org/>`_
- `NumPy for MATLAB users <https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html>`_

There are also many video tutorials online, including `videos from the annual
SciPy conferences
<https://www.youtube.com/user/EnthoughtMedia/playlists?shelf_id=1&sort=dd&view=50>`_.
One of those is a `Python introduction for complete beginners
<https://www.youtube.com/watch?v=Xmxy2NU9LOI>`_, but there are many more
lectures on advanced topics available as well.


.. _design_philosophy:

MNE-Python design philosophy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interactive versus scripted analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MNE-Python has some great interactive
plotting abilities that can help you explore your data, and there are a few
GUI-like interactive plotting commands (like browsing through the raw data and
clicking to mark bad channels, or click-and-dragging to annotate bad temporal
spans). But in general it is not possible to use MNE-Python to mouse-click your
way to a finished, publishable analysis. MNE-Python works best when you
assemble your analysis pipeline into one or more Python scripts. On the plus
side, your scripts act as a record of everything you did in your analysis,
making it easy to tweak your analysis later and/or share it with others
(including your future self).

Integration with the scientific python stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MNE-Python also integrates
well with other standard scientific python libraries. For example, MNE-Python
objects underlyingly store their data in NumPy arrays, making it easy to apply
custom algorithms or pass your data into one of `scikit-learn`_'s machine
learning pipelines. MNE-Python's 2-D plotting functions also return matplotlib
figure objects, and the 3D plotting functions return mayavi scenes, so you can
customize your MNE-Python plots using any of matplotlib or mayavi's plotting
commands. The intent is that MNE-Python will get most neuroscientists 90% of
the way to their desired analysis goal, and other packages can get them over
the finish line.

Submodule-based organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A useful-to-know organizing principle is that
MNE-Python objects and functions are separated into submodules. This can help
you discover related functions if you're using an editor that supports
tab-completion. For example, you can type ``mne.preprocessing.<TAB>`` to see
all the functions in the preprocessing submodule; similarly for visualization
functions (:mod:`mne.viz`), functions for reading and writing data
(:mod:`mne.io`), statistics (:mod:`mne.stats`), etc.  This also helps save
keystrokes — instead of::

    import mne
    mne.preprocessing.eog.peak_finder(...)
    mne.preprocessing.eog.find_eog_events(...)
    mne.preprocessing.eog.create_eog_epochs(...)

you can import submodules directly, and use just the submodule name to access
its functions::

    from mne.preprocessing import eog
    eog.peak_finder(...)
    eog.find_eog_events(...)
    eog.create_eog_epochs(...)

.. _`units`:

Internal representation
~~~~~~~~~~~~~~~~~~~~~~~

When importing data, MNE-Python will always convert
measurements to the same standard units. Thus the in-memory representation of
data are always in:

- Volts (eeg, eog, seeg, emg, ecg, bio, ecog)
- Teslas (magnetometers)
- Teslas/meter (gradiometers)
- Amperes*meter (dipole fits)
- Molar (aka mol/L) (fNIRS data: oxyhemoglobin (hbo), deoxyhemoglobin (hbr))
- Arbitrary units (various derived unitless quantities)

Note, however, that most MNE-Python plotting functions will scale the data when
plotted to yield nice-looking axis annotations in a sensible range; for
example, :meth:`mne.io.Raw.plot_psd` will convert teslas to femtoteslas (fT)
and volts to microvolts (μV) when plotting MEG and EEG data.

The units used in internal data representation are particularly important to
remember when extracting data from MNE-Python objects and manipulating it
outside MNE-Python (e.g., when using other python modules for analysis or
plotting).

.. _`precision`:

Floating-point precision
~~~~~~~~~~~~~~~~~~~~~~~~

MNE-Python performs all computation in memory
using the double-precision 64-bit floating point format. This means that the
data is typecast into float64 format as soon as it is read into memory. The
reason for this is that operations such as filtering and preprocessing are
more accurate when using the 64-bit format. However, for backward
compatibility, MNE-Python writes ``.fif`` files in a 32-bit format by default.
This reduces file size when saving data to disk, but beware that saving
*intermediate results* to disk and re-loading them from disk later may lead to
loss in precision. If you would like to ensure 64-bit precision, there are two
possibilities:

- Chain the operations in memory and avoid saving intermediate results.
- Save intermediate results but change the dtype used for saving, using the
  ``fmt`` parameter of the :meth:`mne.io.Raw.save`. However, note that this
  may render the ``.fif`` files unreadable in software packages other than
  MNE-Python.

.. _flowchart:

Conceptual flowchart of MNE-Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid:: conceptual-flowchart.mmd
