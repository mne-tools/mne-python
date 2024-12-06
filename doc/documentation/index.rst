.. _documentation_overview:

Documentation overview
======================

.. note::

   If you haven't already installed MNE-Python, please take a look
   at our :ref:`installation guides<installers>`. Please also kindly find some
   resources for :doc:`../help/learn_python` if you need to.


The documentation for MNE-Python is divided into four main sections:

1. The :doc:`../auto_tutorials/index` provide narrative explanations, sample
   code, and expected output for the most common MNE-Python analysis tasks. The
   emphasis is on thorough explanations that get new users up to speed quickly,
   at the expense of covering only a limited number of topics.

2. The :doc:`How-to Examples <../auto_examples/index>` provides working code
   samples demonstrating various analysis and visualization techniques. These
   examples often lack the narrative explanations seen in the tutorials, but
   can be a useful way to discover new analysis or plotting ideas, or to see
   how a particular technique you've read about can be applied using
   MNE-Python.

3. The :doc:`glossary` provides short definitions of MNE-Python-specific
   vocabulary and general neuroimaging concepts. The glossary is often a good
   place to look if you don't understand a term or acronym used somewhere else
   in the documentation.

4. The :doc:`API reference <../api/python_reference>` provides documentation for
   the classes, functions and methods in the MNE-Python codebase. This is the
   same information that is rendered when running
   :samp:`help(mne.{<function_name>})` in an interactive Python session, or
   when typing :samp:`mne.{<function_name>}?` in an IPython session or Jupyter
   notebook.

The rest of the MNE-Python documentation pages (parts outside of the four
categories above) are shown in the navigation menu, including the
:ref:`list of example datasets<datasets>`,
:ref:`implementation details<implementation>`, and more.
Documentation for the related C and MATLAB tools are available here:

- `MNE-MATLAB`_ (repository)
- `MNE-C <MNE-C manual_>`_ (PDF)

.. toctree::
   :hidden:

   Tutorials<../auto_tutorials/index>
   Examples<../auto_examples/index>
   glossary
   Implementation details<implementation>
   design_philosophy
   Example datasets<datasets>
   Command-line tools<../generated/commands>
   ../help/migrating
   cookbook
   cite
   cited

.. include:: ../links.inc
