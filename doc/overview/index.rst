:orphan:

Documentation overview
======================

.. note::

   If you haven't already installed Python and MNE-Python, here are the
   :doc:`installation instructions <install/index>`, and some resources for
   :doc:`learn_python`.


The documentation for MNE-Python is divided into five main sections:

1. The :doc:`../auto_tutorials/index` provide narrative explanations, sample
   code, and expected output for the most common MNE-Python analysis tasks. The
   emphasis is on thorough explanations that get new users up to speed quickly,
   at the expense of covering only a limited number of topics.

2. The :doc:`Examples Gallery <../auto_examples/index>` provides working code
   samples demonstrating various analysis and visualization techniques. These
   examples often lack the narrative explanations seen in the tutorials, but
   can be a useful way to discover new analysis or plotting ideas, or to see
   how a particular technique you've read about can be applied using
   MNE-Python.

3. The :doc:`../glossary` provides short definitions of MNE-Python-specific
   vocabulary and general neuroimaging concepts. The glossary is often a good
   place to look if you don't understand a term or acronym used somewhere else
   in the documentation.

4. The :doc:`API reference <../python_reference>` provides documentation for
   the classes, functions and methods in the MNE-Python codebase. This is the
   same information that is rendered when running
   :samp:`help(mne.{<function_name>})` in an interactive Python session, or
   when typing :samp:`mne.{<function_name>}?` in an IPython session or Jupyter
   notebook.

5. Resources related to the MNE-C command-line tools, including:

   - The web version of Matti Hämäläinen's original MNE-C
     :ref:`manual <mne_manual_toc>`, including the :ref:`cookbook <cookbook>`
   - The :ref:`C API reference <c_reference>`
   - A guide to :ref:`command_line_tutorial`

   The MNE-Python codebase is also exposed as a set of Unix command-line
   programs; see :doc:`../generated/commands` for details.

The rest of the MNE-Python documentation (parts outside of the five categories
above) are linked here:


.. toctree::
    :maxdepth: 1

    faq
    design_philosophy
    implementation
    cite
    get_help
