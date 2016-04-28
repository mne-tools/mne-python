.. _manual:

User Manual
===========

If you are new to MNE, consider first reading the :ref:`cookbook`, as it
gives some simple steps for starting with analysis. The other sections provide
more in-depth information about how to use the software.
You can also jump to the :ref:`api_reference` for specific Python function
and class usage information.

.. contents:: Contents
   :local:
   :depth: 1

.. raw:: html

   <h2>Cookbook</h2>
   A quick run-through of the basic steps involved in M/EEG source analysis.

.. toctree::
   :maxdepth: 2

   cookbook

.. raw:: html

   <h2>Reading your data</h2>
   How to get your raw data loaded in MNE.

.. toctree::
   :maxdepth: 1

   io
   memory

.. raw:: html

   <h2>Preprocessing</h2>
   Dealing with artifacts and noise sources in data.

.. toctree::
   :maxdepth: 1

   preprocessing/ica
   preprocessing/maxwell
   preprocessing/ssp
   channel_interpolation

.. raw:: html

   <h2>Source localization</h2>
   Projecting raw data into source (brain) space.

.. toctree::
   :maxdepth: 1

   source_localization/forward
   source_localization/inverse
   source_localization/morph

.. raw:: html

   <h2>Time-frequency analysis</h2>
   Decomposing time-domain signals into time-frequency representations.

.. toctree::
   :maxdepth: 2

   time_frequency

.. raw:: html

   <h2>Statistics</h2>
   Using parametric and non-parametric tests with M/EEG data.

.. toctree::
   :maxdepth: 2

   statistics

.. raw:: html

   <h2>Decoding</h2>

.. toctree::
   :maxdepth: 2

   decoding

.. raw:: html

   <h2>Datasets</h2>
   How to use dataset fetchers for public data

.. toctree::
   :maxdepth: 2

   datasets_index

.. raw:: html

   <h2>Migrating</h2>

.. toctree::
   :maxdepth: 1

   migrating

.. raw:: html

   <h2>Pitfalls</h2>

.. toctree::
   :maxdepth: 2

   pitfalls

.. raw:: html

   <h2>C Tools</h2>

Additional information about various MNE-C tools.

.. toctree::
   :maxdepth: 1

   c_reference
   gui/analyze
   gui/browse

.. raw:: html

   <h2>MATLAB Tools</h2>
   Information about the MATLAB toolbox.

.. toctree::
   :maxdepth: 2

   matlab

.. raw:: html

   <h2>Appendices</h2>

More details about our implementations and software.

.. toctree::
   :maxdepth: 1

   appendix/bem_model
   appendix/martinos
   appendix/c_misc
   appendix/c_release_notes
   appendix/c_EULA
