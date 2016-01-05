.. _manual:

Manual
======

If you are new to MNE, consider first reading the :ref:`cookbook`, as it
gives some simple steps for starting with analysis. The other sections provide
more in-depth information about how to use the software.
You can also jump to the :ref:`api_reference` for specific Python function
and class usage information.

.. contents:: Contents
   :local:
   :depth: 1


Cookbook
--------

A quick run-through of the basic steps involved in M/EEG source analysis.

.. toctree::
   :maxdepth: 2

   cookbook

Reading your data
-----------------

How to get your raw data loaded in MNE.

.. toctree::
   :maxdepth: 1

   io
   memory

Preprocessing
-------------

Dealing with artifacts and noise sources in data.

.. toctree::
   :maxdepth: 2

   preprocessing/ica
   preprocessing/ssp

Source localization
-------------------

Projecting raw data into source (brain) space.

.. toctree::
   :maxdepth: 1

   source_localization/anatomy
   source_localization/forward
   source_localization/covariance
   source_localization/inverse
   source_localization/morph

Time frequency analysis
-----------------------

Decomposing time-domain signals into time-frequency representations.

.. toctree::
   :maxdepth: 2

   time_frequency

Statistics
----------

Using parametric and non-parametric tests with M/EEG data.

.. toctree::
   :maxdepth: 2

   statistics

Decoding
--------

.. toctree::
   :maxdepth: 3

   decoding

Datasets
--------

To enable reproducibility of results, MNE-Python includes several dataset fetchers

.. toctree::
   :maxdepth: 2

   datasets_index

Pitfalls
--------

.. toctree::
   :maxdepth: 2

   pitfalls

C tools
-------

Additional information about various MNE-C tools.

.. toctree::
   :maxdepth: 1

   c_reference
   gui/analyze
   gui/browse
   c_cookbook


MATLAB tools
------------

Information about the MATLAB toolbox.

.. toctree::
   :maxdepth: 2

   matlab

Appendices
----------

More details about our implementations and software.

.. toctree::
   :maxdepth: 1

   appendix/overview
   appendix/bem_model
   appendix/martinos
   appendix/c_misc
   appendix/c_release_notes
   appendix/c_EULA
