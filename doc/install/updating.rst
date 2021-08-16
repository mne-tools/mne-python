Updating MNE-Python
===================

If you want to update MNE-Python to a newer version, there are a few different
options, depending on how you originally installed it.

.. warning::

    Before performing package upgrade operations, check to make sure that the
    environment you wish to modify has been activated (and if not, call
    ``conda activate name_of_environment`` first).


Upgrading MNE-Python only
^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to update MNE-Python only and leave other packages in their current
state, you can usually safely do this with ``pip``, even if you originally
installed via conda. With the ``mne`` environment active, do:

.. code-block:: console

    $ pip install -U mne


Upgrading all packages
^^^^^^^^^^^^^^^^^^^^^^

Generally speaking, if you want to upgrade *your whole software stack*
including all the dependencies, the best approach is to re-create it as a new
virtual environment, because neither conda nor pip are fool-proof at making
sure all packages remain compatible with one another during upgrades.

Here we'll demonstrate renaming the old environment first, as a safety measure.
We'll assume that the existing environment is called ``mne`` and you want to
rename the old one so that the new, upgraded environment can be called ``mne``
instead. Unfortunately ``conda`` doesn't have a "rename" command so we'll first
clone the old one with a new name (``old_mne``), then delete the original, then
create the new, updated environment re-using the original name. In the first
step we'll also use conda in ``--offline`` mode so that it uses cached
copies of all the packages instead of re-downloading them.

.. code-block:: console

    $ conda create --name old_mne --clone mne --offline  # copy with new name,
    $ conda env remove --name mne --all                  # remove original,
    $ conda create --name mne --channel conda-forge mne  # replace with updated

.. note::

    If you installed extra packages into your old ``mne`` environment,
    you'll need to repeat that process after re-creating the updated
    environment. Comparing the output of ``conda list --name old_mne`` versus
    ``conda list --name mne`` will show you what is missing from the new
    environment. On Linux, you can automate that comparison like this:

    .. code-block:: console

        $ diff <(conda list -n mne | cut -d " " -f 1 | sort) <(conda list -n old_mne | cut -d " " -f 1 | sort) | grep "^>" | cut -d " " -f 2


.. _installing_main:

Upgrading to the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning:: In between releases, function and class APIs can change without
    warning.

Sometimes, new features or bugfixes become available that are important to your
research and you just can't wait for the next official release of MNE-Python to
start taking advantage of them. In such cases, you can use ``pip`` to install
the *development version* of MNE-Python:

.. code-block:: console

    $ pip install -U --no-deps https://github.com/mne-tools/mne-python/archive/main.zip
