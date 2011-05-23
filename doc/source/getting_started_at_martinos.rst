.. _getting_started_martinos:

Getting Started at the Martinos Center
======================================

You first need to activate the python scientific environment.
In a terminal do::

    setenv PATH /usr/pubsw/packages/python/epd/bin:${PATH}

.. source /usr/pubsw/packages/python/2.6/scientificpython/bin/tcsh_activate

If you use Bash replace the previous instruction with::

    export PATH=/usr/pubsw/packages/python/epd/bin:${PATH}

.. source /usr/pubsw/packages/python/2.6/scientificpython/bin/bash_activate

Then start the python interpreter with:

    ipython

Although all of the examples in this documentation are in the style
of the standard Python interpreter, the use of IPython is highly
recommended. Then type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go.
Start with the `examples <auto_examples/index.html>`_

