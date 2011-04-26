.. _getting_started:

Getting Started at the Martinos Center
======================================

You first need to activate the python scientific environment.
In a terminal do:

    .. source /usr/pubsw/packages/python/2.6/scientificpython/bin/tcsh_activate
    setenv PATH ${PATH}:/usr/pubsw/packages/python/epd/bin

If you use Bash replace the previous instruction with:

    .. source /usr/pubsw/packages/python/2.6/scientificpython/bin/bash_activate
    export PATH=${PATH}:/usr/pubsw/packages/python/epd/bin

Then start the python interpreter with:

    ipython

Although all of the examples in this documentation are in the style
of the standard Python interpreter, the use of IPython is highly
recommended. Then type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go.
Start with the `examples <auto_examples/index.html>`_

