.. _getting_started:

Getting Started
===============

Inside the Martinos Center
--------------------------
For people within the MGH/MIT/HMS Martinos Center mne is available on the network.

In a terminal do::

    setenv PATH /usr/pubsw/packages/python/epd/bin:${PATH}

If you use Bash replace the previous instruction with::

    export PATH=/usr/pubsw/packages/python/epd/bin:${PATH}

Then start the python interpreter with:

    ipython

Then type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go.
Start with the :ref:`examples-index`.

Outside the Martinos Center
---------------------------

MNE is written in pure Python making it easy to setup of
any machine with Python >=2.6, Numpy >= 1.4, Scipy >= 0.7.2
and matplotlib >= 0.98.4.

Some isolated functions (e.g. filtering with firwin2 require Scipy >= 0.9).

For a fast and up to date scientific Python environment you
can install EPD available at:

http://www.enthought.com/products/epd.php

EPD is free for academic purposes. If you cannot benefit from the
an academic license and you don't want to pay for it, you can
use EPD free which is a lightweight version (no 3D visualization
support for example):

http://www.enthought.com/products/epd_free.php

To test that everything works properly, open up IPython::

    ipython

Although all of the examples in this documentation are in the style
of the standard Python interpreter, the use of IPython is highly
recommended.  Then type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go.

Learning Python
---------------

If you are new to Python here is a very good place to get started:

    * http://scipy-lectures.github.com
