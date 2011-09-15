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
any machine with Python >=2.5, Numpy >= 1.4 and Scipy >= 0.6.

For a free, fast and up to date scientific Python environment you
can install EPD Free available at:

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

If you are new to Python here are some online courses:

    * http://docs.python.org/

    * http://learnpythonthehardway.org/book/

    * http://code.google.com/intl/fr/edu/languages/google-python-class/

    * http://homepage.mac.com/s_lott/books/python.html

More specific tutorials on scientific computing with Python using Numpy and Scipy see:

    * http://showmedo.com/videotutorials/numpy (video)

    * http://www.scipy.org/NumPy_for_Matlab_Users

    * http://mathesaurus.sourceforge.net/matlab-numpy.html

Some discussions online are:

    * http://stackoverflow.com/questions/111857/what-did-you-use-to-teach-yourself-python

    * http://stackoverflow.com/questions/17988/how-to-learn-python
