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
and matplotlib >= 1.1.0.

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
recommended.

Now that you have a working Python environment you can install MNE.

You can manually get the latest version of the code at:

https://github.com/mne-tools/mne-python

Then from the mne-python folder (containing a setup.py file) you can install with::

    python setup.py install

or if you don't have admin access to your python setup (permission denied when install) use::

    python setup.py install --user

You can also install the latest release with easy_install::

    easy_install -U mne

or with pip::

    pip install mne --upgrade

For the latest development version (the most up to date)::

    pip install -e git+https://github.com/mne-tools/mne-python#egg=mne-dev

To check that everything went fine, in ipython, type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go.

If you want to use NVIDIA CUDA for filtering (can yield 3-4x speedups), you'll
need to install the NVIDIA toolkit on your system, and then both pycuda and
scikits.cuda, see:

https://developer.nvidia.com/cuda-downloads
http://mathema.tician.de/software/pycuda
http://wiki.tiker.net/PyCuda/Installation/
https://github.com/lebedov/scikits.cuda

To initialize mne-python cuda support, after installing these dependencies
and running their associated unit tests (to ensure your installation is correct)
you can run:

    >>> mne.cuda.init_cuda() # doctest: +SKIP

If you have everything installed correctly, you should see an INFO-level log
message telling you your CUDA hardware's available memory. To have CUDA
initialized on startup, you can do:

    >>> mne.utils.set_config('MNE_USE_CUDA', 'true') # doctest: +SKIP

You can test if MNE CUDA support is working by running the associated test:

    nosetests mne/tests/test_filter.py

If all tests pass with none skipped, then mne-python CUDA support works.

Learning Python
---------------

If you are new to Python here is a very good place to get started:

    * http://scipy-lectures.github.com
