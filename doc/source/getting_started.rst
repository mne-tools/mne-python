.. _getting_started:

Getting Started
===============

This page will help you get started with MNE-python. If you are new to Python here is a
very good place to get started: http://scipy-lectures.github.com. If you are at the Martinos 
Center, please see this section :ref:`inside_martinos`. If you would like to use a custom
installation of python (or have specific questions about integrating special tools like 
IPython notebooks), please see this section :ref:`detailed_notes`.

Outside the Martinos Center
---------------------------

For a fast and up to date scientific Python environment that resolves all
dependencies you can install Enthought Canopy available at:

https://www.enthought.com/products/canopy/

Canopy is free for academic purposes. If you cannot benefit from the
an academic license and you don't want to pay for it, you can
use Canopy express which is a lightweight version (no 3D visualization
support for example): https://www.enthought.com/store/.

To test that everything works properly, open up IPython::

    ipython --pylab qt

Now that you have a working Python environment you can install MNE.

The first decision you must make is whether you want the most recent stable version or the 
development version (this contains new features, however the function names and usage examples
may not be fully settled).

Stable Version Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also install the latest stable version with with pip::

    pip install mne --upgrade
    
Now that you have installed mne, check and optimize the installation (:ref:`check_and_optimize`)

Development Version Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you know you would like to contribute to the project please follow the instructions here: 
:ref:`using-git`

If you just want to start using the latest development version (the most up to date)::

    pip install -e git+https://github.com/mne-tools/mne-python#egg=mne-dev

.. _check_and_optimize:

Check and Optimize Installation
-------------------------------

To check that everything went fine, in ipython, type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go.

CUDA Optimization
^^^^^^^^^^^^^^^^^

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


.. _detailed_notes:

Detailed Notes
--------------

MNE is written in pure Python making it easy to setup on
any machine with Python >=2.6, NumPy >= 1.6, SciPy >= 0.7.2
and matplotlib >= 1.1.0.

Some isolated functions (e.g. filtering with firwin2) require SciPy >= 0.9.

To run all documentation examples the following additional packages are required:

    * PySurfer (for visualization of source estimates on cortical surfaces)

    * scikit-learn (for supervised and unsupervised machine learning functionality)

    * pandas >= 0.8 (for export to tabular data structures like excel files)

    * h5py (for reading and writing HDF5-formatted files)

Note. For optimal performance we recommend installing recent versions of
NumPy (> 1.7), SciPy (> 0.10) and scikit-learn (>= 0.14).

Development Environment
^^^^^^^^^^^^^^^^^^^^^^^

Note that we explicitly support the following Python setups since they reflect our
development environments and functionality is best tested for them:

    * EPD 7.3 (Mac, Linux)

    * Canopy >= 1.0 (Mac, Linux)

    * Anaconda (Mac)

    * Debian / Ubuntu standard system Python + Scipy stack

Anaconda
^^^^^^^^

Note for developers. To make Anaconda working with our test-suite a few
manual adjustments might be necessary. This may require
manually adjusting the python interpreter invoked by the nosetests and
the sphinx-build 'binaries' (http://goo.gl/Atqh26).
Tested on a recent MacBook Pro running Mac OS X 10.8 and Mac OS X 10.9

multi-threading
^^^^^^^^^^^^^^^

For optimal performance we recommend using numpy / scipy with the multi-threaded
ATLAS, gotoblas2, or intel MKL. For example, the Enthought Canopy and the Anaconda distributions
ship with tested MKL-compiled numpy / scipy versions. Depending on the use case and your system
this may speed up operations by a factor greater than 10.

pylab
^^^^^

Although all of the examples in this documentation are in the style
of the standard Python interpreter, the use of IPython with the pylab option
is highly recommended. In addition, for the setups listed above we would
strongly recommend to use the QT matplotlib backend for fast and correct rendering::

    ipython --pylab qt


On Linux, for example, QT is the only matplotlib backend for which 3D rendering
will work correctly. On Mac OS X for other backends certain matplotlib functions
might not work as expected.

IPython notebooks
^^^^^^^^^^^^^^^^^

To take full advantage of MNE-Python's visualization capacities in combination
with IPython notebooks and inline displaying, please explicitly add the
following magic method invocation to your notebook or configure your notebook
runtime accordingly.

    %pylab inline

If you use another Python setup and you encounter some difficulties please
report them on the MNE mailing list or on github to get assistance.


.. _inside_martinos:

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
