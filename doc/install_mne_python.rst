.. include:: links.inc

.. _install_python_and_mne_python:

Install Python and MNE-Python
-----------------------------

.. contents:: Steps
   :local:
   :depth: 2

1. Install a Python interpreter
###############################

* For a fast and up to date scientific Python environment, **we recommend the
  Anaconda Python 2.7 distribution**. Get it for Windows, OSX, and Linux
  `here <http://docs.continuum.io/anaconda/install>`_.

  .. note :: Python has two major versions currently available, 2.7+ and 3.3+.
             Currently 3D visualization is only officially supported on 2.7.

* Once everything is set up, check the installation:

  .. code-block:: console

      $ conda --version
      conda 4.2.14
      $ python --version
      Python 2.7.12 :: Continuum Analytics, Inc.

  If your installation doesn't look something like this, **something went wrong**.
  Try looking through the Anaconda documentation or Googling for Anaconda install
  tips (StackExchange results are often helpful).

2. Install dependencies and MNE
###############################

* From the command line, install the MNE dependencies to the root Anaconda environment:

  .. raw:: html

      <div class="row container">
        <div class="col-sm-7 container">

  .. code-block:: console

      $ conda install scipy matplotlib scikit-learn mayavi jupyter spyder
      $ pip install PySurfer mne

  .. raw:: html

         </div>
         <div class="col-sm-4 container">
          <div class="panel panel-success">
            <div class="panel-heading"><h1 class="panel-title"><a data-toggle="collapse" href="#collapse_conda"><strong>Experimental</strong> Python 3.6 alternative ▼</a></h1></div>
            <div id="collapse_conda" class="panel-body panel-collapse collapse">
              <p>Try the conda environment available
              <a class="reference external" href="https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml">here</a>:
              </p>
              <div class="highlight-console">
                <div class="highlight">
                  <pre><span></span><span class="gp">$</span> curl -O https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml<br><span class="gp">$</span> conda env create -f environment.yml<br><span class="gp">$</span> source activate mne</pre>
                </div>
              </div>
              <p>If Mayavi plotting in Jupyter Notebooks doesn't work well, using the IPython magic "%gui qt" after importing MNE/Mayavi/PySurfer may 
              <a class="reference external" href="https://github.com/ipython/ipython/issues/10384">help</a>.
              </p>
            </div>
          </div>
        </div>
      </div>

* To check that everything went fine, in Python, type::

      >>> import mne

  If you get a new prompt with no error messages, you should be good to go!

* For advanced topics like how to get NVIDIA :ref:`CUDA` support or if you're
  having trouble, visit :ref:`advanced_setup`.
