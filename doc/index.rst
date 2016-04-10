.. title:: MNE

.. raw:: html

    <div class="container"><div class="row">
    <div class="col-md-8"><div style="text-align: center; height: 270px">
    <span style="display: inline-block; height: 100%; vertical-align: middle"></span>
    <a href="index.html"><img src="_static/mne_logo.png" border="0" alt="MNE" style="vertical-align: middle"></a>
    </div></div>
    <div class="col-md-4"><div style="float: left">
    <a href="index.html"><img src="_static/institutions.png"" border="0" alt="Institutions"/></a>
    </div></div>
    </div></div>

.. raw:: html

   <div class="container-fluid">
   <div class="row">
   <div class="col-md-8">
   <br>

MNE is a community-driven software package designed for **processing
electroencephalography (EEG) and magnetoencephalography (MEG) data**
providing comprehensive tools and workflows for
(:ref:`among other things <what_can_you_do>`):

1. Preprocessing and denoising
2. Source estimation
3. Time–frequency analysis
4. Statistical testing
5. Estimation of functional connectivity
6. Applying machine learning algorithms
7. Visualization of sensor- and source-space data

MNE includes a comprehensive `Python <https://www.python.org/>`_ package
supplemented by tools compiled from C code for the LINUX and Mac OSX
operating systems, as well as a MATLAB toolbox.

**From raw data to source estimates in about 30 lines of code** (:ref:`try it yourself! <getting_started>`):

.. code:: python

    >>> import mne  # doctest: +SKIP
    >>> raw = mne.io.read_raw_fif('raw.fif', preload=True)  # load data  # doctest: +SKIP
    >>> raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels  # doctest: +SKIP
    >>> raw.filter(l_freq=None, h_freq=40.0)  # low-pass filter data  # doctest: +SKIP
    >>> # Extract epochs and save them:
    >>> picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True,  # doctest: +SKIP
    >>>                        exclude='bads')  # doctest: +SKIP
    >>> events = mne.find_events(raw)  # doctest: +SKIP
    >>> reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)  # doctest: +SKIP
    >>> epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,  # doctest: +SKIP
    >>>                     proj=True, picks=picks, baseline=(None, 0),  # doctest: +SKIP
    >>>                     preload=True, reject=reject)  # doctest: +SKIP
    >>> # Compute evoked response and noise covariance
    >>> evoked = epochs.average()  # doctest: +SKIP
    >>> cov = mne.compute_covariance(epochs, tmax=0)  # doctest: +SKIP
    >>> evoked.plot()  # plot evoked  # doctest: +SKIP
    >>> # Compute inverse operator:
    >>> fwd_fname = 'sample_audvis−meg−eeg−oct−6−fwd.fif'  # doctest: +SKIP
    >>> fwd = mne.read_forward_solution(fwd_fname, surf_ori=True)  # doctest: +SKIP
    >>> inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd,  # doctest: +SKIP
    >>>                                              cov, loose=0.2)  # doctest: +SKIP
    >>> # Compute inverse solution:
    >>> stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=1./9.,  # doctest: +SKIP
    >>>                                      method='dSPM')  # doctest: +SKIP
    >>> # Morph it to average brain for group study and plot it
    >>> stc_avg = mne.morph_data('sample', 'fsaverage', stc, 5, smooth=5)  # doctest: +SKIP
    >>> stc_avg.plot()  # doctest: +SKIP

MNE development is driven by :ref:`extensive contributions from the community <whats_new>`.
Direct financial support for the project has been provided by:

- (US) National Institute of Biomedical Imaging and Bioengineering (NIBIB)
  grants 5R01EB009048 and P41EB015896 (Center for Functional Neuroimaging
  Technologies)
- (US) NSF awards 0958669 and 1042134.
- (US) NCRR *Center for Functional Neuroimaging Technologies* P41RR14075-06
- (US) NIH grants 1R01EB009048-01, R01 EB006385-A101, 1R01 HD40712-A1, 1R01
  NS44319-01, and 2R01 NS37462-05
- (US) Department of Energy Award Number DE-FG02-99ER62764 to The MIND
  Institute.
- (FR) IDEX Paris-Saclay, ANR-11-IDEX-0003-02, via the
  `Center for Data Science <http://www.datascience-paris-saclay.fr/>`_.

.. raw:: html

   <div class="col-md-8">
       <script type="text/javascript" src="http://www.ohloh.net/p/586838/widgets/project_basic_stats.js"></script>
   </div>


.. raw:: html

   </div>
   <div class="col-md-4">
   <h2>Documentation</h2>

.. toctree::
   :maxdepth: 1

   getting_started
   tutorials
   auto_examples/index
   faq
   contributing

.. toctree::
   :maxdepth: 1

   python_reference
   manual/index
   whats_new

.. toctree::
   :maxdepth: 1

   cite
   references
   cited

.. raw:: html

   <h2>Community</h2>

* `Analysis talk: join the MNE mailing list <MNE mailing list>`_

* `Feature requests and bug reports on GitHub <https://github.com/mne-tools/mne-python/issues/>`_

* `Chat with developers on Gitter <https://gitter.im/mne-tools/mne-python>`_

.. raw:: html

   <h2>Versions</h2>

   <ul>
      <li><a href=http://martinos.org/mne/stable>Stable</a></li>
      <li><a href=http://martinos.org/mne/dev>Development</a></li>
   </ul>

   <div style="float: left; padding: 10px; width: 100%;">
       <a class="twitter-timeline" href="https://twitter.com/mne_python" data-widget-id="317730454184804352">Tweets by @mne_python</a>
   </div>

   </div>
   </div>
   </div>
