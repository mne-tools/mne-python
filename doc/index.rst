MEG & EEG data analysis package
-------------------------------

.. raw:: html

   <div class="container-fluid">
   <div class="row">
   <div class="col-md-8">
   <br>

MNE is a software package for processing magnetoencephalography
(MEG) and electroencephalography (EEG) data that
provides comprehensive analysis tools and workflows including preprocessing,
source estimation, time–frequency analysis, statistical analysis, and several
methods to estimate functional connectivity between distributed brain regions.

MNE includes tools compiled from C code for the LINUX and Mac OSX
operating systems, as well as a matlab toolbox and a comprehensive
Python package (provided under the simplified BSD license).

.. raw:: html

   <h4>From raw data to dSPM source estimates in 30 lines of code:</h4>

>>> import mne
>>> raw = mne.io.Raw('raw.fif', preload=True)  # load data
>>> raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels
>>> raw.filter(l_freq=None, h_freq=40.0)  # low-pass filter data
>>> # Extract epochs and save them:
>>> picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True,
>>>                        exclude='bads')
>>> events = mne.find_events(raw)
>>> reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
>>> epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,
>>>                     proj=True, picks=picks, baseline=(None, 0),
>>>                     preload=True, reject=reject)
>>> # Compute evoked response and noise covariance
>>> evoked = epochs.average()
>>> cov = mne.compute_covariance(epochs, tmax=0)
>>> evoked.plot()  # plot evoked
>>> # Compute inverse operator:
>>> fwd_fname = 'sample audvis−meg−eeg−oct−6−fwd.fif'
>>> fwd = mne.read forward solution(fwd fname, surf ori=True)
>>> inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd,
>>>                                              cov, loose=0.2)
>>> # Compute inverse solution:
>>> stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=1./9.,
>>>                                      method='dSPM')
>>> # Morph it to average brain for group study and plot it
>>> stc_avg = mne.morph_data('sample', 'fsaverage', stc, 5, smooth=5)
>>> stc_avg.plot()

The MNE development is supported by National Institute of Biomedical Imaging and Bioengineering 
grants 5R01EB009048 and P41EB015896 (Center for Functional Neuroimaging Technologies) as well as 
NSF awards 0958669 and 1042134.

.. raw:: html

   </div>
   <div class="col-md-4">
   <h2>Documentation</h2>

.. toctree::
   :maxdepth: 1

   getting_started.rst
   tutorials
   examples
   manual
   faq.rst
   whats_new.rst
   python_reference.rst
   advanced_setup.rst
   mne-cpp
   cite
   :ref:`ch_reading`

.. raw:: html

   <h2>Community</h2>

* Join the MNE `mailing list <http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis>`_
* `Help/Feature Request/Bug Report <mailto:mne_support@nmr.mgh.harvard.edu>`_
* :ref:`contributing`

.. raw:: html

   <h2>Versions</h2>

   <ul>
      <li><a href=http://martinos.org/mne/stable>Stable</a></li>
      <li><a href=http://martinos.org/mne/dev>Development</a></li>
   </ul>

.. raw:: html

   </div>
   </div>
   </div>

.. raw:: html

    <div>
    <div style="width: 40%; float: left; padding: 20px;">
        <a class="twitter-timeline" href="https://twitter.com/mne_python" data-widget-id="317730454184804352">Tweets by @mne_python</a>
    </div>
    <div style="width: 40%; float: left; padding: 20px;">
        <script type="text/javascript" src="http://www.ohloh.net/p/586838/widgets/project_basic_stats.js"></script>
    </div>
    </div>
