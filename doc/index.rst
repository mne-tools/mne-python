.. title:: MNE

.. include:: links.inc

.. container:: row

  .. container:: col-md-12 nopad

    .. container:: midcenter nopad

      .. image:: _static/institutions.png
         :alt: Institutions


  .. container:: col-md-12 nopad

    .. container:: midcenter nopad tall

      .. image:: _static/mne_logo.png
         :alt: MNE


.. container:: row

  .. container:: col-md-12

    Community-driven software for **processing time-resolved neural
    signals including electroencephalography (EEG) and
    magnetoencephalography (MEG)**, offering comprehensive data analysis
    tools for Windows, OSX, and Linux:

    - Preprocessing and denoising
    - Source estimation
    - Time–frequency analysis
    - Statistical testing
    - Functional connectivity
    - Machine learning
    - Visualization of sensor- and source-space data

    Be sure to check out :ref:`what's new <whats_new>` with the package!

    From raw data to source estimates **in about 20 lines of code** (try it `in an experimental online demo <http://mybinder.org/repo/mne-tools/mne-binder/notebooks/plot_introduction.ipynb>`_!)::

        >>> import mne  # doctest: +SKIP
        >>> raw = mne.io.read_raw_fif('raw.fif')  # load data  # doctest: +SKIP
        >>> raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels  # doctest: +SKIP
        >>> raw.filter(l_freq=None, h_freq=40.0)  # low-pass filter  # doctest: +SKIP
        >>> events = mne.find_events(raw, 'STI014')  # extract events and epoch data # doctest: +SKIP
        >>> epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,  # doctest: +SKIP
        >>>                     reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))  # doctest: +SKIP
        >>> evoked = epochs.average()  # compute evoked  # doctest: +SKIP
        >>> evoked.plot()  # butterfly plot the evoked data # doctest: +SKIP
        >>> cov = mne.compute_covariance(epochs, tmax=0, method='shrunk')  # doctest: +SKIP
        >>> fwd = mne.read_forward_solution(fwd_fname, surf_ori=True)  # doctest: +SKIP
        >>> inv = mne.minimum_norm.make_inverse_operator(  # doctest: +SKIP
        >>>     raw.info, fwd, cov, loose=0.2)  # compute inverse operator # doctest: +SKIP
        >>> stc = mne.minimum_norm.apply_inverse(  # doctest: +SKIP
        >>>     evoked, inv, lambda2=1. / 9., method='dSPM')  # apply it # doctest: +SKIP
        >>> stc_fs = stc.morph('fsaverage')  # morph to fsaverage # doctest: +SKIP
        >>> stc_fs.plot()  # plot source data on fsaverage's brain # doctest: +SKIP

    Direct financial support for MNE has been provided by the United States:

    - NIH National Institute of Biomedical Imaging and Bioengineering
      *5R01EB009048* and *P41EB015896* (Center for Functional Neuroimaging
      Technologies)
    - NSF awards *0958669* and *1042134*.
    - NCRR *P41RR14075-06* (Center for Functional Neuroimaging Technologies)
    - NIH *1R01EB009048-01*, *R01EB006385-A101*, *1R01HD40712-A1*,
      *1R01NS44319-01*, *2R01NS37462-05*
    - Department of Energy Award Number *DE-FG02-99ER62764* (The MIND
      Institute)
    - Amazon Web Services - *Research Grant* issued to Denis A. Engemann

    And France:

    - IDEX Paris-Saclay, *ANR-11-IDEX-0003-02*, via the
      `Center for Data Science <http://www.datascience-paris-saclay.fr/>`_.
    - European Research Council Starting Grant *ERC-YStG-263584* and
      *ERC-YStG-676943*
    - French National Research Agency *ANR-14-NEUC-0002-01*.

.. container:: row

  .. container:: col-md-8 nopad

    .. raw:: html

      <a class="twitter-timeline" href="https://twitter.com/mne_python" data-widget-id="317730454184804352">Updates by @mne_python</a>

  .. container:: col-md-4 nopad

    .. raw:: html

      <script type="text/javascript" src="https://www.ohloh.net/p/586838/widgets/project_basic_stats.js"></script>
