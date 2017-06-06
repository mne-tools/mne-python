Caveats coming from EEGLAB
==========================

EEGLAB vs. MNE: Pros and Cons
-----------------------------

Major differences: MNE-Python ...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* is **Python based**. Python is a fully featured, general programming language. It is easy to learn, easy to read (and thus share, criticize and reproduce code with). It is also free - unlike MATLAB, which you need for EEGLAB, and the MATLAB toolboxes. For example, with MNE, you will never run out of/lack parallel processing toolboxes, and thus MNE more readily uses multiple cores (usually with the `n_jobs` keyword).
* was born as a package for **MEG analysis**. While *MNE has gained strong EEG support*, it in many instances takes a more extensive approach. For example, MEG data is usually multimodal (including magnetometers, gradiometers and EEG), and thus MNE requires you to constantly choose what channel type you want to operate with. MNE also has much more integrated support for source-level/source localisation than EEGLAB.
* is **Object oriented**. Typical user-exposed functions don't operate on raw data matrices, but on `Raw`, `Epochs`, `Evoked`, ... objects. Many object methods operate in-place. For example, when in MNE, you do EEG = pop_eegfiltnew(EEG, ...), in MNE, you do raw.filter(...). In EEGLAB, you do pop_plot_topo(EEG, ...). In MNE, you do evoked.plot_topomap(...). In EEGLAB, you do pop_saveset(EEG, ...). In MNE, you do `epochs.save(...)`. `raw.plot()` is very different from `evoked.plot()`. In-place modification can usually be turned off with the `copy=True` setting.
* does **not feature a GUI**. MNE always requires explicit code. This can be frustrating for beginners, but in the long run, it supports good scientific practices - better reproducible, more scalable analyses. Generally speaking, there is some trade-off between the lack of GUI in MNE and the Python base; while GUIs are more accessible to newcomers, Python more easily allows experienced users to create custom and advanced analysis pipelines and techniques.
* has a different *status and style* of the project. EEGLAB is the biggest EEG analysis package. MNE is a comparatively young project. It follows modern coding standards (including code review and unit tests). Its development is less monolithic, more distributed and democratic than that of EEGLAB. There are many published papers using EEGLAB, and very few with MNE-Python, and many users of EEGLAB and few of MNE.

Minor differences: MNE-Python ...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* is much more developed with regards to **MVPA/decoding analyses**. It is also developed with a close eye on the leading Machine Learning toolbox *scikit-learn*.
* 's *ICA is not as deeply integrated* into the system. Support for clustering ICs is very rudimentary. However, MNE's ICA implementations are fully featured, and in some ways superior to EEGLAB. For example, MNE's infomax implementation can be much faster than EEGLAB's, in large part due to multicore usage. Also, MNE ICA objects allow easily applying the same ICA fit to multiple data sets (e.g., it is easy to fit ICA on high pass filtered data and apply it to unfiltered data).
* does not feature *plugins* as strongly as EEGLAB does. No true equivalents exist for high quality plugins such as LIMO or ERPLAB. Some features which are plugins in EEGLAB are parts of base MNE, such as CORRMAP, and many plugins are at least partially mirrored by MNE core functionality (although ironically, not Alex Gramfort's graph cut plugin for single trial analysis!!!).
* does not have as broad *data system format support* for EEG data formats as EEGLAB does with `fileio`. However, MNE can read many common file formats, including the EEGLAB format, and of course many MEG formats.
* does not have the extensive inbuilt *single-trial* capabilities of EEGLAB. For example, while it is possible to plot single trials sorted by alpha power by feeding mne.viz.plot_epochs_image a function built around psd_epochs, it requires much more user input.
* does not easily allow the rich storage of epochs-level *event information* EEGLAB has. MNE events are stored in the three-column `events` format, all integers.
* does not have an equivalent to EEGLAB's *STUDY* structure. Grand averages of single-subject `evoked` and `average_tfr` objects can be constructed with the `mne.grand_average` function (beware of the default setting of weighting by trials), which return regular `evoked` and `average_tfr` objects. Individual subjects can simply be stored in lists or dicts of `epochs` or `raw` objects.

Partial EEGLAB/MNE translation table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================================================================================================================
                              Software
Concept                       EEGLAB          			        MNE
Importing data		`EEG = pop_fileio('/your/data.dat');`		raw = mne.io.read_raw('/your/data-raw.fif')

Filtering raw data	`EEG = pop_eegfiltnew(EEG, 1, 40);`			raw.filter(1, 40)

Run ICA				`EEG = pop_runica(EEG);`					ica = mne.preprocessing.ica.ica().fit(raw)


Epoching data		`EEG_epochs = pop_epoch(EEG,				`epochs = mne.Epochs(raw,
																events=mne.find_events(raw),
                              {'1', '2'},		                event_id=dict(cond1=1, cond2=2),
                              [-.2, .8]);`						tmin=-2, tmax=.8)`

Selecting epochs    `EEG_epochs2 = pop_epoch(EEG_epochs,		`epochs_2 = epochs["cond2"]`
                            {'2'});`



ERPs															`evoked = epochs.average()`

Plot Butterfly ERP	`pop_timtopo(EEG_epochs, ...);`             `evoked.jointplot()`

Contrast ERPs		`pop_compareerps(EEG_epochs, EEG_epochs2);`	`(evoked - evoked2).plot()`

Saving data         `EEG = pop_saveset(EEG,'filename',set);`	`[raw/epochs/evoked].save('/your/data-[raw/epo/evo].fif')

============================================================================================================================


Tips on specific features for people coming from EEGLAB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The MNE event system stores onset, duration, and type (as integers). Beware of the find_events function; with its default settings, it will not find many events in typical EEG formats. Try using `shortest_event=0, min_duration=0, consecutive=True` if that happens to you.
* The MNE resampling function is much slower than the EEGLAB equivalent. As a temporary work-around, many MNE functions allow a `decim` keyword to downsize analyses or objects. Also, remember that MNE can be more readily parallelized than MNE; via e.g. joblib, a whole experiment can be downsampled simultaneously.
* The MNE `eog` channel type is generally conceptualized around bipolar eye channels. Channels marked as `eog` are not included in many analyses. Typical EEGLAB behavior is roughly equivalent to setting all EEG channels, including monopolar eye sensors, to channel type `eeg`.
* MNE evoked objects support arithmetic; you can do `(epochs["critical"].average() - epochs["control"].average()).plot()`. However note that this is weighted by trial numbers! (Use `combine_evoked` to use uniform weights.) Probably the best way to plot multi-channel ERPs in MNE is `evoked.plot(spatial_colors=True)`.
* To recreate the ERPimages sorted by response time typical of EEGLAB, reaction times can be found with the ... command, and stored in the second column of an `events` matrix. The plot can then be created with with `mne.viz.plot_epochs_image(epochs, ... order=events[:,1].argsort())`.
* Beware that `mne.viz.plot_epochs_image` will by default plot all channels; use the `picks` kwarg, unless you want 128 figures popping up on your screen.
* no true equivalent to the summary-style plots from .....  exist.
* if you are annoyed by `mne.Epochs` giving too extensive feedback during e.g. baselining, control the logging verbosity with mne.set_log_level().
