Ten facts about MNE
====================


1. What the FIF does MNE stand for?
-----------------------------------
Historically, MNE was a software for computing cortically constrained
Minimum Norm Estimates from MEG and EEG data. The historical core
functions of MNE were written by Matti Hämäläinen in Boston and originate
in part from the Elekta software that is shipped with its MEG systems.
Ah yes, the FIFF is Elektas Functional Imaging File Format that goes
along with `.fif` file extensions and is natively used by its MEG systems.
For these reasons the MNE software is internally relying on the FIFF files.
Today the situation is a bit different though. MNE is nowadays developed
mostly in Python by an international team of researchers from diverse
laboratories and has widened its scope. MNE supports advanced sensor space
analyses for EEG, temporal ICA, many different file formats and many other
inverse solvers, for example beamformers. Some of our contributors even
use it for intracranial data. If you want, MNE can be thought of as MEG'n'EEG.

2. Reading data into the MNE layout
-----------------------------------
One of the first things you might be wondering about is how to get your
data into mne. Assuming that you have unprocessed data, you will probably
be happy with at least one of these readers:

* :func:`read_raw_fif <mne.io.read_raw_fif>`
* :func:`read_raw_kit <mne.io.read_raw_kit>`
* :func:`read_raw_bti <mne.io.read_raw_bti>`
* :func:`read_raw_ctf <mne.io.read_raw_ctf>`
* :func:`read_raw_brainvision <mne.io.read_raw_brainvision>`
* :func:`read_raw_cnt <mne.io.read_raw_cnt>`
* :func:`read_raw_edf <mne.io.read_raw_edf>`
* :func:`read_raw_eeglab <mne.io.read_raw_eeglab>`
* :func:`read_raw_egi <mne.io.read_raw_egi>`
* :func:`read_raw_nicolet <mne.io.read_raw_nicolet>`

They all have in common to return an mne.io.Raw object and the MEG
readers perform conversions of sensor positions and channel names
to make the meta data compatible with the conventions of the FIFF
format. Yes, at this point MNE relies on the historical layout and
therefore expects MEG data to look like Elekta Neuromag data and to
conform to Freesurfer data layouts. This is somewhat relaxed for EEG
data, which have less to do with Neuromag and very often are not
used for source space analyses.

3. objects and methods
----------------------
We said above that there are MNE objects. This is of course computer
science jargon. What it actually means is that you get a data structure
that not only contains the actual channels by timepoints time series
and the information about channel types and locations, meta-data if
you want. Indeed the structures that MNE is using provide so called
methods. These are nothing but functions that are configured to take
the data and the meta-data of the object as parameters. Sounds
complicated, but it's actually simplifying your life as you will see
below. Whether you consider Raw objects that describe continous data,
Epochs objects describing segmented single trial data, or Evoked objects
describing averaged data, all have in common that they share certain methods.

- Try raw.plot, epochs.plot, evoked.plot and any other method that has
  a name that starts with `plot`. By using the call operators `()`
  you invoke these methods, e.g. `epochs.plot()`.
  Yes, you don't have to pass arguments but you will get an informative
  visualization of your data. The method knows what to do wiht the object.
  Look up the documentation for configuration options.

- Try raw.pick_types, epochs.pick_tyoes, evoked.pick_types and any other
  method that has a name that starts with `pick`. These methods will
  allow you to select channels either by name or by type. Picking
  is MNE jargon and stands for channel selection.

- Some of these methods can actually change the state of the object,
  e.g. permanently remove or transform data. To preserve your input
  data can explicitly use the .copy method to manipulate a copy of
  your inputs. Example: raw.copy().pick_types(meg=False, eeg=True)

- This examplifies another important concept, that is chaining. Most
  methods return the object and hence allow you to write handy pipelines.
  Guess what this code does::

    >>> (fig = raw.copy()
    >>>           .pick_types(meg=False, eeg=True)
    >>>           .resample(sfreq=100)
    >>>           .filter(1, 30)
    >>>          .plot())

  Yes, it creates a figure after filtering a resampled copy of the EEG
  data. In fact you can also recognize methods by certain linguistic
  cues. Methods typically use english verbs. So `raw.ch_names` is
  not a method. It's just an attribute that cannot be invoked like
  a function.

- Last but not least, many MNE objects returned a .save method that
  allows you to store your data into a FIFF file.


7. Channel info and basic object design
---------------------------------------
Besides `.ch_names` another important attribute is .info. It contains
the channel information and some details about the processing history.
This is especially relevant if your data cannot be reas using the io
functions listed above. You then need to learn how to creat an info.

8. Modularity
-------------
Beyond methods another concept that is important to get are modules.
Think of them as name spaces, another computer science term.
Ok, think of street names in different cities. Sending a parcel to the
Washington street in New York or San Francisco typically
does not involve a conflict, as these streets are in different cities.
Now you know what is the idea behind a name space. You can
read a lot of resources that you will find when googling accordingly.
What is important here is that our modules are organized by
processing contexts. Looking for I/O operations? `from mne import io`.
Wanna do preprocessing? `from mne import preprocessing`.
Wanna do visualization? `from mne import viz`.
Decoding? `from mne import decoding`. I'm sure you got it,
so explore your intuitions when searching for a certain function.

9. User interfaces and scripting
--------------------------------
Did you happen to notice that some of the figures returned by `.plot`
methods allow you to interact with the data? Look at raw.plot and
epochs.plot for example. They allow you to update channel selections,
scalings and time ranges. However, they do not replace scripting.
The MNE philosophy is to facilitate diagnostic plotting but does
not support doing analysis by clicking your way. MNE is meant to be
a toolbox, and its your taks to combine the tools by writing scripts.
This should really save you time, first of all by being able to reuse
code and avoiding to click it again. Second by documenting what you
did. Reviewers are asking you to update your analysis that you actually
finished 1 year ago? Luckily you have a script.


10. Eighty percent or think Python
----------------------------------
A related point is that MNE functions are there to make it fun to
process common tasks and facilitate doing difficult things noone
but you knows about. This means that you will notice certain limits
here and there, the viz functions does not exactly plot things as
you want them, even when using the options provided by that function.
In fact our goal is to guess which are the essential 80 percent that
you need in order be happy in 80 percent of the time. Where you need
more Python is there for you. You can easily access the data, e.g.
`raw[:10, :1000]` or `eopchs.get_data()` or `evoked.data` and
manipulate them using numpy. Each .plot method returns a matplotlib
figure object. Both packages are exquisitly documented and often
writing Python code amounts to looking up the right library that
allows you to tackle the problem in a few lines.
