.. include:: ../git_links.inc

MNE quickstart and background
=============================

1. Reading data into MNE
------------------------
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

They all have in common to return an :class:`mne.io.Raw`-like object.
See :ref:`ch_convert`.

2. MNE gives you objects with methods
-------------------------------------
We said above that there are MNE objects. This is of course computer
science jargon. What it actually means is that you get a data structure
that is more than the channels by time series
and the information about channel types and locations, meta-data if
you want. Indeed the structures that MNE is using provide so called
methods. These are nothing but functions that are configured to take
the data and the meta-data of the object as parameters. Sounds
complicated, but it's actually simplifying your life as you will see
below. Whether you consider Raw objects that describe continuous data,
Epochs objects describing segmented single trial data, or Evoked objects
describing averaged data, all have in common that they share certain methods.

- Try :meth:`raw.plot <mne.io.Raw.plot>`,
  :meth:`epochs.plot <mne.Epochs.plot>`,
  :meth:`evoked.plot <mne.Evoked.plot>` and any other method that has
  a name that starts with `plot`. By using the call operators `()`
  you invoke these methods, e.g.
  :meth:`epochs.plot() <mne.Epochs.plot>`.
  Yes, you don't have to pass arguments but you will get an informative
  visualization of your data. The method knows what to do with the object.
  Look up the documentation for configuration options.

- Try :func:`raw.pick_types <mne.io.Raw.pick_types>`,
  :func:`epochs.pick_types <mne.Epochs.pick_types>`
  :func:`evoked.pick_types <mne.Evoked.pick_types>`
  and any other method that has a name that starts with `pick`. These
  methods will allow you to select channels either by name or by type.
  Picking is MNE jargon and stands for channel selection.

- Some of these methods can actually change the state of the object,
  e.g. permanently remove or transform data. To preserve your input
  data can explicitly use the .copy method to manipulate a copy of
  your inputs. Example::

    >>> raw.copy().pick_types(meg=False, eeg=True)  # doctest: +SKIP

- This examplifies another important concept, that is chaining. Most
  methods return the object and hence allow you to write handy pipelines.
  Guess what this code does::

    >>> (fig = raw.copy()  # doctest: +SKIP
    >>>           .pick_types(meg=False, eeg=True)  # doctest: +SKIP
    >>>           .resample(sfreq=100)  # doctest: +SKIP
    >>>           .filter(1, 30)  # doctest: +SKIP
    >>>           .plot())  # doctest: +SKIP

  Yes, it creates a figure after filtering a resampled copy of the EEG
  data. In fact you can also recognize methods by certain linguistic
  cues. Methods typically use english verbs. So `raw.ch_names` is
  not a method. It's just an attribute that cannot be invoked like
  a function.

- Last but not least, many MNE objects returned a `.save` method that
  allows you to store your data into a FIFF file.


3. A key thing for MNE objects is the measurement info
------------------------------------------------------
Besides ``.ch_names`` another important attribute is ``.info``. It contains
the channel information and some details about the processing history.
This is especially relevant if your data cannot be read using the io
functions listed above. You then need to learn how to create an info.
See :ref:`tut_info_objects`.

4. MNE is modular
-----------------
Beyond methods another concept that is important to get are *modules*.
Think of them as name spaces, another computer science term.
Ok, think of street names in different cities. Sending a parcel to the
Washington street in New York or San Francisco typically
does not involve a conflict, as these streets are in different cities.
Now you know what is the idea behind a name space. You can
read a lot of resources that you will find when googling accordingly.
What is important here is that our modules are organized by
processing contexts. Looking for I/O operations for raw data?::

    >>> from mne import io

Wanna do preprocessing?::

    >>> from mne import preprocessing

Wanna do visualization?::

    >>> from mne import viz

Decoding?::

    >>> from mne import decoding

I'm sure you got it, so explore your intuitions when searching for
a certain function.

5. Inspect and script
---------------------
Did you happen to notice that some of the figures returned by ``.plot``
methods allow you to interact with the data? Look at :meth:`raw.plot <mne.io.Raw.plot>` and
:meth:`epochs.plot <mne.Epochs.plot>` for example. They allow you to update channel selections,
scalings and time ranges. However, they do not replace scripting.
The MNE philosophy is to facilitate diagnostic plotting but does
not support doing analysis by clicking your way. MNE is meant to be
a toolbox, and its your task to combine the tools by **writing scripts**.
This should save you time in the long run by:

1. Enabling code reuse.
2. Documenting what you did.

Reviewers are asking you to update your analysis that you actually
finished 1 year ago? Luckily you have a script.

6. Eighty percent or Python
---------------------------
A related point is that MNE functions are there to make it fun to
process common tasks and facilitate doing difficult things.
This means that you will notice certain limits
here and there, the viz functions do not exactly plot things as
you want them, even when using the options provided by that function.
In fact our goal is to guess which are the essential 80 percent that
you need in order be happy in 80 percent of the time. Where you need
more Python is there for you. You can easily access the data, e.g.
`raw[:10, :1000]` or `epochs.get_data()` or `evoked.data` and
manipulate them using numpy or pass them to high-level machine learning code
from `scikit-learn`_. Each ``.plot`` method
returns a matplotlib figure object. Both packages have great documentations
and often writing Python code amounts to looking up the right library that
allows you to tackle the problem in a few lines.
