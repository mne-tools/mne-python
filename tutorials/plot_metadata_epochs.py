"""
Pandas querying and metadata with Epochs objects
------------------------------------------------

Demonstrating pandas-style string querying with Epochs metadata.

Sometimes you've got a more complex trials structure that cannot be easily
summarized as a set of unique integers. In this case, it may be useful to use
the ``metadata`` attribute of ``Epochs`` objects. This must be a pandas
``DataFrame`` where each row is an epoch, and each column corresponds to a
metadata attribute of each epoch. Columns must be either strings, ints, or
floats.

First we'll load some data - subjects were presented with individual words
on a screen, and information about each word (e.g., word frequency) was
collected in a metadata object.

Loading the data
================
First we'll load the data...this is unnecessarily complex right now because
the data is stored as a dataframe...we should fix this :-)
"""
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves.urllib.request import urlopen

# Load the data from the interwebz (XXX need to fix this)
varname = 'https://www.dropbox.com/s/5y2rv7vlgilh52y/KWORD_VARIABLES_DGMH2015.txt?dl=1'
dataname = 'https://www.dropbox.com/s/6mpunoswlxaa9bi/KWORD_ERP_LEXICAL_DECISION_DGMH2015.txt?dl=1'

with urlopen(varname) as u:
    meta = pd.read_csv(u, delim_whitespace=True)

with urlopen(dataname) as u:  # warning: this is a 80MB download
    df = pd.read_csv(u, delim_whitespace=True, index_col=1).drop(["WORD#", "ELEC#"], axis=1)

# First we'll munge the data so it's in an Epochs structure
electrodes = df["ELECNAME"].unique()
words = df.index.unique()

mfile = mne.__file__.replace('__init__.py', 'channels/data/montages/standard_1005.elc')
channel_types = ['misc' if "REJ" in ch else 'eeg' for ch in electrodes]
df = df.drop(["ELECNAME"], axis=1)

times = [int(x[:-2]) for x in df.columns]
tmin, *_, tmax = np.array(times) / 1000
sfreq = int(len(times) / (tmax - tmin))

# create info
info = mne.create_info(electrodes.tolist(), sfreq, channel_types, mfile)

# reshape data
data = np.asarray([df.loc[word].values.astype(float) for word in words])

# events and event_id are just a running index per word ...
events = np.repeat(np.arange(data.shape[0]).reshape(-1, 1) + 1, 3, axis=1)
event_id = {word:ii + 1 for ii, word in enumerate(words)}

# Prepare metadata
meta["freq_rank"] = (meta["WordFrequency"].rank(method="dense") // 300 + 1)
meta["is_concrete"] = np.where(meta["WordFrequency"] > meta["WordFrequency"].median(), 'Concrete', 'Abstract')

# Construct the EpochsArray
epochs = mne.EpochsArray(data / 1e6, info, events, tmin=tmin, event_id=event_id, metadata=meta)
epochs.metadata.head()

################################################################################
#

epochs.average().plot_joint(title="Grand Average (75 subjects, {} words)".format(len(meta)), show=False)


################################################################################

names = "Concreteness WordFrequency NumberOfLetters".split()
reg = epochs.regress(names)
for nm, coefs in reg.items():
    coefs.plot_joint(title=nm, show=False)

################################################################################

names = "Concreteness WordFrequency NumberOfLetters".split()
reg = epochs.regress(names, by='is_concrete')

for kind in meta['is_concrete'].unique():
    reg[kind]['Concreteness'].plot_joint(title='Concreteness: ' + str(kind), show=False)
################################################################################

av1 = epochs['Concreteness < 5 and WordFrequency < 2'].average()
av2 = epochs['Concreteness > 5 and WordFrequency > 2'].average()

av1.plot_joint(show=False)
av2.plot_joint(show=False)

################################################################################
words = ['film', 'cent', 'shot', 'cold', 'main']
epochs['WORD in {}'.format(words)].plot_image(show=False)

################################################################################

categories = ["NumberOfLetters", "is_concrete"]
avs = epochs.average(by=categories)

colors = np.linspace(0, 1, num=len(avs))

style_plot = dict(
    colors=plt.cm.viridis(colors),
    linestyles={'Concrete': '-', 'Abstract': '--'}
)
fig, ax = plt.subplots()
mne.viz.evoked.plot_compare_evokeds(
    avs, **style_plot, picks=list(avs.values())[0].ch_names.index("Pz"), show=False, axes=ax
).set_size_inches((6, 3))
ax.legend(loc=[1.05, .1])

plt.show()
