---
name: Bug report
about: Create a report to help us improve

---
Detailed instructions on how to file a bug:
https://martinos.org/mne/stable/faq.html#i-think-i-found-a-bug-what-do-i-do

If your issue is a usage question, submit it here instead:
- The MNE-Python gitter: https://gitter.im/mne-tools/mne-python
- The MNE-Python mailing list: https://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis

#### Describe the bug
Please provide a clear and concise description of what the bug is.

#### Steps and/or code to reproduce
Please provide a code snippet or [Minimal Working Example (MWE)](https://en.wikipedia.org/wiki/Minimal_Working_Example)
to replicate your problem. 

This MWE should be self contained, MNE-Python. Other MNE-Python contributors
should be able to copy and paste the provided snippet and replicate your
results. 
When possible use MNE-Python testing examples to reproduce the errors. Otherwise,
provide an anonymous version of the data in order to replicate the errors.

If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com

Example:

```py
import mne
import numpy as np
from mne.preprocessing import ICA, create_ecg_epochs
from mne.datasets import sample

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.pick_types(meg=True, eeg=False, exclude='bads', stim=True)
raw.filter(1, 30, fir_design='firwin')

events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.2, tmax=0.5, preload=True)

ica = ICA(n_components=0.95, method='fastica').fit(epochs)

ica.exclude = np.array([0, 1, 2])

_ = ica.apply(epochs) # This Breaks
```

#### Expected results
Provide a clear and concise description of what you expected to happen.

Example: 

ica is applied to epochs

#### Actual results
Please paste or specifically describe the actual output or traceback. 

Example:

A `ValueError` is raised
```py
Traceback (most recent call last):
  File "<ipython-input-9-8bd3ad3cbd93>", line 1, in <module>
    _ = ica.apply(epochs)
  File "~/miniconda3/envs/mne-pip/lib/python3.6/site-packages/mne/preprocessing/ica.py", line 1252, in apply
    n_pca_components=n_pca_components)
  File "~/miniconda3/envs/mne-pip/lib/python3.6/site-packages/mne/preprocessing/ica.py", line 1309, in _apply_epochs
    data = self._pick_sources(data, include=include, exclude=exclude)
  File "~/miniconda3/envs/mne-pip/lib/python3.6/site-packages/mne/preprocessing/ica.py", line 1368, in _pick_sources
    elif exclude not in (None, []):
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

#### Additional information
<details>
paste the output of `mne.sys_info()` here below this line

</details>
