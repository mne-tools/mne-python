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

This MWE should be self contained, MNE-Python. Other mne-python contributors
should be able to copy and paste the provided snippet and replicate your
results. In this regard, it is preferred that the MWE uses MNE sample data.
Otherwise, provide all the data needed to replicate.

If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com

Example:
```py
import numpy as np
import mne

m = mne.channels.read_montage("biosemi64")

info = mne.create_info(m.ch_names[:-3], sfreq=512, ch_types="eeg", montage=m)
np.random.seed(1)
values = np.random.randint(-100, 0, 64)
mne.viz.plot_topomap(values, info)
```
-->

#### Expected results
Provide a clear and concise description of what you expected to happen.

#### Actual results
Please paste or specifically describe the actual output or traceback. 

#### Additional information
<details>
paste the output of `mne.sys_info()` here below this line

</details>
