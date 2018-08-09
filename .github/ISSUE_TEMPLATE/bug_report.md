---
name: Bug report
about: Create a report to help us improve.

---

Detailed instructions on how to file a bug can be found in our [FAQ](https://martinos.org/mne/stable/faq.html#i-think-i-found-a-bug-what-do-i-do).

If your issue is a usage question, please consider asking on our [Gitter channel](https://gitter.im/mne-tools/mne-python) or on our [mailing list](https://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis) instead of opening an issue.


#### Describe the bug
Please provide a clear and concise description of the bug.


#### Steps and/or code to reproduce
Please provide a code snippet or [minimal working example (MWE)](https://en.wikipedia.org/wiki/Minimal_Working_Example)
to replicate your problem. 

This MWE should be self-contained, which means that other MNE-Python contributors
should be able to copy and paste the provided snippet and replicate the bug. 
If possible, use MNE-Python testing examples to reproduce the error. Otherwise,
provide a small and anonymized portion of your data required to reproduce the bug.

If the code is too long, feel free to put it in a [public gist](https://gist.github.com) and link
it in the issue.

Example:

```Python
import mne

fname = mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(fname)
raw = mne.concatenate_raws([raw])
raw.save('test_raw.fif', overwrite=True)
raw_read = mne.io.read_raw_fif('test_raw.fif')  # this breaks
```


#### Expected results
Provide a clear and concise description of what you expected to happen.


#### Actual results
Please paste or specifically describe the actual output or traceback. 


#### Additional information
Paste the output of `mne.sys_info()` here.
