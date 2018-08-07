---
name: Bug report
about: Create a report to help us improve

---
<!--
Further instructions on how to fill a bug refer here:
https://martinos.org/mne/stable/faq.html#i-think-i-found-a-bug-what-do-i-do

If your issue is a usage question, submit it here instead:
- The mne-python gitter: https://gitter.im/mne-tools/mne-python
- The mne-analysis mailing list: https://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis
-->

#### Describe the bug
<!--
Please provide a clear and concise description of what the bug is.
-->

#### Steps/Code to Reproduce
<!--
Please provide a code snippet or Minimal Working Example (MWE) to replicate your
problem. https://en.wikipedia.org/wiki/Minimal_Working_Example

This MWE should be auto conained. Other mne-python contributors should be able
to copy paste the provided snippet and replicate your results. In this regard,
it is prefered that the MWE use MNE sample data. Otherwise, provide all the data
needed to replicate.

If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com

Example:
```
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = ["Help I have a bug" for i in range(1000)]

vectorizer = CountVectorizer(input=docs, analyzer='word')
lda_features = vectorizer.fit_transform(docs)

lda_model = LatentDirichletAllocation(
    n_topics=10,
    learning_method='online',
    evaluate_every=10,
    n_jobs=4,
)
model = lda_model.fit(lda_features)
```
-->

#### Expected Results
<!-- 
Provide a clear and concise description of what you expected results.

Example: No error is thrown.
-->

#### Actual Results
<!-- 
Please paste or specifically describe the actual output or traceback. 
Include screenshots if necessary.

-->

#### Versions (``mne.sys_info()`` output)
<details>
<!--
[paste the output of ``mne.sys_info()`` here below this line]
-->

</details>


<!-- Thanks for contributing! -->
