<!--
If your issue is a usage question, submit it here instead:
- The imbalanced learn gitter: https://gitter.im/mne-tools/mne-python
-->

<!-- Instructions For Filing a Bug: https://martinos.org/mne/stable/faq.html#i-think-i-found-a-bug-what-do-i-do -->

#### Description
<!-- Example: Joblib Error thrown when calling fit on LatentDirichletAllocation with evaluate_every > 0-->

#### Steps/Code to Reproduce
<!--
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
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

#### Expected Results
<!-- Example: No error is thrown. Please paste or describe the expected results.-->

#### Actual Results
<!-- Please paste or specifically describe the actual output or traceback. -->

#### Versions (``mne.sys_info()`` output)
<details>
<!--
[paste the output of ``mne.sys_info()`` here below this line]
-->

</details>


<!-- Thanks for contributing! -->
