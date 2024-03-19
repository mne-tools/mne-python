Fix bug where using ``phase="minimum"`` in filtering functions like
:meth:`mne.io.Raw.filter` constructed a filter half the desired length with
compromised attenuation. Now ``phase="minimum"`` has the same length and comparable
suppression as ``phase="minimum"``, and the old (incorrect) behavior can be achieved
with ``phase="minimum-half"``, by `Eric Larson`_.
