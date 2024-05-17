The color scaling of Evoked topomaps added to reports via :meth:`mne.Report.add_evokeds`
was sometimes sub-optimal if bad channels were present in the data. This has now been fixed
and should be more consistent with the topomaps shown in the joint plots, by `Richard Höchenberger`_.
