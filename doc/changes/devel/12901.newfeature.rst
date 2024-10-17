Improved reporting and plotting options:

- :meth:`mne.Report.add_projs` can now plot with :func:`mne.viz.plot_projs_joint` rather than :func:`mne.viz.plot_projs_topomap`
- :class:`mne.Report` now has attributes ``img_max_width`` and ``img_max_res`` that can be used to control image scaling.
- :class:`mne.Report` now has an attribute ``collapse`` that allows collapsing sections and/or subsections by default.
- :func:`mne.viz.plot_head_positions` now has a ``totals=True`` option to show the total distance and angle of the head.

Changes by `Eric Larson`_.
