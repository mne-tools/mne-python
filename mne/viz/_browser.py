from abc import ABC

import numpy as np


class MNEBrowserParams:
    def __init__(self, **kwargs):
        # default key to close window
        self.close_key = 'escape'
        vars(self).update(**kwargs)


class MNEDataBrowser(ABC):
    def __init__(self, **kwargs):
        from .. import BaseEpochs
        from ..io import BaseRaw
        from ..preprocessing import ICA

        self._data = None
        self._times = None

        self.mne = MNEBrowserParams(**kwargs)

        inst = kwargs['inst'] if 'inst' in kwargs else None
        ica = kwargs['ica'] if 'ica' in kwargs else None

        # what kind of data are we dealing with?
        if inst is not None:
            if isinstance(ica, ICA):
                self.mne.instance_type = 'ica'
            elif isinstance(inst, BaseRaw):
                self.mne.instance_type = 'raw'
            elif isinstance(inst, BaseEpochs):
                self.mne.instance_type = 'epochs'
            else:
                raise TypeError('Expected an instance of Raw, Epochs, or ICA, '
                                f'got {type(inst)}.')
        else:
            self.mne.instance_type = None

        self.mne.ica_type = None
        if self.mne.instance_type == 'ica':
            if isinstance(self.mne.ica_inst, BaseRaw):
                self.mne.ica_type = 'raw'
            elif isinstance(self.mne.ica_inst, BaseEpochs):
                self.mne.ica_type = 'epochs'
        self.mne.is_epochs = 'epochs' in (self.mne.instance_type,
                                          self.mne.ica_type)

        # things that always start the same
        self.mne.ch_start = 0
        self.mne.projector = None
        if hasattr(self.mne, 'projs'):
            self.mne.projs_active = np.array([p['active']
                                              for p in self.mne.projs])
        self.mne.whitened_ch_names = list()
        if hasattr(self.mne, 'noise_cov'):
            self.mne.use_noise_cov = self.mne.noise_cov is not None
        self.mne.zorder = dict(patch=0, grid=1, ann=2, events=3, bads=4,
                               data=5, mag=6, grad=7, scalebar=8, vline=9)
        # additional params for epochs (won't affect raw / ICA)
        self.mne.epoch_traces = list()
        self.mne.bad_epochs = list()
        if inst is not None:
            self.mne.sampling_period = (np.diff(inst.times[:2])[0]
                                        / inst.info['sfreq'])
        # annotations
        self.mne.annotations = list()
        self.mne.hscroll_annotations = list()
        self.mne.annotation_segments = list()
        self.mne.annotation_texts = list()
        self.mne.new_annotation_labels = list()
        self.mne.annotation_segment_colors = dict()
        self.mne.annotation_hover_line = None
        self.mne.draggable_annotations = False
        # lines
        self.mne.event_lines = None
        self.mne.event_texts = list()
        self.mne.vline_visible = False
        # scalings
        if hasattr(self.mne, 'butterfly'):
            self.mne.scale_factor = 0.5 if self.mne.butterfly else 1.
        self.mne.scalebars = dict()
        self.mne.scalebar_texts = dict()
        # ancillary child figures
        self.mne.child_figs = list()
        self.mne.fig_help = None
        self.mne.fig_proj = None
        self.mne.fig_histogram = None
        self.mne.fig_selection = None
        self.mne.fig_annotation = None
