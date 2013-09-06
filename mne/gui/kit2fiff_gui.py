"""Mayavi/traits GUI for averaging two sets of KIT marker points"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from Queue import Queue
from threading import Thread

import numpy as np
from scipy.linalg import inv

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.api import confirm, error, FileDialog, OK, YES, information
from traits.api import HasTraits, HasPrivateTraits, cached_property, \
                       on_trait_change, Instance, Property, Array, Bool, \
                       Button, Enum, File, Int, List, Str
from traitsui.api import View, Item, Group, HGroup, VGroup, CheckListEditor, \
                         EnumEditor, Handler
from traitsui.menu import NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from ..fiff.kit.coreg import read_hsp, read_elp, get_head_coord_trans
from ..fiff.kit.kit import RawKIT, KIT
from ..transforms import apply_trans, als_ras_trans, als_ras_trans_mm
from ..transforms.coreg import decimate_points, fit_matched_pts
from .marker_gui import CombineMarkersPanel
from .viewer import HeadViewController, headview_borders, headview_item, \
                    PointObject


use_editor = CheckListEditor(cols=5, values=[(i, str(i)) for i in xrange(5)])
hsp_wildcard = ['Plain Text Files (*.txt)|*.txt', 'All Files (*.*)|*.*']
kit_con_wildcard = ['Continuous KIT Files (*.sqd;*.con)|*.sqd;*.con']


class Kit2FiffFrameHandler(Handler):
    """Handler that checks for unfinished processes before closing its window
    """
    def close(self, info, is_ok):
        if info.object.kit2fiff_panel.kit2fiff_coreg_panel\
                                                    .queue.unfinished_tasks:
            msg = ("Can not close the window while saving is still in "
                   "progress. Please wait until all files are processed.")
            title = "Saving Still in Progress"
            information(None, msg, title)
            return False
        else:
            return True


class Kit2FiffCoregPanel(HasPrivateTraits):
    """Control panel for kit2fiff conversion"""
    # Source Files
    sqd_file = File(exists=True, filter=kit_con_wildcard)
    sqd_fname = Property(Str, depends_on='sqd_file')
    hsp_file = File(exists=True, filter=hsp_wildcard, desc="Digitizer head "
                    "shape")
    hsp_fname = Property(Str, depends_on='hsp_file')
    fid_file = File(exists=True, filter=['*.txt'], desc="Digitizer fiducials")
    fid_fname = Property(Str, depends_on='fid_file')
    reset_dig = Button

    # Raw
    raw = Property(depends_on=['sqd_file'])

    # Marker Points
    mrk_ALS = Array(float, shape=(5, 3))
    mrk = Property(depends_on=('mrk_ALS'))
    use_mrk = List(range(5), desc="Which marker points to use for the device "
                   "head coregistration.")

    # Polhemus Fiducials
    elp_raw = Property(depends_on=['fid_file'])
    hsp_raw = Property(depends_on=['hsp_file'])
    hsp_to_mne_trans = Property(depends_on=['elp_raw'])

    # Polhemus data (in neuromag space)
    elp_src = Property(depends_on=['hsp_to_mne_trans'])
    fid_src = Property(depends_on=['hsp_to_mne_trans'])
    hsp_src = Property(depends_on=['hsp_raw', 'hsp_to_mne_trans'])

    dev_head_trans = Array(shape=(4, 4))
    head_dev_trans = Array(shape=(4, 4))

    # Events
    events = Array(Int, shape=(None,), value=[])
    stim_chs = Enum(">", "<")
    stim_slope = Enum("-", "+")
    event_info = Property(Str, depends_on=['events', 'stim_chs'])

    # Visualization
    scene = Instance(MlabSceneModel)
    fid_obj = Instance(PointObject)
    elp_obj = Instance(PointObject)
    hsp_obj = Instance(PointObject)

    # Output
    can_save = Property(Bool, depends_on=['raw', 'fid_src', 'elp_src',
                                          'hsp_src', 'dev_head_trans'])
    save_as = Button(label='Save FIFF...')
    queue = Instance(Queue, ())
    queue_feedback = Str('')
    queue_current = Str('')
    queue_len = Int(0)
    queue_len_str = Property(Str, depends_on=['queue_len'])
    error = Str('')

    view = View(VGroup(VGroup(Item('sqd_file', label="Data"),
                              Item('sqd_fname', show_label=False,
                                   style='readonly'),
                              Item('hsp_file', label='Dig Head Shape'),
                              Item('hsp_fname', show_label=False,
                                   style='readonly'),
                              Item('fid_file', label='Dig Points'),
                              Item('fid_fname', show_label=False,
                                   style='readonly'),
                              Item('reset_dig', label='Clear Digitizer Files',
                                   show_label=False),
                              Item('use_mrk', editor=use_editor,
                                   style='custom'),
                              label="Sources", show_border=True),
                    VGroup(Item('stim_chs', label="Stim Channel Binary Coding",
                                style='custom',
                                editor=EnumEditor(values={'>': '1:low to high',
                                                          '<': '2:high to low',
                                                          },
                                                  cols=2),
                                help="Specifies the bit order in event "
                                "channels. Assign the first bit (1) to the "
                                "first or the last trigger channel."),
                           Item('stim_slope', label="Stim Channel Event Type",
                                style='custom',
                                editor=EnumEditor(values={'+': '2:Peak',
                                                          '-': '1:Trough'},
                                                  cols=2),
                                help="Whether events are marked by a decrease "
                                "(trough) or an increase (peak) in trigger "
                                "channel values"),
                           label='Events', show_border=True),
                       Item('save_as', enabled_when='can_save',
                            show_label=False),
                       Item('queue_feedback', show_label=False,
                            style='readonly'),
                       Item('queue_current', show_label=False,
                            style='readonly'),
                       Item('queue_len_str', show_label=False,
                            style='readonly'),
                       ))

    def __init__(self, *args, **kwargs):
        super(Kit2FiffCoregPanel, self).__init__(*args, **kwargs)

        # setup save worker
        def worker():
            while True:
                raw, fname = self.queue.get()
                basename = os.path.basename(fname)
                self.queue_len -= 1
                self.queue_current = 'Processing: %s' % basename

                # task
                try:
                    raw.save(fname, overwrite=True)
                except Exception as err:
                    self.error = str(err)
                    res = "Error saving: %s"
                else:
                    res = "Saved: %s"

                # finalize
                self.queue_current = ''
                self.queue_feedback = res % basename
                self.queue.task_done()

        t = Thread(target=worker)
        t.daemon = True
        t.start()

    @cached_property
    def _get_sqd_fname(self):
        if self.sqd_file:
            return os.path.basename(self.sqd_file)
        else:
            return '-'

    @cached_property
    def _get_hsp_fname(self):
        if self.hsp_file:
            return os.path.basename(self.hsp_file)
        else:
            return '-'

    @cached_property
    def _get_fid_fname(self):
        if self.fid_file:
            return os.path.basename(self.fid_file)
        else:
            return '-'

    @cached_property
    def _get_mrk(self):
        return apply_trans(als_ras_trans, self.mrk_ALS)

    @cached_property
    def _get_elp_raw(self):
        if not self.fid_file:
            return

        try:
            pts = read_elp(self.fid_file)
        except Exception as err:
            error(None, str(err), "Error Reading Fiducials")
            self.reset_traits(['fid_file'])
            raise
        else:
            return pts

    @cached_property
    def _get_hsp_raw(self):
        fname = self.hsp_file
        if not fname:
            return

        try:
            pts = read_hsp(fname)

            n_pts = len(pts)
            if n_pts > KIT.DIG_POINTS:
                pts = decimate_points(pts, 5)
                n_new = len(pts)
                msg = ("The selected head shape contained {n_in} points, which is "
                       "more than recommended ({n_rec}), and was automatically "
                       "downsampled to {n_new} points. The preferred way to "
                       "downsample is using FastScan.")
                msg = msg.format(n_in=n_pts, n_rec=KIT.DIG_POINTS, n_new=n_new)
                information(None, msg, "Head Shape Downsampled")

        except Exception as err:
            error(None, str(err), "Error Reading Head Shape")
            self.reset_traits(['hsp_file'])
            raise
        else:
            return pts

    @cached_property
    def _get_hsp_to_mne_trans(self):
        if self.elp_raw is None:
            return
        pts = apply_trans(als_ras_trans_mm, self.elp_raw[:3])
        nasion, lpa, rpa = pts
        trans = get_head_coord_trans(nasion, lpa, rpa)
        als_trans = np.vstack((np.hstack((als_ras_trans_mm, [[0], [0], [0]])),
                               [0, 0, 0, 1]))
        trans = np.dot(trans, als_trans)
        return trans

    @cached_property
    def _get_fid_src(self):
        if self.elp_raw is None:
            return np.empty((0, 3))
        pts = self.elp_raw[:3]
        pts = apply_trans(self.hsp_to_mne_trans, pts)
        return pts

    # cached_property
    def _get_elp_src(self):
        if self.elp_raw is None:
            return np.empty((0, 3))
        pts = self.elp_raw[3:]
        pts = apply_trans(self.hsp_to_mne_trans, pts)
        return pts

    @cached_property
    def _get_hsp_src(self):
        if (self.hsp_raw is None) or not np.any(self.hsp_to_mne_trans):
            return  np.empty((0, 3))
        else:
            pts = apply_trans(self.hsp_to_mne_trans, self.hsp_raw)
            return pts

    @on_trait_change('elp_src,use_mrk,mrk')
    def update_dev_head_trans(self):
        if (self.mrk is None) or not np.any(self.fid_src):
            trans = np.zeros((4, 4))
            self.dev_head_trans = trans
            self.head_dev_trans = trans
            return

        src_pts = self.mrk
        dst_pts = self.elp_src

        n_use = len(self.use_mrk)
        if n_use < 3:
            error(None, "Estimating the device head transform requires at "
                  "least 3 marker points. Please adjust the markers used.",
                  "Not Enough Marker Points")
            return
        elif n_use < 5:
            src_pts = src_pts[self.use_mrk]
            dst_pts = dst_pts[self.use_mrk]

        trans = fit_matched_pts(src_pts, dst_pts, out='trans')
        self.dev_head_trans = trans
        self.head_dev_trans = inv(trans)

    @cached_property
    def _get_event_info(self):
        """
        Return a string with the number of events found for each trigger value
        """
        if len(self.events) == 0:
            return "No events found."

        count = ["Events found:"]
        events = np.array(self.events)
        for i in np.unique(events):
            n = np.sum(events == i)
            count.append('%3i: %i' % (i, n))

        return os.linesep.join(count)

    @cached_property
    def _get_can_save(self):
        can_save = (
                    # with head shape:
                    (self.raw and np.any(self.dev_head_trans)
                     and np.any(self.hsp_src) and np.any(self.elp_src)
                     and np.any(self.fid_src)) or
                    # without head shape:
                    (self.raw and not (self.hsp_file or self.fid_file
                                       or np.any(self.mrk))))
        return can_save

    @cached_property
    def _get_raw(self):
        if not self.sqd_file:
            return

        try:
            raw = RawKIT(self.sqd_file, preload=False)
        except Exception as err:
            error(None, str(err), "Error Creating KIT Raw")
            raise
        else:
            return raw

    @cached_property
    def _get_queue_len_str(self):
        if self.queue_len:
            return "Queue length: %i" % self.queue_len
        else:
            return ''

    def _reset_dig_fired(self):
        self.reset_traits(['hsp_file', 'fid_file'])

    def _save_as_fired(self):
        # find default path
        stem, _ = os.path.splitext(self.sqd_file)
        if not stem.endswith('raw'):
            stem += '-raw'
        default_path = stem + '.fif'

        # save as dialog
        dlg = FileDialog(action="save as",
                         wildcard="fiff raw file (*.fif)|*.fif",
                         default_path=default_path)
        dlg.open()
        if dlg.return_code != OK:
            return

        fname = dlg.path
        if not fname.endswith('.fif'):
            fname += '.fif'
            if os.path.exists(fname):
                answer = confirm(None, "The file %r already exists. Should it "
                                 "be replaced?", "Overwrite File?")
                if answer != YES:
                    return

        raw = self.raw
        raw.set_stimchannels(self.stim_chs, self.stim_slope)

        if np.any(self.fid_src):
            raw.set_dig_neuromag(self.fid_src, self.elp_src, self.hsp_src,
                                 self.dev_head_trans)

        self.queue.put((raw, fname))
        self.queue_len += 1

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.fid_obj = PointObject(scene=self.scene, color=(25, 225, 25),
                                   point_scale=5e-3)
        self.sync_trait('fid_src', self.fid_obj, 'points', mutual=False)
        self.sync_trait('head_dev_trans', self.fid_obj, 'trans', mutual=False)

        self.elp_obj = PointObject(scene=self.scene, color=(50, 50, 220),
                                   point_scale=1e-2, opacity=.2)
        self.sync_trait('elp_src', self.elp_obj, 'points', mutual=False)
        self.sync_trait('head_dev_trans', self.elp_obj, 'trans', mutual=False)

        self.hsp_obj = PointObject(scene=self.scene, color=(200, 200, 200),
                                   point_scale=2e-3)
        self.sync_trait('hsp_src', self.hsp_obj, 'points', mutual=False)
        self.sync_trait('head_dev_trans', self.hsp_obj, 'trans', mutual=False)


class Kit2FiffPanel(HasTraits):
    scene = Instance(MlabSceneModel, ())
    marker_panel = Instance(CombineMarkersPanel)
    kit2fiff_coreg_panel = Instance(Kit2FiffCoregPanel)

    view = View(Group(Item('marker_panel', label="Markers", style="custom",
                           dock='tab'),
                      Item('kit2fiff_coreg_panel', label="Kit2Fiff",
                           style="custom", dock='tab'),
                      layout='tabbed', show_labels=False)
                      )

    def _marker_panel_default(self):
        panel = CombineMarkersPanel(scene=self.scene)
        return panel

    def _kit2fiff_coreg_panel_default(self):
        panel = Kit2FiffCoregPanel(scene=self.scene)
        return panel

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.marker_panel.mrk1_obj.trans = als_ras_trans
        self.marker_panel.mrk2_obj.trans = als_ras_trans
        self.marker_panel.mrk3_obj.trans = als_ras_trans

        mrk = self.marker_panel.mrk3
        mrk.sync_trait('points', self.kit2fiff_coreg_panel, 'mrk_ALS',
                       mutual=False)


view_hrs = View(HGroup(Item('scene',
                            editor=SceneEditor(scene_class=MayaviScene)),
                       VGroup(headview_borders,
                              Item('panel', style='custom'),
                              show_labels=False),
                       show_labels=False,
                      ),
                resizable=True,
                buttons=NoButtons)


class Kit2FiffFrame(HasTraits):
    """GUI for interpolating between two KIT marker files"""
    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)
    kit2fiff_panel = Instance(Kit2FiffPanel)

    def _headview_default(self):
        hv = HeadViewController(scene=self.scene, scale=160, system='RAS')
        return hv

    def _kit2fiff_panel_default(self):
        p = Kit2FiffPanel(scene=self.scene)
        return p

    view = View(HGroup(VGroup(Item('scene',
                                   editor=SceneEditor(scene_class=MayaviScene),
                                   dock='vertical', show_label=False),
                              VGroup(headview_item,
                                     show_labels=False,
                                     ),
                              ),
                       VGroup(Item('kit2fiff_panel', style='custom'),
                              show_labels=False),
                       show_labels=False,
                      ),
                handler=Kit2FiffFrameHandler(),
                width=1100, height=700, resizable=True, buttons=NoButtons)
