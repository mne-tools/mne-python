"""Mayavi/traits GUI for converting data from KIT systems"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from ..externals.six.moves import queue
from threading import Thread

import numpy as np
from scipy.linalg import inv

# allow import without traits
try:
    from mayavi.core.ui.mayavi_scene import MayaviScene
    from mayavi.tools.mlab_scene_model import MlabSceneModel
    from pyface.api import confirm, error, FileDialog, OK, YES, information
    from traits.api import (HasTraits, HasPrivateTraits, cached_property,
                            Instance, Property, Bool, Button, Enum, File, Int,
                            List, Str, Array, DelegatesTo)
    from traitsui.api import (View, Item, HGroup, VGroup, spring,
                              CheckListEditor, EnumEditor, Handler)
    from traitsui.menu import NoButtons
    from tvtk.pyface.scene_editor import SceneEditor
except:
    from ..utils import trait_wraith
    HasTraits = object
    HasPrivateTraits = object
    Handler = object
    cached_property = trait_wraith
    MayaviScene = trait_wraith
    MlabSceneModel = trait_wraith
    Bool = trait_wraith
    Button = trait_wraith
    DelegatesTo = trait_wraith
    Enum = trait_wraith
    File = trait_wraith
    Instance = trait_wraith
    Int = trait_wraith
    List = trait_wraith
    Property = trait_wraith
    Str = trait_wraith
    Array = trait_wraith
    spring = trait_wraith
    View = trait_wraith
    Item = trait_wraith
    HGroup = trait_wraith
    VGroup = trait_wraith
    EnumEditor = trait_wraith
    NoButtons = trait_wraith
    CheckListEditor = trait_wraith
    SceneEditor = trait_wraith

from ..io.kit.coreg import read_hsp
from ..io.kit.kit import RawKIT, KIT
from ..transforms import apply_trans, als_ras_trans, als_ras_trans_mm
from ..coreg import (read_elp, _decimate_points, fit_matched_points,
                     get_ras_to_neuromag_trans)
from ._marker_gui import CombineMarkersPanel, CombineMarkersModel
from ._viewer import HeadViewController, headview_item, PointObject


use_editor = CheckListEditor(cols=5, values=[(i, str(i)) for i in range(5)])
backend_is_wx = False  # is there a way to determine this?
if backend_is_wx:
    # wx backend allows labels for wildcards
    hsp_points_wildcard = ['Head Shape Points (*.txt)|*.txt']
    hsp_fid_wildcard = ['Head Shape Fiducials (*.txt)|*.txt']
    kit_con_wildcard = ['Continuous KIT Files (*.sqd;*.con)|*.sqd;*.con']
else:
    hsp_points_wildcard = ['*.txt']
    hsp_fid_wildcard = ['*.txt']
    kit_con_wildcard = ['*.sqd;*.con']


class Kit2FiffModel(HasPrivateTraits):
    """Data Model for Kit2Fiff conversion

     - Markers are transformed into RAS coordinate system (as are the sensor
       coordinates).
     - Head shape digitizer data is transformed into neuromag-like space.

    """
    # Input Traits
    markers = Instance(CombineMarkersModel, ())
    sqd_file = File(exists=True, filter=kit_con_wildcard)
    hsp_file = File(exists=True, filter=hsp_points_wildcard, desc="Digitizer "
                    "head shape")
    fid_file = File(exists=True, filter=hsp_fid_wildcard, desc="Digitizer "
                    "fiducials")
    stim_chs = Enum(">", "<", "man")
    stim_chs_manual = Array(int, (8,), range(168, 176))
    stim_slope = Enum("-", "+")
    # Marker Points
    use_mrk = List(list(range(5)), desc="Which marker points to use for the device "
                   "head coregistration.")

    # Derived Traits
    mrk = Property(depends_on=('markers.mrk3.points'))

    # Polhemus Fiducials
    elp_raw = Property(depends_on=['fid_file'])
    hsp_raw = Property(depends_on=['hsp_file'])
    polhemus_neuromag_trans = Property(depends_on=['elp_raw'])

    # Polhemus data (in neuromag space)
    elp = Property(depends_on=['elp_raw', 'polhemus_neuromag_trans'])
    fid = Property(depends_on=['elp_raw', 'polhemus_neuromag_trans'])
    hsp = Property(depends_on=['hsp_raw', 'polhemus_neuromag_trans'])

    # trans
    dev_head_trans = Property(depends_on=['elp', 'mrk', 'use_mrk'])
    head_dev_trans = Property(depends_on=['dev_head_trans'])

    # info
    sqd_fname = Property(Str, depends_on='sqd_file')
    hsp_fname = Property(Str, depends_on='hsp_file')
    fid_fname = Property(Str, depends_on='fid_file')
    can_save = Property(Bool, depends_on=['sqd_file', 'fid', 'elp', 'hsp',
                                          'dev_head_trans'])

    @cached_property
    def _get_can_save(self):
        "Only allow saving when either all or no head shape elements are set."
        has_sqd = bool(self.sqd_file)
        if not has_sqd:
            return False

        has_all_hsp = (np.any(self.dev_head_trans) and np.any(self.hsp)
                       and np.any(self.elp) and np.any(self.fid))
        if has_all_hsp:
            return True

        has_any_hsp = self.hsp_file or self.fid_file or np.any(self.mrk)
        return not has_any_hsp

    @cached_property
    def _get_dev_head_trans(self):
        if (self.mrk is None) or not np.any(self.fid):
            return np.eye(4)

        src_pts = self.mrk
        dst_pts = self.elp

        n_use = len(self.use_mrk)
        if n_use < 3:
            error(None, "Estimating the device head transform requires at "
                  "least 3 marker points. Please adjust the markers used.",
                  "Not Enough Marker Points")
            return
        elif n_use < 5:
            src_pts = src_pts[self.use_mrk]
            dst_pts = dst_pts[self.use_mrk]

        trans = fit_matched_points(src_pts, dst_pts, out='trans')
        return trans

    @cached_property
    def _get_elp(self):
        if self.elp_raw is None:
            return np.empty((0, 3))
        pts = self.elp_raw[3:8]
        pts = apply_trans(self.polhemus_neuromag_trans, pts)
        return pts

    @cached_property
    def _get_elp_raw(self):
        if not self.fid_file:
            return

        try:
            pts = read_elp(self.fid_file)
            if len(pts) < 8:
                raise ValueError("File contains %i points, need 8" % len(pts))
        except Exception as err:
            error(None, str(err), "Error Reading Fiducials")
            self.reset_traits(['fid_file'])
            raise
        else:
            return pts

    @cached_property
    def _get_fid(self):
        if self.elp_raw is None:
            return np.empty((0, 3))
        pts = self.elp_raw[:3]
        pts = apply_trans(self.polhemus_neuromag_trans, pts)
        return pts

    @cached_property
    def _get_fid_fname(self):
        if self.fid_file:
            return os.path.basename(self.fid_file)
        else:
            return '-'

    @cached_property
    def _get_head_dev_trans(self):
        return inv(self.dev_head_trans)

    @cached_property
    def _get_hsp(self):
        if (self.hsp_raw is None) or not np.any(self.polhemus_neuromag_trans):
            return  np.empty((0, 3))
        else:
            pts = apply_trans(self.polhemus_neuromag_trans, self.hsp_raw)
            return pts

    @cached_property
    def _get_hsp_fname(self):
        if self.hsp_file:
            return os.path.basename(self.hsp_file)
        else:
            return '-'

    @cached_property
    def _get_hsp_raw(self):
        fname = self.hsp_file
        if not fname:
            return

        try:
            pts = read_hsp(fname)

            n_pts = len(pts)
            if n_pts > KIT.DIG_POINTS:
                msg = ("The selected head shape contains {n_in} points, "
                       "which is more than the recommended maximum ({n_rec}). "
                       "The file will be automatically downsampled, which "
                       "might take a while. A better way to downsample is "
                       "using FastScan.")
                msg = msg.format(n_in=n_pts, n_rec=KIT.DIG_POINTS)
                information(None, msg, "Too Many Head Shape Points")
                pts = _decimate_points(pts, 5)

        except Exception as err:
            error(None, str(err), "Error Reading Head Shape")
            self.reset_traits(['hsp_file'])
            raise
        else:
            return pts

    @cached_property
    def _get_mrk(self):
        return apply_trans(als_ras_trans, self.markers.mrk3.points)

    @cached_property
    def _get_polhemus_neuromag_trans(self):
        if self.elp_raw is None:
            return
        pts = apply_trans(als_ras_trans_mm, self.elp_raw[:3])
        nasion, lpa, rpa = pts
        trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        trans = np.dot(trans, als_ras_trans_mm)
        return trans

    @cached_property
    def _get_sqd_fname(self):
        if self.sqd_file:
            return os.path.basename(self.sqd_file)
        else:
            return '-'

    def clear_all(self):
        """Clear all specified input parameters"""
        self.markers.clear = True
        self.reset_traits(['sqd_file', 'hsp_file', 'fid_file'])

    def get_event_info(self):
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

    def get_raw(self, preload=False):
        """Create a raw object based on the current model settings
        """
        if not self.sqd_file:
            raise ValueError("sqd file not set")

        if self.stim_chs == 'man':
            stim = self.stim_chs_manual
        else:
            stim = self.stim_chs

        raw = RawKIT(self.sqd_file, preload=preload, stim=stim,
                     slope=self.stim_slope)

        if np.any(self.fid):
            raw._set_dig_neuromag(self.fid, self.elp, self.hsp,
                                  self.dev_head_trans)
        return raw


class Kit2FiffFrameHandler(Handler):
    """Handler that checks for unfinished processes before closing its window
    """
    def close(self, info, is_ok):
        if info.object.kit2fiff_panel.queue.unfinished_tasks:
            msg = ("Can not close the window while saving is still in "
                   "progress. Please wait until all files are processed.")
            title = "Saving Still in Progress"
            information(None, msg, title)
            return False
        else:
            return True


class Kit2FiffPanel(HasPrivateTraits):
    """Control panel for kit2fiff conversion"""
    model = Instance(Kit2FiffModel)

    # model copies for view
    use_mrk = DelegatesTo('model')
    sqd_file = DelegatesTo('model')
    hsp_file = DelegatesTo('model')
    fid_file = DelegatesTo('model')
    stim_chs = DelegatesTo('model')
    stim_chs_manual = DelegatesTo('model')
    stim_slope = DelegatesTo('model')

    # info
    can_save = DelegatesTo('model')
    sqd_fname = DelegatesTo('model')
    hsp_fname = DelegatesTo('model')
    fid_fname = DelegatesTo('model')

    # Source Files
    reset_dig = Button

    # Visualization
    scene = Instance(MlabSceneModel)
    fid_obj = Instance(PointObject)
    elp_obj = Instance(PointObject)
    hsp_obj = Instance(PointObject)

    # Output
    save_as = Button(label='Save FIFF...')
    clear_all = Button(label='Clear All')
    queue = Instance(queue.Queue, ())
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
                    VGroup(Item('stim_slope', label="Event Onset",
                                style='custom',
                                editor=EnumEditor(
                                           values={'+': '2:Peak (0 to 5 V)',
                                                   '-': '1:Trough (5 to 0 V)'},
                                           cols=2),
                                help="Whether events are marked by a decrease "
                                "(trough) or an increase (peak) in trigger "
                                "channel values"),
                           Item('stim_chs', label="Binary Coding",
                                style='custom',
                                editor=EnumEditor(values={'>': '1:1 ... 128',
                                                          '<': '3:128 ... 1',
                                                          'man': '2:Manual'},
                                                  cols=2),
                                help="Specifies the bit order in event "
                                "channels. Assign the first bit (1) to the "
                                "first or the last trigger channel."),
                           Item('stim_chs_manual', label='Stim Channels',
                                style='custom',
                                visible_when="stim_chs == 'man'"),
                           label='Events', show_border=True),
                       HGroup(Item('save_as', enabled_when='can_save'), spring,
                              'clear_all', show_labels=False),
                       Item('queue_feedback', show_label=False,
                            style='readonly'),
                       Item('queue_current', show_label=False,
                            style='readonly'),
                       Item('queue_len_str', show_label=False,
                            style='readonly'),
                       ))

    def __init__(self, *args, **kwargs):
        super(Kit2FiffPanel, self).__init__(*args, **kwargs)

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

        # setup mayavi visualization
        m = self.model
        self.fid_obj = PointObject(scene=self.scene, color=(25, 225, 25),
                                   point_scale=5e-3)
        m.sync_trait('fid', self.fid_obj, 'points', mutual=False)
        m.sync_trait('head_dev_trans', self.fid_obj, 'trans', mutual=False)

        self.elp_obj = PointObject(scene=self.scene, color=(50, 50, 220),
                                   point_scale=1e-2, opacity=.2)
        m.sync_trait('elp', self.elp_obj, 'points', mutual=False)
        m.sync_trait('head_dev_trans', self.elp_obj, 'trans', mutual=False)

        self.hsp_obj = PointObject(scene=self.scene, color=(200, 200, 200),
                                   point_scale=2e-3)
        m.sync_trait('hsp', self.hsp_obj, 'points', mutual=False)
        m.sync_trait('head_dev_trans', self.hsp_obj, 'trans', mutual=False)

        self.scene.camera.parallel_scale = 0.15
        self.scene.mlab.view(0, 0, .15)

    def _clear_all_fired(self):
        self.model.clear_all()

    @cached_property
    def _get_queue_len_str(self):
        if self.queue_len:
            return "Queue length: %i" % self.queue_len
        else:
            return ''

    def _reset_dig_fired(self):
        self.reset_traits(['hsp_file', 'fid_file'])

    def _save_as_fired(self):
        # create raw
        try:
            raw = self.model.get_raw()
        except Exception as err:
            error(None, str(err), "Error Creating KIT Raw")
            raise

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

        self.queue.put((raw, fname))
        self.queue_len += 1


class Kit2FiffFrame(HasTraits):
    """GUI for interpolating between two KIT marker files"""
    model = Instance(Kit2FiffModel, ())
    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)
    marker_panel = Instance(CombineMarkersPanel)
    kit2fiff_panel = Instance(Kit2FiffPanel)

    view = View(HGroup(VGroup(Item('marker_panel', style='custom'),
                              show_labels=False),
                       VGroup(Item('scene',
                                   editor=SceneEditor(scene_class=MayaviScene),
                                   dock='vertical', show_label=False),
                              VGroup(headview_item, show_labels=False),
                              ),
                       VGroup(Item('kit2fiff_panel', style='custom'),
                              show_labels=False),
                       show_labels=False,
                      ),
                handler=Kit2FiffFrameHandler(),
                height=700, resizable=True, buttons=NoButtons)

    def _headview_default(self):
        return HeadViewController(scene=self.scene, scale=160, system='RAS')

    def _kit2fiff_panel_default(self):
        return Kit2FiffPanel(scene=self.scene, model=self.model)

    def _marker_panel_default(self):
        return CombineMarkersPanel(scene=self.scene, model=self.model.markers,
                                   trans=als_ras_trans)
