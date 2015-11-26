"""Mayavi/traits GUI for converting data from KIT systems"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import numpy as np
from scipy.linalg import inv
from threading import Thread

from ..externals.six.moves import queue
from ..io.meas_info import _read_dig_points, _make_dig_points
from ..utils import logger


# allow import without traits
try:
    from mayavi.core.ui.mayavi_scene import MayaviScene
    from mayavi.tools.mlab_scene_model import MlabSceneModel
    from pyface.api import confirm, error, FileDialog, OK, YES, information
    from traits.api import (HasTraits, HasPrivateTraits, cached_property,
                            Instance, Property, Bool, Button, Enum, File,
                            Float, Int, List, Str, Array, DelegatesTo)
    from traitsui.api import (View, Item, HGroup, VGroup, spring, TextEditor,
                              CheckListEditor, EnumEditor, Handler)
    from traitsui.menu import NoButtons
    from tvtk.pyface.scene_editor import SceneEditor
except Exception:
    from ..utils import trait_wraith
    HasTraits = HasPrivateTraits = Handler = object
    cached_property = MayaviScene = MlabSceneModel = Bool = Button = Float = \
        DelegatesTo = Enum = File = Instance = Int = List = Property = \
        Str = Array = spring = View = Item = HGroup = VGroup = EnumEditor = \
        NoButtons = CheckListEditor = SceneEditor = TextEditor = trait_wraith

from ..io.kit.kit import RawKIT, KIT
from ..transforms import (apply_trans, als_ras_trans, als_ras_trans_mm,
                          get_ras_to_neuromag_trans, Transform)
from ..coreg import _decimate_points, fit_matched_points
from ._marker_gui import CombineMarkersPanel, CombineMarkersModel
from ._help import read_tooltips
from ._viewer import (HeadViewController, headview_item, PointObject,
                      _testing_mode)


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


tooltips = read_tooltips('kit2fiff')


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
    stim_coding = Enum(">", "<", "channel")
    stim_chs = Str("")
    stim_chs_array = Property(depends_on='stim_chs')
    stim_chs_ok = Property(depends_on='stim_chs_array')
    stim_chs_comment = Property(depends_on='stim_chs_array')
    stim_slope = Enum("-", "+")
    stim_threshold = Float(1.)

    # Marker Points
    use_mrk = List(list(range(5)), desc="Which marker points to use for the "
                   "device head coregistration.")

    # Derived Traits
    mrk = Property(depends_on='markers.mrk3.points')

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
    can_save = Property(Bool, depends_on=['stim_chs_ok', 'sqd_file', 'fid',
                                          'elp', 'hsp', 'dev_head_trans'])

    @cached_property
    def _get_can_save(self):
        "Only allow saving when either all or no head shape elements are set."
        if not self.stim_chs_ok or not self.sqd_file:
            return False

        has_all_hsp = (np.any(self.dev_head_trans) and np.any(self.hsp) and
                       np.any(self.elp) and np.any(self.fid))
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
            pts = _read_dig_points(self.fid_file)
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
            return np.empty((0, 3))
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
            pts = _read_dig_points(fname)
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

    @cached_property
    def _get_stim_chs_array(self):
        if not self.stim_chs.strip():
            return True
        try:
            out = eval("r_[%s]" % self.stim_chs, vars(np))
            if out.dtype.kind != 'i':
                raise TypeError("Need array of int")
        except:
            return None
        else:
            return out

    @cached_property
    def _get_stim_chs_comment(self):
        if self.stim_chs_array is None:
            return "Invalid!"
        elif self.stim_chs_array is True:
            return "Ok: Default channels"
        else:
            return "Ok: %i channels" % len(self.stim_chs_array)

    @cached_property
    def _get_stim_chs_ok(self):
        return self.stim_chs_array is not None

    def clear_all(self):
        """Clear all specified input parameters"""
        self.markers.clear = True
        self.reset_traits(['sqd_file', 'hsp_file', 'fid_file', 'use_mrk'])

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
        if not self.can_save:
            raise ValueError("Not all necessary parameters are set")

        # stim channels and coding
        if self.stim_chs_array is True:
            if self.stim_coding == 'channel':
                stim_code = 'channel'
                raise NotImplementedError("Finding default event channels")
            else:
                stim = self.stim_coding
                stim_code = 'binary'
        else:
            stim = self.stim_chs_array
            if self.stim_coding == 'channel':
                stim_code = 'channel'
            elif self.stim_coding == '<':
                stim_code = 'binary'
            elif self.stim_coding == '>':
                # if stim is
                stim = stim[::-1]
                stim_code = 'binary'
            else:
                raise RuntimeError("stim_coding=%r" % self.stim_coding)

        logger.info("Creating raw with stim=%r, slope=%r, stim_code=%r, "
                    "stimthresh=%r", stim, self.stim_slope, stim_code,
                    self.stim_threshold)
        raw = RawKIT(self.sqd_file, preload=preload, stim=stim,
                     slope=self.stim_slope, stim_code=stim_code,
                     stimthresh=self.stim_threshold)

        if np.any(self.fid):
            raw.info['dig'] = _make_dig_points(self.fid[0], self.fid[1],
                                               self.fid[2], self.elp,
                                               self.hsp)
            raw.info['dev_head_t'] = Transform('meg', 'head',
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
    stim_coding = DelegatesTo('model')
    stim_chs = DelegatesTo('model')
    stim_chs_ok = DelegatesTo('model')
    stim_chs_comment = DelegatesTo('model')
    stim_slope = DelegatesTo('model')
    stim_threshold = DelegatesTo('model')

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

    view = View(
        VGroup(VGroup(Item('sqd_file', label="Data",
                           tooltip=tooltips['sqd_file']),
                      Item('sqd_fname', show_label=False, style='readonly'),
                      Item('hsp_file', label='Dig Head Shape'),
                      Item('hsp_fname', show_label=False, style='readonly'),
                      Item('fid_file', label='Dig Points'),
                      Item('fid_fname', show_label=False, style='readonly'),
                      Item('reset_dig', label='Clear Digitizer Files',
                           show_label=False),
                      Item('use_mrk', editor=use_editor, style='custom'),
                      label="Sources", show_border=True),
               VGroup(Item('stim_slope', label="Event Onset", style='custom',
                           tooltip=tooltips['stim_slope'],
                           editor=EnumEditor(
                               values={'+': '2:Peak (0 to 5 V)',
                                       '-': '1:Trough (5 to 0 V)'},
                               cols=2)),
                      Item('stim_coding', label="Value Coding", style='custom',
                           editor=EnumEditor(values={'>': '1:little-endian',
                                                     '<': '2:big-endian',
                                                     'channel': '3:Channel#'},
                                             cols=3),
                           tooltip=tooltips["stim_coding"]),
                      Item('stim_chs', label='Channels', style='custom',
                           tooltip=tooltips["stim_chs"],
                           editor=TextEditor(evaluate_name='stim_chs_ok',
                                             auto_set=True)),
                      Item('stim_chs_comment', label='>', style='readonly'),
                      Item('stim_threshold', label='Threshold',
                           tooltip=tooltips['stim_threshold']),
                      label='Events', show_border=True),
               HGroup(Item('save_as', enabled_when='can_save'), spring,
                      'clear_all', show_labels=False),
               Item('queue_feedback', show_label=False, style='readonly'),
               Item('queue_current', show_label=False, style='readonly'),
               Item('queue_len_str', show_label=False, style='readonly')
               )
    )

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
        self.elp_obj = PointObject(scene=self.scene, color=(50, 50, 220),
                                   point_scale=1e-2, opacity=.2)
        self.hsp_obj = PointObject(scene=self.scene, color=(200, 200, 200),
                                   point_scale=2e-3)
        if not _testing_mode():
            for name, obj in zip(['fid', 'elp', 'hsp'],
                                 [self.fid_obj, self.elp_obj, self.hsp_obj]):
                m.sync_trait(name, obj, 'points', mutual=False)
                m.sync_trait('head_dev_trans', obj, 'trans', mutual=False)
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
