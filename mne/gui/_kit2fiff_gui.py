"""Mayavi/traits GUI for converting data from KIT systems."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from collections import Counter
import os
import queue
import sys

import numpy as np
from scipy.linalg import inv
from threading import Thread

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.api import (confirm, error, FileDialog, OK, YES, information,
                        ProgressDialog, warning)
from traits.api import (HasTraits, HasPrivateTraits, cached_property, Instance,
                        Property, Bool, Button, Enum, File, Float, Int, List,
                        Str, Array, DelegatesTo, on_trait_change)
from traits.trait_base import ETSConfig
from traitsui.api import (View, Item, HGroup, VGroup, spring, TextEditor,
                          CheckListEditor, EnumEditor, Handler)
from traitsui.menu import NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from ..io.constants import FIFF
from ..io._digitization import _read_dig_points, _make_dig_points
from ..io.kit.kit import (RawKIT, KIT, _make_stim_channel, _default_stim_chs,
                          UnsupportedKITFormat)
from ..transforms import (apply_trans, als_ras_trans,
                          get_ras_to_neuromag_trans, Transform)
from ..coreg import _decimate_points, fit_matched_points
from ..utils import get_config, set_config, logger, warn
from ._backend import _get_pyface_backend
from ..event import _find_events
from ._marker_gui import CombineMarkersPanel, CombineMarkersModel
from ._help import read_tooltips
from ._viewer import HeadViewController, PointObject

use_editor = CheckListEditor(cols=5, values=[(i, str(i)) for i in range(5)])

if _get_pyface_backend() == 'wx':
    # wx backend allows labels for wildcards
    hsp_wildcard = ['Head Shape Points (*.hsp;*.txt)|*.hsp;*.txt']
    elp_wildcard = ['Head Shape Fiducials (*.elp;*.txt)|*.elp;*.txt']
    kit_con_wildcard = ['Continuous KIT Files (*.sqd;*.con)|*.sqd;*.con']
if sys.platform in ('win32', 'linux2'):
    # on Windows and Ubuntu, multiple wildcards does not seem to work
    hsp_wildcard = ['*.hsp', '*.txt']
    elp_wildcard = ['*.elp', '*.txt']
    kit_con_wildcard = ['*.sqd', '*.con']
else:
    hsp_wildcard = ['*.hsp;*.txt']
    elp_wildcard = ['*.elp;*.txt']
    kit_con_wildcard = ['*.sqd;*.con']


tooltips = read_tooltips('kit2fiff')


class Kit2FiffModel(HasPrivateTraits):
    """Data Model for Kit2Fiff conversion.

    - Markers are transformed into RAS coordinate system (as are the sensor
      coordinates).
    - Head shape digitizer data is transformed into neuromag-like space.
    """

    # Input Traits
    markers = Instance(CombineMarkersModel, ())
    sqd_file = File(exists=True, filter=kit_con_wildcard)
    allow_unknown_format = Bool(False)
    hsp_file = File(exists=True, filter=hsp_wildcard)
    fid_file = File(exists=True, filter=elp_wildcard)
    stim_coding = Enum(">", "<", "channel")
    stim_chs = Str("")
    stim_chs_array = Property(depends_on=['raw', 'stim_chs', 'stim_coding'])
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

    # event preview
    raw = Property(depends_on='sqd_file')
    misc_chs = Property(List, depends_on='raw')
    misc_chs_desc = Property(Str, depends_on='misc_chs')
    misc_data = Property(Array, depends_on='raw')
    can_test_stim = Property(Bool, depends_on='raw')

    # info
    sqd_fname = Property(Str, depends_on='sqd_file')
    hsp_fname = Property(Str, depends_on='hsp_file')
    fid_fname = Property(Str, depends_on='fid_file')
    can_save = Property(Bool, depends_on=['stim_chs_ok', 'fid',
                                          'elp', 'hsp', 'dev_head_trans'])

    # Show GUI feedback (like error messages and progress bar)
    show_gui = Bool(False)

    @cached_property
    def _get_can_save(self):
        """Only allow saving when all or no head shape elements are set."""
        if not self.stim_chs_ok:
            return False

        has_all_hsp = (np.any(self.dev_head_trans) and np.any(self.hsp) and
                       np.any(self.elp) and np.any(self.fid))
        if has_all_hsp:
            return True

        has_any_hsp = self.hsp_file or self.fid_file or np.any(self.mrk)
        return not has_any_hsp

    @cached_property
    def _get_can_test_stim(self):
        return self.raw is not None

    @cached_property
    def _get_dev_head_trans(self):
        if (self.mrk is None) or not np.any(self.fid):
            return np.eye(4)

        src_pts = self.mrk
        dst_pts = self.elp

        n_use = len(self.use_mrk)
        if n_use < 3:
            if self.show_gui:
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
            if self.show_gui:
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
                       "using FastScan.".
                       format(n_in=n_pts, n_rec=KIT.DIG_POINTS))
                if self.show_gui:
                    information(None, msg, "Too Many Head Shape Points")
                pts = _decimate_points(pts, 5)

        except Exception as err:
            if self.show_gui:
                error(None, str(err), "Error Reading Head Shape")
            self.reset_traits(['hsp_file'])
            raise
        else:
            return pts

    @cached_property
    def _get_misc_chs(self):
        if not self.raw:
            return
        return [i for i, ch in enumerate(self.raw.info['chs']) if
                ch['kind'] == FIFF.FIFFV_MISC_CH]

    @cached_property
    def _get_misc_chs_desc(self):
        if self.misc_chs is None:
            return "No SQD file selected..."
        elif np.all(np.diff(self.misc_chs) == 1):
            return "%i:%i" % (self.misc_chs[0], self.misc_chs[-1] + 1)
        else:
            return "%i... (discontinuous)" % self.misc_chs[0]

    @cached_property
    def _get_misc_data(self):
        if not self.raw:
            return
        if self.show_gui:
            # progress dialog with indefinite progress bar
            prog = ProgressDialog(title="Loading SQD data...",
                                  message="Loading stim channel data from SQD "
                                  "file ...")
            prog.open()
            prog.update(0)
        else:
            prog = None

        try:
            data, times = self.raw[self.misc_chs]
        except Exception as err:
            if self.show_gui:
                error(None, "Error reading SQD data file: %s (Check the "
                      "terminal output for details)" % str(err),
                      "Error Reading SQD File")
            raise
        finally:
            if self.show_gui:
                prog.close()
        return data

    @cached_property
    def _get_mrk(self):
        return apply_trans(als_ras_trans, self.markers.mrk3.points)

    @cached_property
    def _get_polhemus_neuromag_trans(self):
        if self.elp_raw is None:
            return
        nasion, lpa, rpa = apply_trans(als_ras_trans, self.elp_raw[:3])
        trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        return np.dot(trans, als_ras_trans)

    @cached_property
    def _get_raw(self):
        if not self.sqd_file:
            return
        try:
            return RawKIT(self.sqd_file, stim=None,
                          allow_unknown_format=self.allow_unknown_format)
        except UnsupportedKITFormat as exception:
            warning(
                None,
                "The selected SQD file is written in an old file format (%s) "
                "that is not officially supported. Confirm that the results "
                "are as expected. This warning is displayed only once per "
                "session." % (exception.sqd_version,),
                "Unsupported SQD File Format")
            self.allow_unknown_format = True
            return self._get_raw()
        except Exception as err:
            self.reset_traits(['sqd_file'])
            if self.show_gui:
                error(None, "Error reading SQD data file: %s (Check the "
                      "terminal output for details)" % str(err),
                      "Error Reading SQD File")
            raise

    @cached_property
    def _get_sqd_fname(self):
        if self.sqd_file:
            return os.path.basename(self.sqd_file)
        else:
            return '-'

    @cached_property
    def _get_stim_chs_array(self):
        if self.raw is None:
            return
        elif not self.stim_chs.strip():
            picks = _default_stim_chs(self.raw.info)
        else:
            try:
                picks = eval("r_[%s]" % self.stim_chs, vars(np))
                if picks.dtype.kind != 'i':
                    raise TypeError("Need array of int")
            except Exception:
                return None

        if self.stim_coding == '<':  # Big-endian
            return picks[::-1]
        else:
            return picks

    @cached_property
    def _get_stim_chs_comment(self):
        if self.raw is None:
            return ""
        elif not self.stim_chs_ok:
            return "Invalid!"
        elif not self.stim_chs.strip():
            return "Default:  The first 8 MISC channels"
        else:
            return "Ok:  %i channels" % len(self.stim_chs_array)

    @cached_property
    def _get_stim_chs_ok(self):
        return self.stim_chs_array is not None

    def clear_all(self):
        """Clear all specified input parameters."""
        self.markers.clear = True
        self.reset_traits(['sqd_file', 'hsp_file', 'fid_file', 'use_mrk'])

    def get_event_info(self):
        """Count events with current stim channel settings.

        Returns
        -------
        event_count : Counter
            Counter mapping event ID to number of occurrences.
        """
        if self.misc_data is None:
            return
        idx = [self.misc_chs.index(ch) for ch in self.stim_chs_array]
        data = self.misc_data[idx]
        if self.stim_coding == 'channel':
            coding = 'channel'
        else:
            coding = 'binary'
        stim_ch = _make_stim_channel(data, self.stim_slope,
                                     self.stim_threshold, coding,
                                     self.stim_chs_array)
        events = _find_events(stim_ch, self.raw.first_samp, consecutive=True,
                              min_samples=3)
        return Counter(events[:, 2])

    def get_raw(self, preload=False):
        """Create a raw object based on the current model settings."""
        if not self.can_save:
            raise ValueError("Not all necessary parameters are set")

        # stim channels and coding
        if self.stim_coding == 'channel':
            stim_code = 'channel'
        elif self.stim_coding in '<>':
            stim_code = 'binary'
        else:
            raise RuntimeError("stim_coding=%r" % self.stim_coding)

        logger.info("Creating raw with stim=%r, slope=%r, stim_code=%r, "
                    "stimthresh=%r", self.stim_chs_array, self.stim_slope,
                    stim_code, self.stim_threshold)
        raw = RawKIT(self.sqd_file, preload=preload, stim=self.stim_chs_array,
                     slope=self.stim_slope, stim_code=stim_code,
                     stimthresh=self.stim_threshold,
                     allow_unknown_format=self.allow_unknown_format)

        if np.any(self.fid):
            raw.info['dig'] = _make_dig_points(self.fid[0], self.fid[1],
                                               self.fid[2], self.elp,
                                               self.hsp)
            raw.info['dev_head_t'] = Transform('meg', 'head',
                                               self.dev_head_trans)
        return raw


class Kit2FiffFrameHandler(Handler):
    """Check for unfinished processes before closing its window."""

    def close(self, info, is_ok):  # noqa: D102
        if info.object.kit2fiff_panel.queue.unfinished_tasks:
            msg = ("Can not close the window while saving is still in "
                   "progress. Please wait until all files are processed.")
            title = "Saving Still in Progress"
            information(None, msg, title)
            return False
        else:
            # store configuration, but don't prevent from closing on error
            try:
                info.object.save_config()
            except Exception as exc:
                warn("Error saving GUI configuration:\n%s" % (exc,))
            return True


class Kit2FiffPanel(HasPrivateTraits):
    """Control panel for kit2fiff conversion."""

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
    misc_chs_desc = DelegatesTo('model')
    can_test_stim = DelegatesTo('model')
    test_stim = Button(label="Find Events")
    plot_raw = Button(label="Plot Raw")

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
                      Item('hsp_file', label='Digitizer\nHead Shape',
                           tooltip=tooltips['hsp_file']),
                      Item('hsp_fname', show_label=False, style='readonly'),
                      Item('fid_file', label='Digitizer\nFiducials',
                           tooltip=tooltips['fid_file']),
                      Item('fid_fname', show_label=False, style='readonly'),
                      Item('reset_dig', label='Clear Digitizer Files',
                           show_label=False),
                      Item('use_mrk', editor=use_editor, style='custom',
                           tooltip=tooltips['use_mrk']),
                      label="Sources", show_border=True),
               VGroup(Item('misc_chs_desc', label='MISC Channels',
                           style='readonly'),
                      Item('stim_slope', label="Event Onset", style='custom',
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
                      Item('stim_chs_comment', label='Evaluation',
                           style='readonly', show_label=False),
                      Item('stim_threshold', label='Threshold',
                           tooltip=tooltips['stim_threshold']),
                      HGroup(Item('test_stim', enabled_when='can_test_stim',
                                  show_label=False),
                             Item('plot_raw', enabled_when='can_test_stim',
                                  show_label=False),
                             show_labels=False),
                      label='Events', show_border=True),
               HGroup(Item('save_as', enabled_when='can_save'), spring,
                      'clear_all', show_labels=False),
               Item('queue_feedback', show_label=False, style='readonly'),
               Item('queue_current', show_label=False, style='readonly'),
               Item('queue_len_str', show_label=False, style='readonly')
               )
    )

    def __init__(self, *args, **kwargs):  # noqa: D102
        super(Kit2FiffPanel, self).__init__(*args, **kwargs)

        # setup save worker
        def worker():  # noqa: D102
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
        self.fid_obj = PointObject(scene=self.scene, color=(0.1, 1., 0.1),
                                   point_scale=5e-3, name='Fiducials')
        self._update_fid()
        self.elp_obj = PointObject(scene=self.scene,
                                   color=(0.196, 0.196, 0.863),
                                   point_scale=1e-2, opacity=.2, name='ELP')
        self._update_elp()
        self.hsp_obj = PointObject(scene=self.scene, color=(0.784,) * 3,
                                   point_scale=2e-3, name='HSP')
        self._update_hsp()
        self.scene.camera.parallel_scale = 0.15
        self.scene.mlab.view(0, 0, .15)

    @on_trait_change('model:fid,model:head_dev_trans')
    def _update_fid(self):
        if self.fid_obj is not None:
            self.fid_obj.points = apply_trans(self.model.head_dev_trans,
                                              self.model.fid)

    @on_trait_change('model:hsp,model:head_dev_trans')
    def _update_hsp(self):
        if self.hsp_obj is not None:
            self.hsp_obj.points = apply_trans(self.model.head_dev_trans,
                                              self.model.hsp)

    @on_trait_change('model:elp,model:head_dev_trans')
    def _update_elp(self):
        if self.elp_obj is not None:
            self.elp_obj.points = apply_trans(self.model.head_dev_trans,
                                              self.model.elp)

    def _clear_all_fired(self):
        self.model.clear_all()

    @cached_property
    def _get_queue_len_str(self):
        if self.queue_len:
            return "Queue length: %i" % self.queue_len
        else:
            return ''

    def _plot_raw_fired(self):
        self.model.raw.plot()

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

    def _test_stim_fired(self):
        try:
            events = self.model.get_event_info()
        except Exception as err:
            error(None, "Error reading events from SQD data file: %s (Check "
                  "the terminal output for details)" % str(err),
                  "Error Reading events from SQD file")
            raise

        if len(events) == 0:
            information(None, "No events were found with the current "
                        "settings.", "No Events Found")
        else:
            lines = ["Events found (ID: n events):"]
            for id_ in sorted(events):
                lines.append("%3i: \t%i" % (id_, events[id_]))
            information(None, '\n'.join(lines), "Events in SQD File")


class Kit2FiffFrame(HasTraits):
    """GUI for interpolating between two KIT marker files."""

    model = Instance(Kit2FiffModel)
    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)
    marker_panel = Instance(CombineMarkersPanel)
    kit2fiff_panel = Instance(Kit2FiffPanel)

    view = View(HGroup(VGroup(Item('marker_panel', style='custom'),
                              show_labels=False),
                       VGroup(Item('scene',
                                   editor=SceneEditor(scene_class=MayaviScene),
                                   dock='vertical', show_label=False),
                              VGroup(Item('headview', style='custom'),
                                     show_labels=False),
                              ),
                       VGroup(Item('kit2fiff_panel', style='custom'),
                              show_labels=False),
                       show_labels=False,
                       ),
                handler=Kit2FiffFrameHandler(),
                height=700, resizable=True, buttons=NoButtons)

    def __init__(self, *args, **kwargs):  # noqa: D102
        logger.debug(
            "Initializing Kit2fiff-GUI with %s backend", ETSConfig.toolkit)
        HasTraits.__init__(self, *args, **kwargs)

    # can't be static method due to Traits
    def _model_default(self):
        # load configuration values and make sure they're valid
        config = get_config(home_dir=os.environ.get('_MNE_FAKE_HOME_DIR'))
        stim_threshold = 1.
        if 'MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD' in config:
            try:
                stim_threshold = float(
                    config['MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD'])
            except ValueError:
                warn("Ignoring invalid configuration value for "
                     "MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD: %r (expected "
                     "float)" %
                     (config['MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD'],))
        stim_slope = config.get('MNE_KIT2FIFF_STIM_CHANNEL_SLOPE', '-')
        if stim_slope not in '+-':
            warn("Ignoring invalid configuration value for "
                 "MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD: %s (expected + or -)" %
                 stim_slope)
            stim_slope = '-'
        stim_coding = config.get('MNE_KIT2FIFF_STIM_CHANNEL_CODING', '>')
        if stim_coding not in ('<', '>', 'channel'):
            warn("Ignoring invalid configuration value for "
                 "MNE_KIT2FIFF_STIM_CHANNEL_CODING: %s (expected <, > or "
                 "channel)" % stim_coding)
            stim_coding = '>'
        return Kit2FiffModel(
            stim_chs=config.get('MNE_KIT2FIFF_STIM_CHANNELS', ''),
            stim_coding=stim_coding,
            stim_slope=stim_slope,
            stim_threshold=stim_threshold,
            show_gui=True)

    def _headview_default(self):
        return HeadViewController(scene=self.scene, scale=160, system='RAS')

    def _kit2fiff_panel_default(self):
        return Kit2FiffPanel(scene=self.scene, model=self.model)

    def _marker_panel_default(self):
        return CombineMarkersPanel(scene=self.scene, model=self.model.markers,
                                   trans=als_ras_trans)

    def save_config(self, home_dir=None):
        """Write configuration values."""
        set_config('MNE_KIT2FIFF_STIM_CHANNELS', self.model.stim_chs, home_dir,
                   set_env=False)
        set_config('MNE_KIT2FIFF_STIM_CHANNEL_CODING', self.model.stim_coding,
                   home_dir, set_env=False)
        set_config('MNE_KIT2FIFF_STIM_CHANNEL_SLOPE', self.model.stim_slope,
                   home_dir, set_env=False)
        set_config('MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD',
                   str(self.model.stim_threshold), home_dir, set_env=False)
