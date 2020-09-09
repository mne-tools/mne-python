# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD
import numpy as np
from ...utils import warn


class _LinkViewer(object):
    """Class to link multiple _TimeViewer objects."""

    def __init__(self, brains, time=True, camera=False, colorbar=True,
                 picking=False):
        self.brains = brains
        self.time_viewers = [brain.time_viewer for brain in brains]

        # check time infos
        times = [brain._times for brain in brains]
        if time and not all(np.allclose(x, times[0]) for x in times):
            warn('stc.times do not match, not linking time')
            time = False

        if camera:
            self.link_cameras()

        if time:
            # link time sliders
            self.link_sliders(
                name="time",
                callback=self.set_time_point,
                event_type="always"
            )

            # link playback speed sliders
            self.link_sliders(
                name="playback_speed",
                callback=self.set_playback_speed,
                event_type="always"
            )

            # link toggle to start/pause playback
            for time_viewer in self.time_viewers:
                time_viewer.actions["play"].triggered.disconnect()
                time_viewer.actions["play"].triggered.connect(
                    self.toggle_playback)

            # link time course canvas
            def _time_func(*args, **kwargs):
                for time_viewer in self.time_viewers:
                    time_viewer.callbacks["time"](*args, **kwargs)

            for time_viewer in self.time_viewers:
                if time_viewer.show_traces:
                    time_viewer.mpl_canvas.time_func = _time_func

        if picking:
            def _func_add(*args, **kwargs):
                for time_viewer in self.time_viewers:
                    time_viewer._add_point(*args, **kwargs)
                    time_viewer.plotter.update()

            def _func_remove(*args, **kwargs):
                for time_viewer in self.time_viewers:
                    time_viewer._remove_point(*args, **kwargs)

            # save initial picked points
            initial_points = dict()
            for hemi in ('lh', 'rh'):
                initial_points[hemi] = set()
                for time_viewer in self.time_viewers:
                    initial_points[hemi] |= \
                        set(time_viewer.picked_points[hemi])

            # link the viewers
            for time_viewer in self.time_viewers:
                time_viewer.clear_points()
                time_viewer._add_point = time_viewer.add_point
                time_viewer.add_point = _func_add
                time_viewer._remove_point = time_viewer.remove_point
                time_viewer.remove_point = _func_remove

            # link the initial points
            leader = self.time_viewers[0]  # select a time_viewer as leader
            for hemi in initial_points.keys():
                if hemi in time_viewer.brain._hemi_meshes:
                    mesh = time_viewer.brain._hemi_meshes[hemi]
                    for vertex_id in initial_points[hemi]:
                        leader.add_point(hemi, mesh, vertex_id)

        if colorbar:
            leader = self.time_viewers[0]  # select a time_viewer as leader
            fmin = leader.brain._data["fmin"]
            fmid = leader.brain._data["fmid"]
            fmax = leader.brain._data["fmax"]
            for time_viewer in self.time_viewers:
                time_viewer.callbacks["fmin"](fmin)
                time_viewer.callbacks["fmid"](fmid)
                time_viewer.callbacks["fmax"](fmax)

            for slider_name in ('fmin', 'fmid', 'fmax'):
                func = getattr(self, "set_" + slider_name)
                self.link_sliders(
                    name=slider_name,
                    callback=func,
                    event_type="always"
                )

    def set_fmin(self, value):
        for time_viewer in self.time_viewers:
            time_viewer.callbacks["fmin"](value)

    def set_fmid(self, value):
        for time_viewer in self.time_viewers:
            time_viewer.callbacks["fmid"](value)

    def set_fmax(self, value):
        for time_viewer in self.time_viewers:
            time_viewer.callbacks["fmax"](value)

    def set_time_point(self, value):
        for time_viewer in self.time_viewers:
            time_viewer.callbacks["time"](value, update_widget=True)

    def set_playback_speed(self, value):
        for time_viewer in self.time_viewers:
            time_viewer.callbacks["playback_speed"](value, update_widget=True)

    def toggle_playback(self):
        leader = self.time_viewers[0]  # select a time_viewer as leader
        value = leader.callbacks["time"].slider_rep.GetValue()
        # synchronize starting points before playback
        self.set_time_point(value)
        for time_viewer in self.time_viewers:
            time_viewer.toggle_playback()

    def link_sliders(self, name, callback, event_type):
        from ..backends._pyvista import _update_slider_callback
        for time_viewer in self.time_viewers:
            slider = time_viewer.sliders[name]
            if slider is not None:
                _update_slider_callback(
                    slider=slider,
                    callback=callback,
                    event_type=event_type
                )

    def link_cameras(self):
        from ..backends._pyvista import _add_camera_callback

        def _update_camera(vtk_picker, event):
            for time_viewer in self.time_viewers:
                time_viewer.plotter.update()

        leader = self.time_viewers[0]  # select a time_viewer as leader
        camera = leader.plotter.camera
        _add_camera_callback(camera, _update_camera)
        for time_viewer in self.time_viewers:
            for renderer in time_viewer.plotter.renderers:
                renderer.camera = camera
