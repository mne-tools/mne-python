# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD
import numpy as np
from ...utils import warn


class _LinkViewer(object):
    """Class to link multiple Brain objects."""

    def __init__(self, brains, time=True, camera=False, colorbar=True,
                 picking=False):
        self.brains = brains
        self.leader = self.brains[0]  # select a brain as leader

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
            for brain in self.brains:
                brain.actions["play"].triggered.disconnect()
                brain.actions["play"].triggered.connect(
                    self.toggle_playback)

            # link time course canvas
            def _time_func(*args, **kwargs):
                for brain in self.brains:
                    brain.callbacks["time"](*args, **kwargs)

            for brain in self.brains:
                if brain.show_traces:
                    brain.mpl_canvas.time_func = _time_func

        if picking:
            def _func_add(*args, **kwargs):
                for brain in self.brains:
                    brain._add_vertex_glyph2(*args, **kwargs)
                    brain.plotter.update()

            def _func_remove(*args, **kwargs):
                for brain in self.brains:
                    brain._remove_vertex_glyph2(*args, **kwargs)

            # save initial picked points
            initial_points = dict()
            for hemi in ('lh', 'rh'):
                initial_points[hemi] = set()
                for brain in self.brains:
                    initial_points[hemi] |= \
                        set(brain.picked_points[hemi])

            # link the viewers
            for brain in self.brains:
                brain.clear_glyphs()
                brain._add_vertex_glyph2 = brain._add_vertex_glyph
                brain._add_vertex_glyph = _func_add
                brain._remove_vertex_glyph2 = brain._remove_vertex_glyph
                brain._remove_vertex_glyph = _func_remove

            # link the initial points
            for hemi in initial_points.keys():
                if hemi in brain._layered_meshes:
                    mesh = brain._layered_meshes[hemi]._polydata
                    for vertex_id in initial_points[hemi]:
                        self.leader._add_vertex_glyph(hemi, mesh, vertex_id)

        if colorbar:
            fmin = self.leader._data["fmin"]
            fmid = self.leader._data["fmid"]
            fmax = self.leader._data["fmax"]
            for brain in self.brains:
                brain.callbacks["fmin"](fmin)
                brain.callbacks["fmid"](fmid)
                brain.callbacks["fmax"](fmax)

            for slider_name in ('fmin', 'fmid', 'fmax'):
                func = getattr(self, "set_" + slider_name)
                self.link_sliders(
                    name=slider_name,
                    callback=func,
                    event_type="always"
                )

    def set_fmin(self, value):
        for brain in self.brains:
            brain.callbacks["fmin"](value)

    def set_fmid(self, value):
        for brain in self.brains:
            brain.callbacks["fmid"](value)

    def set_fmax(self, value):
        for brain in self.brains:
            brain.callbacks["fmax"](value)

    def set_time_point(self, value):
        for brain in self.brains:
            brain.callbacks["time"](value, update_widget=True)

    def set_playback_speed(self, value):
        for brain in self.brains:
            brain.callbacks["playback_speed"](value, update_widget=True)

    def toggle_playback(self):
        value = self.leader.callbacks["time"].slider_rep.GetValue()
        # synchronize starting points before playback
        self.set_time_point(value)
        for brain in self.brains:
            brain.toggle_playback()

    def link_sliders(self, name, callback, event_type):
        from ..backends._pyvista import _update_slider_callback
        for brain in self.brains:
            slider = brain.sliders[name]
            if slider is not None:
                _update_slider_callback(
                    slider=slider,
                    callback=callback,
                    event_type=event_type
                )

    def link_cameras(self):
        from ..backends._pyvista import _add_camera_callback

        def _update_camera(vtk_picker, event):
            for brain in self.brains:
                brain.plotter.update()

        camera = self.leader.plotter.camera
        _add_camera_callback(camera, _update_camera)
        for brain in self.brains:
            for renderer in brain.plotter.renderers:
                renderer.camera = camera
