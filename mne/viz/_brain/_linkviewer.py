# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ...utils import warn
from ..ui_events import link


class _LinkViewer:
    """Class to link multiple Brain objects."""

    def __init__(self, brains, time=True, camera=False, colorbar=True, picking=False):
        self.brains = brains
        self.leader = self.brains[0]  # select a brain as leader
        self._camera_observer = None

        # check time infos
        times = [brain._times for brain in brains]
        if time and not all(np.allclose(x, times[0]) for x in times):
            warn("stc.times do not match, not linking time")
            time = False

        if camera:
            self.link_cameras()

        # A camera observer (below) holds a reference back to this
        # _LinkViewer (and hence to all linked brains) that gc.collect()
        # cannot resolve on its own: VTK's AddObserver keeps the callback
        # alive from the C side, outside of Python's reference-cycle
        # tracking. So unlink it whenever any linked brain closes (harmless
        # no-op if camera linking was never enabled).
        for brain in brains:
            brain._renderer._window_close_connect(self._unlink_camera)

        events_to_link = []
        if time:
            events_to_link.append("time_change")
        if colorbar:
            events_to_link.append("colormap_range")

        for brain in brains[1:]:
            link(self.leader, brain, include_events=events_to_link)

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
            for hemi in ("lh", "rh"):
                initial_points[hemi] = set()
                for brain in self.brains:
                    initial_points[hemi] |= set(brain.get_picked_points()[hemi])

            # link the viewers
            for brain in self.brains:
                brain.clear_glyphs()
                brain._add_vertex_glyph2 = brain._add_vertex_glyph
                brain._add_vertex_glyph = _func_add
                brain._remove_vertex_glyph2 = brain._remove_vertex_glyph
                brain._remove_vertex_glyph = _func_remove

            # link the initial points
            for hemi in initial_points.keys():
                if hemi in brain.layered_meshes:
                    mesh = brain.layered_meshes[hemi]._polydata
                    for vertex_id in initial_points[hemi]:
                        self.leader._add_vertex_glyph(hemi, mesh, vertex_id)

    def set_fmin(self, value):
        self.leader.update_lut(fmin=value)

    def set_fmid(self, value):
        self.leader.update_lut(fmid=value)

    def set_fmax(self, value):
        self.leader.update_lut(fmax=value)

    def set_time_point(self, value):
        self.leader.set_time_point(value)

    def set_playback_speed(self, value):
        self.leader.set_playback_speed(value)

    def toggle_playback(self):
        self.leader.toggle_playback()

    def link_cameras(self):
        from ..backends._pyvista import _add_camera_callback

        def _update_camera(vtk_picker, event):
            for brain in self.brains:
                brain.plotter.update()

        camera = self.leader.plotter.camera
        tag = _add_camera_callback(camera, _update_camera)
        self._camera_observer = (camera, tag)
        for brain in self.brains:
            for renderer in brain.plotter.renderers:
                renderer.camera = camera

    def _unlink_camera(self):
        if self._camera_observer is not None:
            camera, tag = self._camera_observer
            camera.RemoveObserver(tag)
            self._camera_observer = None
