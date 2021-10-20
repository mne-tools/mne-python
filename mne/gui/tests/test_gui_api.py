# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

def test_gui_api(renderer_interactive_pyvistaqt):
    def noop(x):
        return

    renderer = renderer_interactive_pyvistaqt._get_renderer(size=(300, 300))
    renderer._dock_initialize(name="Input", area="left")
    renderer._dock_add_text("text", "value", "placeholder")
    renderer._dock_add_file_button("file", "file", noop)
    renderer._dock_add_check_box("check", True, noop)
    renderer._dock_add_radio_buttons("radio", ["radio", "button"], noop)
    renderer.show()
    renderer.close()
