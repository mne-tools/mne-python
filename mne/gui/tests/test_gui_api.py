# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

def test_gui_api(renderer_interactive_pyvistaqt):
    """Test GUI API."""
    def noop(x):
        return

    renderer = renderer_interactive_pyvistaqt._get_renderer(size=(300, 300))
    renderer._dock_initialize(name='', area='left')
    renderer._dock_add_text('', '', '')
    renderer._dock_add_file_button('', '', noop, directory=True)
    renderer._dock_add_file_button('', '', noop, save=True)
    renderer._dock_add_file_button('', '', noop,
                                   input_text_widget=False)
    renderer._dock_initialize(name='', area='right')
    renderer._dock_add_spin_box('', 0, [0, 1], noop, step=None)
    renderer._dock_add_spin_box('', 0, [0, 1], noop, step=0.1)
    widget = renderer._dock_add_check_box('', True, noop)
    widget.set_value(True)
    widget.set_enabled(False)
    widget = renderer._dock_add_radio_buttons('foo', ['foo', 'bar'], noop)
    widget.set_value(0, True)
    widget.set_enabled(False)
    renderer.show()
    renderer.close()
