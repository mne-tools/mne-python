# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


def test_gui_api(renderer_pyvistaqt):
    """Test GUI API."""
    from unittest.mock import Mock
    renderer = renderer_pyvistaqt._get_renderer(size=(300, 300))

    # --- BEGIN: dock ---
    renderer._dock_initialize(name='', area='left')

    # label
    widget = renderer._dock_add_label('', align=True)
    widget.update()
    widget.set_enabled(False)

    # button
    mock = Mock()
    renderer._dock_add_button('', mock)
    assert mock.call_count == 0
    # XXX: trigger the click with set_value(True) then:
    # assert mock.call_count == 1

    # slider
    mock = Mock()
    widget = renderer._dock_add_slider('', 0, [0, 10], mock)
    assert widget.get_value() == 0
    assert mock.call_count == 0
    widget.set_value(5)
    assert widget.get_value() == 5
    assert mock.call_count == 1

    # check box
    mock = Mock()
    widget = renderer._dock_add_check_box('', False, mock)
    assert not widget.get_value()
    assert mock.call_count == 0
    widget.set_value(True)  # change value to trigger the callback
    assert widget.get_value()
    assert mock.call_count == 1

    # spin box
    mock = Mock()
    renderer._dock_add_spin_box('', 0, [0, 1], mock, step=0.1)
    widget = renderer._dock_add_spin_box('', 0, [0, 1], mock, step=None)
    assert widget.get_value() == 0
    assert mock.call_count == 0
    widget.set_value(0.5)
    assert widget.get_value() == 0.5
    assert mock.call_count == 1

    # combo box
    mock = Mock()
    widget = renderer._dock_add_combo_box('', 'foo', ['foo', 'bar'], mock)
    assert widget.get_value() == 'foo'
    assert mock.call_count == 0
    widget.set_value('bar')
    assert widget.get_value() == 'bar'
    assert mock.call_count == 1

    # radio buttons
    mock = Mock()
    widget = renderer._dock_add_radio_buttons('foo', ['foo', 'bar'], mock)
    assert widget.get_value(0)
    assert mock.call_count == 0
    widget.set_value(1, True)
    assert widget.get_value(1)
    assert mock.call_count == 1

    # text field
    widget = renderer._dock_add_text('', 'foo', '')
    assert widget.get_value() == 'foo'
    widget.set_value('bar')
    assert widget.get_value() == 'bar'

    # file button
    mock = Mock()
    renderer._dock_add_file_button('', '', mock, directory=True)
    renderer._dock_add_file_button('', '', mock, save=True)
    widget = renderer._dock_add_file_button('', '', mock,
                                            input_text_widget=False)
    assert mock.call_count == 0
    # XXX: the internal file dialogs may hang without signals

    renderer._dock_initialize(name='', area='right')
    renderer._dock_named_layout(name='')
    renderer._dock_add_group_box(name='')
    renderer._dock_add_stretch()
    renderer._dock_add_layout()
    renderer._dock_finalize()
    renderer._dock_hide()
    renderer._dock_show()
    # --- END: dock ---

    renderer.show()
    renderer.close()
