# -*- coding: utf-8 -*-
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: Simplified BSD

import pytest

from mne.viz import set_3d_backend
from mne.viz.backends.renderer import _get_backend
from mne.viz.backends.tests._utils import skips_if_not_pyvistaqt


def _setup_app(backend):
    # testing utils
    widget_checks = set()

    def callback(x=None):
        widget_checks.add('click' if x is None else x)

    window = backend._Application()
    central_layout = backend._VBoxLayout(scroll=(500, 500))
    canvas = backend._Canvas(5, 5, 96)
    canvas.ax.plot(range(10), range(10), label='plot')
    central_layout._add_widget(canvas)
    central_layout._add_widget(backend._Label('test'))
    text = backend._Text('test', 'placeholder', callback)
    central_layout._add_widget(text)
    button = backend._Button('test2', callback)
    central_layout._add_widget(button)
    slider = backend._Slider(0, (0, 100), callback)
    central_layout._add_widget(slider)
    checkbox = backend._CheckBox(0, callback)
    central_layout._add_widget(checkbox)
    spinbox = backend._SpinBox(10, (5, 50), callback, step=4)
    central_layout._add_widget(spinbox)
    combobox = backend._ComboBox('40', ('5', '50', '40'), callback)
    central_layout._add_widget(combobox)
    radio_buttons = backend._RadioButtons('40', ('5', '50', '40'), callback)
    central_layout._add_widget(radio_buttons)
    groupbox = backend._GroupBox(
        'menu', [backend._Button(f'{i}', callback) for i in range(5)])
    central_layout._add_widget(groupbox)
    groupbox._set_enabled(False)
    groupbox._set_enabled(True)
    file_button = backend._FileButton(callback, window=window)
    central_layout._add_widget(file_button)
    play_menu = backend._PlayMenu(0, (0, 100), callback)
    central_layout._add_widget(play_menu)
    window._add_keypress(callback)
    window._set_central_layout(central_layout)
    window._set_focus()
    window._show()
    # pop up
    popup = backend._Popup('Info', 'this is a message', 'test', callback)
    return (window, central_layout, text, button, slider, checkbox, spinbox,
            combobox, radio_buttons, file_button, play_menu, popup,
            widget_checks)


@skips_if_not_pyvistaqt
def test_widget_abstraction_pyvistaqt():
    """Test the GUI widgets abstraction."""
    from qtpy.QtCore import QTimer
    set_3d_backend('pyvistaqt')
    backend = _get_backend()

    (window, central_layout, text, button, slider, checkbox, spinbox, combobox,
     radio_buttons, file_button, play_menu, popup, widget_checks) = \
        _setup_app(backend)

    # first, test popup
    popup.button(popup.Ok).click()
    assert 'Ok' in widget_checks

    # test each widget
    text.setText('foo')
    assert 'foo' in widget_checks

    button.click()
    assert 'click' in widget_checks

    slider.setValue(10)
    assert 10 in widget_checks

    checkbox._set_checked(True)
    assert True in widget_checks

    spinbox.setValue(20)
    assert 20 in widget_checks

    combobox._set_value('5')
    assert '5' in widget_checks

    radio_buttons._set_value('50')
    assert '50' in widget_checks

    # this was tested manually but creates a blocking window so can't
    # be tested here
    # file_button.click()
    assert hasattr(file_button, 'click')

    play_menu._play.click()
    assert 1 in widget_checks

    # timer done separately because multithreading is not allowed in qt
    progress_bar = backend._ProgressBar(100)
    timer = QTimer()
    timer.timeout.connect(progress_bar ._increment)
    timer.setInterval(250)
    timer.start()
    central_layout._add_widget(progress_bar)


@pytest.mark.slowtest
def test_widget_abstraction_notebook(nbexec):
    """Test the GUI widgets abstraction in notebook."""
    from mne.viz import set_3d_backend
    from mne.viz.backends.renderer import _get_backend
    from mne.viz.backends.tests.test_abstract import _setup_app
    from IPython import get_ipython
    import threading

    set_3d_backend('notebook')
    backend = _get_backend()

    ipython = get_ipython()
    ipython.magic('%matplotlib widget')

    (window, central_layout, text, button, slider, checkbox, spinbox, combobox,
     radio_buttons, file_button, play_menu, popup, widget_checks) = \
        _setup_app(backend)

    # first, test popup
    popup._buttons['Ok'].click()
    assert 'Ok' in widget_checks

    # test each widget
    text.value = 'foo'
    assert 'foo' in widget_checks

    button.click()
    assert 'click' in widget_checks

    slider.value = 10
    assert 10 in widget_checks

    checkbox._set_checked(True)
    assert True in widget_checks

    spinbox.value = 20
    assert 20 in widget_checks

    combobox._set_value('5')
    assert '5' in widget_checks

    radio_buttons._set_value('50')
    assert '50' in widget_checks

    # this was tested manually but creates a blocking window so can't
    # be tested here
    # file_button.click()
    assert hasattr(file_button, 'click')

    play_menu._play.value = 1
    assert 1 in widget_checks

    # timer done separately because multithreading is not allowed in qt
    progress_bar = backend._ProgressBar(100)

    def update_progress_bar():
        progress_bar._increment()
        threading.Timer(0.25, update_progress_bar).start()

    update_progress_bar()
    central_layout._add_widget(progress_bar)
