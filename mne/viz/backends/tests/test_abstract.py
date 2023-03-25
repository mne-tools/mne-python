# -*- coding: utf-8 -*-
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: Simplified BSD

from pathlib import Path
import sys

import pytest

from mne.viz.backends.renderer import _get_backend
from mne.viz.backends.tests._utils import skips_if_not_pyvistaqt

from mne.utils import check_version


def _do_widget_tests(backend):
    # testing utils
    widget_checks = set()

    def callback(x=None):
        add = x
        if add is None:
            add = 'click'
        elif isinstance(add, str):
            add = add.lstrip('&')  # new notebooks can add this
        widget_checks.add(add)

    window = backend._AppWindow()
    central_layout = backend._VBoxLayout(scroll=(500, 500))
    renderer = backend._3DRenderer(name='test', size=(200, 200))
    renderer.sphere([0, 0, 0], 'red', 1)
    central_layout._add_widget(renderer.plotter)
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
    progress_bar = backend._ProgressBar(100)
    progress_bar._increment()
    central_layout._add_widget(progress_bar)
    window._add_keypress(callback)
    window._set_central_layout(central_layout)
    window._set_focus()
    window._show()
    # pop up
    popup = backend._Popup('Info', 'this is a message', 'test', callback)

    # do tests
    # first, test popup
    popup._click('Ok')
    assert 'Ok' in widget_checks

    window._trigger_keypress('a')
    assert 'a' in widget_checks
    window._trigger_keypress('escape')
    assert 'escape' in widget_checks

    # test each widget
    text._set_value('foo')
    assert 'foo' in widget_checks

    button._click()
    assert 'click' in widget_checks

    slider._set_value(10)
    assert 10 in widget_checks

    checkbox._set_checked(True)
    assert True in widget_checks

    spinbox._set_value(20)
    assert 20 in widget_checks

    combobox._set_value('5')
    assert '5' in widget_checks

    radio_buttons._set_value('50')
    assert '50' in widget_checks

    # this was tested manually but creates a blocking window so can't
    # be tested here
    # file_button.click()
    assert hasattr(file_button, 'click')

    play_menu._set_value(99)
    assert 99 in widget_checks

    window._close()


@skips_if_not_pyvistaqt
def test_widget_abstraction_pyvistaqt(renderer_pyvistaqt):
    """Test the GUI widgets abstraction."""
    backend = _get_backend()
    assert Path(backend.__file__).stem == '_qt'
    _do_widget_tests(backend)


nb_skip_mark = pytest.mark.skipif(
    sys.platform.startswith('win') or not check_version('ipympl'),
    reason='need ipympl and nbexec does not work on Windows')


# Marking directly with skipif causes problems for nbexec, so let's get it in
# via a param
@pytest.mark.parametrize('skippy', [pytest.param('', marks=nb_skip_mark)])
def test_widget_abstraction_notebook(renderer_notebook, nbexec, skippy):
    """Test the GUI widgets abstraction in notebook."""
    from pathlib import Path
    from mne.viz import set_3d_backend
    from mne.viz.backends.renderer import _get_backend
    from mne.viz.backends.tests.test_abstract import _do_widget_tests
    from IPython import get_ipython

    set_3d_backend('notebook')
    backend = _get_backend()
    assert Path(backend.__file__).stem == '_notebook'

    ipython = get_ipython()
    ipython.magic('%matplotlib widget')
    _do_widget_tests(backend)
