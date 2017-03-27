import os
from collections import defaultdict

from os.path import join as pjoin

import numpy.testing as npt

from dipy.data import read_viz_icons, fetch_viz_icons
from dipy.viz import ui
from dipy.viz import window
from dipy.data import DATA_DIR

from dipy.viz.ui import UI

from dipy.testing.decorators import xvfb_it

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
if use_xvfb == 'skip':
    skip_it = True
else:
    skip_it = False


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_ui(recording=False):
    print("Using VTK {}".format(vtk.vtkVersion.GetVTKVersion()))
    filename = "test_ui.log.gz"
    recording_filename = pjoin(DATA_DIR, filename)

    # Define some counter callback.
    states = defaultdict(lambda: 0)

    # Broken UI Element
    class BrokenUI(UI):

        def __init__(self):
            self.actor = vtk.vtkActor()
            super(BrokenUI, self).__init__()

        def add_callback(self, event_type, callback):
            """ Adds events to an actor.

            Parameters
            ----------
            event_type : string
                event code
            callback : function
                callback function
            """
            super(BrokenUI, self).add_callback(self.actor, event_type, callback)

    broken_ui = BrokenUI()
    npt.assert_raises(NotImplementedError, broken_ui.get_actors)
    npt.assert_raises(NotImplementedError, broken_ui.set_center, (1, 2))
    # /Broken UI Element

    # Rectangle
    rectangle_test = ui.Rectangle2D(size=(10, 10))
    rectangle_test.get_actors()
    another_rectangle_test = ui.Rectangle2D(size=(1, 1))
    # /Rectangle

    # Button
    fetch_viz_icons()

    icon_files = dict()
    icon_files['stop'] = read_viz_icons(fname='stop2.png')
    icon_files['play'] = read_viz_icons(fname='play3.png')

    button_test = ui.Button2D(icon_fnames=icon_files)
    button_test.set_center((20, 20))

    def counter(i_ren, obj, button):
        states[i_ren.event.name] += 1

    # Assign the counter callback to every possible event.
    for event in ["CharEvent", "MouseMoveEvent",
                  "KeyPressEvent", "KeyReleaseEvent",
                  "LeftButtonPressEvent", "LeftButtonReleaseEvent",
                  "RightButtonPressEvent", "RightButtonReleaseEvent",
                  "MiddleButtonPressEvent", "MiddleButtonReleaseEvent"]:
        button_test.add_callback(button_test.actor, event, counter)

    def make_invisible(i_ren, obj, button):
        # i_ren: CustomInteractorStyle
        # obj: vtkActor picked
        # button: Button2D
        button.set_visibility(False)
        i_ren.force_render()
        i_ren.event.abort()

    def modify_button_callback(i_ren, obj, button):
        # i_ren: CustomInteractorStyle
        # obj: vtkActor picked
        # button: Button2D
        button.next_icon()
        i_ren.force_render()

    button_test.on_right_mouse_button_pressed = make_invisible
    button_test.on_left_mouse_button_pressed = modify_button_callback

    button_test.scale((2, 2))
    button_color = button_test.color
    button_test.color = button_color
    # /Button

    # TextBox
    textbox_test = ui.TextBox2D(height=3, width=10, text="Text")
    textbox_test.set_message("Enter Text")
    textbox_test.set_center((10, 100))

    another_textbox_test = ui.TextBox2D(height=3, width=10, text="Enter Text")

    # /TextBox

    # Panel
    panel = ui.Panel2D(center=(440, 90), size=(300, 150), color=(1, 1, 1), align="right")
    panel.add_element(rectangle_test, 'absolute', (580, 150))
    panel.add_element(button_test, 'relative', (0.2, 0.2))
    npt.assert_raises(ValueError, panel.add_element, another_rectangle_test, 'error_string', (1, 2))
    # /Panel

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title="DIPY UI Example")

    show_manager.ren.add(panel)
    show_manager.ren.add(another_textbox_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(states.items()))
    else:
        show_manager.play_events_from_file(recording_filename)
        msg = "Wrong count for '{}'."
        expected = [('CharEvent', 0),
                    ('KeyPressEvent', 0),
                    ('KeyReleaseEvent', 0),
                    ('MouseMoveEvent', 451),
                    ('LeftButtonPressEvent', 21),
                    ('RightButtonPressEvent', 4),
                    ('MiddleButtonPressEvent', 0),
                    ('LeftButtonReleaseEvent', 21),
                    ('MouseWheelForwardEvent', 0),
                    ('MouseWheelBackwardEvent', 0),
                    ('MiddleButtonReleaseEvent', 0),
                    ('RightButtonReleaseEvent', 4)]

        # Useful loop for debugging.
        for event, count in expected:
            if states[event] != count:
                print("{}: {} vs. {} (expected)".format(event,
                                                        states[event],
                                                        count))

        for event, count in expected:
            npt.assert_equal(states[event], count, err_msg=msg.format(event))

        # Dummy Show Manager
        dummy_renderer = window.Renderer()
        dummy_show_manager = window.ShowManager(dummy_renderer, size=(800, 800), reset_camera=False,
                                                interactor_style='trackball')
        npt.assert_raises(TypeError, button_test.add_to_renderer, dummy_renderer)
        # /Dummy Show Manager


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_text_actor_2d():
    # TextActor2D
    text_actor = ui.TextActor2D()
    text_actor.message = "Hello World!"
    npt.assert_equal("Hello World!", text_actor.message)
    text_actor.font_size = 18
    npt.assert_equal("18", str(text_actor.font_size))
    text_actor.font_family = "Arial"
    npt.assert_equal("Arial", text_actor.font_family)
    with npt.assert_raises(ValueError):
        text_actor.font_family = "Verdana"
    text_actor.justification = "left"
    text_actor.justification = "right"
    text_actor.justification = "center"
    npt.assert_equal("Centered", text_actor.justification)
    with npt.assert_raises(ValueError):
        text_actor.justification = "bottom"
    text_actor.bold = True
    text_actor.bold = False
    npt.assert_equal(False, text_actor.bold)
    text_actor.italic = True
    text_actor.italic = False
    npt.assert_equal(False, text_actor.italic)
    text_actor.shadow = True
    text_actor.shadow = False
    npt.assert_equal(False, text_actor.shadow)
    text_actor.color = (1, 0, 0)
    npt.assert_equal((1, 0, 0), text_actor.color)
    text_actor.position = (2, 3)
    npt.assert_equal((2, 3), text_actor.position)
    # /TextActor2D
