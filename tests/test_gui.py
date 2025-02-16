import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from vispy.scene import Node
from particle_life_simulator.Class_GUI import GUI


class MockSceneCanvas:
    """Mock for VisPy SceneCanvas to prevent actual window creation."""
    
    def __init__(self, *args, **kwargs):
        self.central_widget = MagicMock()
        self.central_widget.add_view.return_value = MagicMock()
        self.central_widget.add_view.return_value.scene = Node()  # Using real Node instance for Markers

        self.events = MagicMock()
        self.events.mouse_release = MagicMock()

    def close(self):
        pass


@pytest.fixture
def create_mocked_gui():
    """Creates a GUI instance with mocked dependencies."""
    with patch("particle_life_simulator.Class_GUI.app.quit"), \
         patch("particle_life_simulator.Class_GUI.scene.SceneCanvas", MockSceneCanvas), \
         patch("particle_life_simulator.Class_GUI.tk.Tk", MagicMock()):
        
        return GUI(window_width=1920, window_height=1080, particle_size=5)


def test_gui_initialization(create_mocked_gui):
    """Tests whether the GUI initializes with correct default values."""
    gui = create_mocked_gui
    assert gui.window_width == 1920
    assert gui.window_height == 1080
    assert gui.particle_size == 5
    assert isinstance(gui.color_lookup, dict)
    assert gui.scatter is not None  


def test_gui_fps_update(create_mocked_gui):
    """Checks if the FPS label updates correctly."""
    gui = create_mocked_gui
    gui.update_fps(60.5)
    
    # FPS should be formatted with two decimal places
    assert gui.fps_label.text == "FPS: 60.50"


def test_gui_draw_particles(create_mocked_gui):
    """Verifies that draw_particles correctly processes and updates particle data."""
    gui = create_mocked_gui
    particles = np.array([[100, 200, 1], [300, 400, 2]], dtype=np.float32)

    gui.scatter.set_data = MagicMock()  
    gui.draw_particles(particles, num_particles=2)

    gui.scatter.set_data.assert_called_once() 


def test_gui_draw_particles_invalid_data(create_mocked_gui):
    """Ensures draw_particles does not crash with missing color data."""
    gui = create_mocked_gui
    invalid_particles = np.array([[100, 200], [300, 400]], dtype=np.float32)

    gui.scatter.set_data = MagicMock()

    try:
        gui.draw_particles(invalid_particles, num_particles=2)
    except Exception as e:
        pytest.fail(f"draw_particles() raised an unexpected exception: {e}")


def test_gui_stop_simulation(create_mocked_gui):
    """Checks if stop_simulation correctly closes the GUI."""
    gui = create_mocked_gui
    gui.canvas.close = MagicMock()
    gui.stop_simulation()

    gui.canvas.close.assert_called_once()
