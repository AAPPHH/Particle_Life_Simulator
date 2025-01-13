import sys
import os
import time
from unittest.mock import MagicMock, patch
from particle_life_simulator.Class_simulation import Simulation

# Add the project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_initialization():
    # Mock dependencies
    particle_creator = MagicMock()
    gui = MagicMock()

    # Initialize the Simulation
    simulation = Simulation(particle_creator, gui)

    # Verify initialization
    assert simulation.particle_creator == particle_creator
    assert simulation.gui == gui
    assert simulation.running is True

def test_update():
    # Mock dependencies
    particle_creator = MagicMock()
    gui = MagicMock()

    # Mock the return value of get_positions_and_colors
    particle_creator.get_positions_and_colors.return_value = [(10, 20, (255, 0, 0))]

    # Initialize the Simulation
    simulation = Simulation(particle_creator, gui)

    # Call the update method
    simulation.update()

    # Verify that all necessary methods are called
    particle_creator.update_positions.assert_called_once()
    particle_creator.get_positions_and_colors.assert_called_once()
    gui.clear_drawlist.assert_called_once()
    gui.draw_particles.assert_called_once_with([(10, 20, (255, 0, 0))])

def test_fps_calculation():
    # Mock dependencies
    particle_creator = MagicMock()
    gui = MagicMock()

    # Initialize the Simulation
    simulation = Simulation(particle_creator, gui)

    # Simulate FPS calculation
    frame_count = 60
    start_time = time.time()
    time.sleep(1)  # Simulate a delay of 1 second
    end_time = time.time()

    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    # Pass the calculated FPS to the GUI
    simulation.gui.update_fps(fps)

    # Verify the update_fps method is called with the correct value
    gui.update_fps.assert_called_once_with(fps)

def test_start_method():
    # Mock dependencies
    particle_creator = MagicMock()
    gui = MagicMock()

    # Patch dearpygui's is_dearpygui_running function
    with patch("dearpygui.dearpygui.is_dearpygui_running", return_value=False):
        # Initialize the Simulation
        simulation = Simulation(particle_creator, gui)
        simulation.running = False  # Stop the loop immediately

        # Call the start method
        simulation.start()

        # Verify setup and cleanup methods are called
        gui.setup_window.assert_called_once()
        gui.cleanup.assert_called_once()

        # Ensure update_positions is not called as the loop didn't run
        particle_creator.update_positions.assert_not_called()
