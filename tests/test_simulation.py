import os
import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from particle_life_simulator.Class_simulation import Simulation

# Ensure headless mode for VisPy
os.environ["VISPY_USE_APP"] = "egl"

@pytest.fixture
def mock_simulation():
    """Creates a simulation with mocked GUI and particle creator"""
    mock_gui = MagicMock()
    mock_gui.update_fps = MagicMock()
    mock_gui.draw_particles = MagicMock()

    mock_particle_creator = MagicMock()
    mock_particle_creator.get_positions_and_colors.return_value = np.array([[0, 0, 1]])
    mock_particle_creator.update_positions = MagicMock()

    # Fully mock VisPy application to avoid GUI errors in CI
    with patch('vispy.app.use_app'), patch('vispy.app.Canvas'):
        sim = Simulation(particle_creator=mock_particle_creator, gui=mock_gui, benchmark_mode=False)

    sim.stop()  # Ensure the simulation is stopped initially

    return sim

@patch('vispy.app.use_app')
@patch('vispy.app.Canvas')
def test_simulation_start_stop(mock_canvas, mock_app, mock_simulation):
    """Tests if the simulation starts and stops correctly"""
    mock_simulation.start()
    assert mock_simulation.running  # Simulation should be running after start

    mock_simulation.stop()
    assert not mock_simulation.running  # Simulation should be stopped after stop

@patch('vispy.app.use_app')
@patch('vispy.app.Canvas')
def test_simulation_on_timer(mock_canvas, mock_app, mock_simulation):
    """Tests if the timer updates particles and GUI properly"""
    mock_simulation.last_time = time.perf_counter() - 1.1  # Simulate time elapsed

    event = MagicMock()
    mock_simulation.on_timer(event)

    # Verify that the particle positions were updated and the GUI was redrawn
    mock_simulation.particle_creator.update_positions.assert_called_once()
    mock_simulation.gui.draw_particles.assert_called_once()
    mock_simulation.gui.update_fps.assert_called_once()

@patch('vispy.app.use_app')
@patch('vispy.app.Canvas')
def test_simulation_benchmark_mode(mock_canvas, mock_app, mock_simulation):
    """Tests if the benchmark mode stops after 60 seconds"""
    mock_simulation.benchmark_mode = True
    mock_simulation.start_time = time.perf_counter() - 61  # Simulate 61 seconds elapsed
    mock_simulation.last_time = time.perf_counter() - 2  # Simulate time elapsed

    mock_simulation.on_timer(MagicMock())  # Trigger timer event

    # Verify that FPS data was collected and the simulation stopped
    assert len(mock_simulation.fps_list) > 0  # FPS data should be stored
    assert not mock_simulation.running  # Simulation should stop after benchmark duration
