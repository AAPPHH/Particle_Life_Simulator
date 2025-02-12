import pytest
import time
import numpy as np
from unittest.mock import MagicMock
from particle_life_simulator.Class_simulation import Simulation


@pytest.fixture
def mock_simulation():
    """Creates a simulation with mocked GUI and particle creator"""
    mock_gui = MagicMock()
    mock_gui.update_fps = MagicMock()
    mock_gui.draw_particles = MagicMock()

    mock_particle_creator = MagicMock()
    mock_particle_creator.get_positions_and_colors.return_value = np.array([[0, 0, 1]])
    mock_particle_creator.update_positions = MagicMock()

    sim = Simulation(particle_creator=mock_particle_creator, gui=mock_gui, benchmark_mode=False)
    sim.running = False  
    
    return sim


def test_simulation_start_stop(mock_simulation):
    """Tests if the simulation starts and stops correctly"""
    mock_simulation.start()
    assert mock_simulation.running  # Should be running

    mock_simulation.stop()
    assert not mock_simulation.running  # Should be stopped


def test_simulation_on_timer(mock_simulation):
    """Tests if the timer updates particles and GUI properly"""
    mock_simulation.last_time = time.perf_counter() - 1.1  

    event = MagicMock()
    mock_simulation.on_timer(event)

    mock_simulation.particle_creator.update_positions.assert_called_once()
    mock_simulation.gui.draw_particles.assert_called_once()
    mock_simulation.gui.update_fps.assert_called_once()


def test_simulation_benchmark_mode(mock_simulation):
    """Tests if the benchmark mode stops after 60 seconds"""
    mock_simulation.benchmark_mode = True
    mock_simulation.start_time = time.perf_counter() - 61  # Simulate 61 seconds runtime

    mock_simulation.on_timer(MagicMock())  # Trigger timer event
    assert len(mock_simulation.fps_list) > 0  # FPS data should be stored
