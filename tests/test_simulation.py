import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from vispy import app
from particle_life_simulator.Class_simulation import Simulation


@pytest.fixture
def simulation_with_mocks():
    """
    Creates a simulation instance with mocked GUI and particle creator.
    The Timer object is also mocked to prevent GUI-related errors in CI environments.
    """
    mock_particle_creator = MagicMock()
    mock_particle_creator.get_positions_and_colors.return_value = np.array([[0, 0, 1]])
    mock_particle_creator.update_positions = MagicMock()

    mock_gui = MagicMock()
    mock_gui.update_fps = MagicMock()
    mock_gui.draw_particles = MagicMock()

    # Mock VisPy's Timer to avoid actual GUI interactions in CI
    with patch('vispy.app.Timer') as MockTimer:
        mock_timer = MockTimer.return_value
        mock_timer.start = MagicMock()
        mock_timer.stop = MagicMock()
        
        simulation = Simulation(
            particle_creator=mock_particle_creator,
            gui=mock_gui,
            benchmark_mode=False
        )
        
    return simulation


def test_simulation_start_stop(simulation_with_mocks):
    """Tests whether the simulation starts and stops correctly."""
    simulation_with_mocks.start()
    assert simulation_with_mocks.running  # The simulation should be running after start

    simulation_with_mocks.stop()
    assert not simulation_with_mocks.running  # The simulation should be stopped after stop


def test_simulation_on_timer(simulation_with_mocks):
    """
    Tests whether the timer event updates particle positions and GUI properly.
    Simulates an elapsed time to trigger FPS calculation.
    """
    simulation_with_mocks.last_time = time.perf_counter() - 1.1  # Simulate time passing
    simulation_with_mocks.on_timer(MagicMock())

    # Verify that particle positions were updated and GUI was redrawn
    simulation_with_mocks.particle_creator.update_positions.assert_called_once()
    simulation_with_mocks.gui.draw_particles.assert_called_once()
    simulation_with_mocks.gui.update_fps.assert_called_once()


def test_simulation_benchmark_mode(simulation_with_mocks):
    """Tests whether the benchmark mode stops after 60 seconds and collects FPS data."""
    simulation_with_mocks.benchmark_mode = True
    simulation_with_mocks.start_time = time.perf_counter() - 61  # Simulate 61 seconds elapsed
    simulation_with_mocks.last_time = time.perf_counter() - 2  # Simulate elapsed time

    simulation_with_mocks.on_timer(MagicMock())  # Trigger timer event

    # Ensure the simulation stops after benchmark duration and FPS data is collected
    assert len(simulation_with_mocks.fps_list) > 0  # FPS data should be stored
    assert not simulation_with_mocks.running  # Simulation should stop after 60 seconds


def test_simulation_benchmark_mode_avg_fps(simulation_with_mocks, capsys):
    """Tests if the benchmark mode correctly calculates the average FPS."""
    simulation_with_mocks.benchmark_mode = True
    simulation_with_mocks.fps_list = [30, 40, 50]  # Sample FPS values

    avg_fps = sum(simulation_with_mocks.fps_list) / len(simulation_with_mocks.fps_list)

    simulation_with_mocks.stop()  # Ensure the benchmark mode stops

    assert avg_fps == pytest.approx(40.00, rel=0.01)  # Verify the average FPS calculation
