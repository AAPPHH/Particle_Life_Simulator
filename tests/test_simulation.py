import os
os.environ["VISPY_USE_APP"] = "pyqt5"

import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
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
    assert len(simulation_with_mocks.fps_list) > 0  
    assert not simulation_with_mocks.running  


def test_simulation_benchmark_mode_avg_fps(simulation_with_mocks, capsys):
    """Tests if the benchmark mode correctly calculates the average FPS."""
    simulation_with_mocks.benchmark_mode = True
    simulation_with_mocks.fps_list = [30, 40, 50]

    avg_fps = sum(simulation_with_mocks.fps_list) / len(simulation_with_mocks.fps_list)

    simulation_with_mocks.stop()  

    assert avg_fps == pytest.approx(40.00, rel=0.01)  


def test_simulation_stop_prevents_timer(simulation_with_mocks):
    """Ensures on_timer is not triggered after stop() is called."""
    simulation_with_mocks.stop()
    event = MagicMock()
    simulation_with_mocks.on_timer(event)
    simulation_with_mocks.particle_creator.update_positions.assert_not_called()


def test_simulation_fps_calculation(simulation_with_mocks):
    """Tests if FPS calculation matches expected values."""
    simulation_with_mocks.frame_count = 120
    simulation_with_mocks.last_time = time.perf_counter() - 2 
    
    simulation_with_mocks.on_timer(MagicMock())

    expected_fps = 120 / 2  
    simulation_with_mocks.gui.update_fps.assert_called_with(expected_fps)


def test_simulation_low_fps(simulation_with_mocks, capsys):
    """Tests if low FPS values are correctly handled."""
    simulation_with_mocks.benchmark_mode = True
    simulation_with_mocks.start_time = time.perf_counter() - 61  
    simulation_with_mocks.fps_list = [5, 5, 5]

    simulation_with_mocks.on_timer(MagicMock())

    captured = capsys.readouterr()
    assert "5.00" in captured.out  # Expected average FPS output
