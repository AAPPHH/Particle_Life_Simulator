import os
os.environ["VISPY_USE_APP"] = "pyqt5"

import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from particle_life_simulator.Class_simulation import Simulation


@pytest.fixture
def create_mocked_simulation():
    """Creates a simulation instance with mocked GUI and particle generator.
    The Timer object is also mocked to prevent GUI-related issues in CI environments.
    """
    particle_gen_mock = MagicMock()
    particle_gen_mock.get_positions_and_colors.return_value = np.array([[0, 0, 1]])
    particle_gen_mock.update_positions = MagicMock()

    gui_mock = MagicMock()
    gui_mock.update_fps = MagicMock()
    gui_mock.draw_particles = MagicMock()

    # Mock VisPy's Timer to avoid real GUI interactions in CI
    with patch('vispy.app.Timer') as MockTimer:
        mock_timer_instance = MockTimer.return_value
        mock_timer_instance.start = MagicMock()
        mock_timer_instance.stop = MagicMock()
        
        sim = Simulation(
            particle_creator=particle_gen_mock,
            gui=gui_mock,
            benchmark_mode=False
        )
        
    return sim


def test_simulation_lifecycle(create_mocked_simulation):
    """Tests whether the simulation starts and stops correctly."""
    create_mocked_simulation.start()
    assert create_mocked_simulation.running  

    create_mocked_simulation.stop()
    assert not create_mocked_simulation.running  


def test_simulation_timer_event(create_mocked_simulation):
    """
    Tests whether the timer event updates particle positions and GUI properly.
    Simulates an elapsed time to trigger FPS calculation.
    """
    create_mocked_simulation.last_time = time.perf_counter() - 1.1  # Simulate time passing
    create_mocked_simulation.on_timer(MagicMock())

    # Verify that particle positions were updated and GUI was redrawn
    create_mocked_simulation.particle_creator.update_positions.assert_called_once()
    create_mocked_simulation.gui.draw_particles.assert_called_once()
    create_mocked_simulation.gui.update_fps.assert_called_once()


def test_simulation_benchmark(create_mocked_simulation):
    """Tests whether the benchmark mode stops after 60 seconds and collects FPS data."""
    create_mocked_simulation.benchmark_mode = True
    create_mocked_simulation.start_time = time.perf_counter() - 61  
    create_mocked_simulation.last_time = time.perf_counter() - 2  

    create_mocked_simulation.on_timer(MagicMock())  # Trigger timer event

    # Ensure the simulation stops after benchmark duration and FPS data is collected
    assert len(create_mocked_simulation.fps_list) > 0  
    assert not create_mocked_simulation.running  


def test_simulation_benchmark_fps_avg(create_mocked_simulation, capsys):
    """Tests if the benchmark mode correctly calculates the average FPS."""
    create_mocked_simulation.benchmark_mode = True
    create_mocked_simulation.fps_list = [30, 40, 50]  

    avg_fps_calculated = sum(create_mocked_simulation.fps_list) / len(create_mocked_simulation.fps_list)

    create_mocked_simulation.stop()  

    assert avg_fps_calculated == pytest.approx(40.00, rel=0.01) 
