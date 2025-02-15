import os
import sys
import pytest
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../particle_life_simulator")))


from particle_life_simulator.main import main


@patch("particle_life_simulator.main.GUI")
@patch("particle_life_simulator.main.CreateParticle")
@patch("particle_life_simulator.main.Simulation")
@patch("vispy.app.run")
def test_main_execution(mock_run, mock_sim, mock_particles, mock_gui):
    """Checks if main() correctly initializes key components without enforcing specific parameters."""

    main()

    # Verify if the expected instances were created
    mock_gui.assert_called_once()  
    mock_particles.assert_called_once()
    mock_sim.assert_called_once()
    mock_run.assert_called_once()


@pytest.mark.benchmark
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Benchmark tests are only for local execution")
def test_simulation_speed(benchmark):
    """Measures the simulation's performance under high particle load."""

    from particle_life_simulator.Class_Particle import CreateParticle
    from particle_life_simulator.Class_simulation import Simulation

    p_set = CreateParticle(
        num_particles=100000,
        x_max=1920,
        y_max=1080,
        speed_range=(-1, 1),
        max_speed=1.0,
        radius=3,
        num_colors=5,
        interaction_strength=0.5,
    )
    p_set.generate_particles()

    # GUI is required but mocked
    gui_mock = MagicMock()
    sim_test = Simulation(p_set, gui=gui_mock)

    def run_sim():
        sim_test.on_timer(None)  

    benchmark(run_sim)
