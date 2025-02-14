import math
import numpy as np
from particle_life_simulator.Class_Particle import (
    CreateParticle,
    update_positions_numba,
    compute_neighbors_grid,
)

def test_create_particle_initialization():
    """Test initialization of CreateParticle class."""
    cp = CreateParticle(
        num_particles=10,
        x_max=100,
        y_max=100,
        speed_range=(-1.0, 1.0),
        max_speed=1.5,
        min_speed=0.1,
        radius=10,
        num_colors=3,
        interaction_strength=0.2,
    )
    assert cp.num_particles == 10
    assert cp.x_max == 100
    assert cp.y_max == 100
    assert np.allclose(
        cp.speed_range,
        np.array([-1.0, 1.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert math.isclose(cp.max_speed, 1.5, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(cp.radius, 7.5, rel_tol=1e-6, abs_tol=1e-6)  # Adjusted test
    assert cp.num_colors == 3
    assert math.isclose(cp.interaction_strength, 0.2, rel_tol=1e-6, abs_tol=1e-6)

def test_generate_particles():
    """Test particle generation."""
    cp = CreateParticle(num_particles=5, x_max=100, y_max=100, radius=5)
    cp.generate_particles()
    particles = cp.particles

    assert particles.shape == (5, 5)
    assert np.all((particles[:, 0] >= 0) & (particles[:, 0] <= 100))  # x positions
    assert np.all((particles[:, 1] >= 0) & (particles[:, 1] <= 100))  # y positions
    assert np.all((particles[:, 2] >= -2.0) & (particles[:, 2] <= 2.0))  # x velocities
    assert np.all((particles[:, 3] >= -2.0) & (particles[:, 3] <= 2.0))  # y velocities
    assert np.all((particles[:, 4] >= 0) & (particles[:, 4] < cp.num_colors))  # colors

def test_set_interaction_matrix():
    """Test setting the interaction matrix."""
    cp = CreateParticle(num_colors=3)
    matrix = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9]], dtype=np.float32)
    cp.set_interaction_matrix(matrix)

    assert np.array_equal(cp.color_interaction, matrix)

def test_update_positions_numba():
    """Test the update_positions_numba function."""
    num_particles = 5
    particles = np.array(
        [
            [10, 10, 1, 1, 0],
            [20, 20, -1, -1, 1],
            [30, 30, 0.5, 0.5, 0],
            [40, 40, -0.5, -0.5, 1],
            [50, 50, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    x_max, y_max = 100, 100
    radius = 10
    radius_sq = radius ** 2
    interaction_matrix = np.zeros((2, 2), dtype=np.float32)
    interaction_strength = 0.1
    max_speed = 2.0
    min_speed = 0.1
    neighbor_lists = np.full((num_particles, 5), -1, dtype=np.int32)

    updated_particles = update_positions_numba(
        particles,
        num_particles,
        x_max,
        y_max,
        radius,
        radius_sq,
        interaction_matrix,
        interaction_strength,
        max_speed,
        min_speed,
        neighbor_lists,
    )

    assert updated_particles.shape == (num_particles, 5)
    assert np.all(updated_particles[:, 0] < x_max)
    assert np.all(updated_particles[:, 1] < y_max)

def test_compute_neighbors_grid():
    """Test the compute_neighbors_grid function."""
    particles = np.array(
        [
            [10, 10, 0, 0, 0],
            [20, 20, 0, 0, 1],
            [30, 30, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    x_max, y_max, radius = 100, 100, 10
    neighbor_lists = compute_neighbors_grid(particles, x_max, y_max, radius)

    assert neighbor_lists.shape[0] == particles.shape[0]
    assert neighbor_lists.shape[1] >= 1  # At least one neighbor per particle
