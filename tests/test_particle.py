import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from particle_life_simulator.Class_Particle import CreateParticle

def test_particle_initialization():
    """Test if a CreateParticle object initializes without errors."""
    num_particles = 100
    x_max, y_max = 1920, 1080
    speed_range = (-3, 3)
    radius = 10
    num_colors = 5
    particle_simulator = CreateParticle(
        num_particles=num_particles,
        x_max=x_max,
        y_max=y_max,
        speed_range=speed_range,
        radius=radius,
        num_colors=num_colors,
    )

    # Verify attributes
    assert particle_simulator.num_particles == num_particles
    assert particle_simulator.x_max == x_max
    assert particle_simulator.y_max == y_max
    assert particle_simulator.speed_range == speed_range
    assert particle_simulator.radius == radius
    assert particle_simulator.num_colors == num_colors
    assert len(particle_simulator.particles) == 0


def test_generate_particles():
    """Test if particles are generated correctly."""
    particle_simulator = CreateParticle(num_particles=10, x_max=100, y_max=100, radius=5, num_colors=3)
    particle_simulator.generate_particles()

    # Ensure the correct number of particles are generated
    assert len(particle_simulator.particles) == particle_simulator.num_particles

    # Ensure no particles overlap
    for i, (x1, y1, _, _, _) in enumerate(particle_simulator.particles):
        for j, (x2, y2, _, _, _) in enumerate(particle_simulator.particles):
            if i != j:
                distance = np.hypot(x2 - x1, y2 - y1)
                assert distance >= 2 * particle_simulator.radius

    # Ensure colors are within the expected range
    for _, _, _, _, color in particle_simulator.particles:
        assert 0 <= color < particle_simulator.num_colors  


def test_update_positions():
    """Test the update_positions method."""
    particle_simulator = CreateParticle(num_particles=1, x_max=100, y_max=100, radius=5, num_colors=5)
    particle_simulator.particles = [(10, 10, 2, 3, 1)]  # Initial position, velocity, and color
    particle_simulator.update_positions()

    x, y, vx, vy, color = particle_simulator.particles[0]
    assert (x, y) == (12, 13)
    assert color == 1  


def test_handle_boundaries():
    """Test the boundary handling logic."""
    particle_simulator = CreateParticle(x_max=100, y_max=100, radius=5, num_colors=5)

    # Check particles wrapping at boundaries
    x, y = particle_simulator._handle_boundaries(-1, 50)   # Left 
    assert (x, y) == (95, 50)

    x, y = particle_simulator._handle_boundaries(101, 50)  # Right 
    assert (x, y) == (5, 50)

    x, y = particle_simulator._handle_boundaries(50, -1)   # Top 
    assert (x, y) == (50, 95)

    x, y = particle_simulator._handle_boundaries(50, 101)  # Bottom 
    assert (x, y) == (50, 5)


def test_handle_collision():
    """Test the collision handling logic."""
    particle_simulator = CreateParticle()
    vx1, vy1, vx2, vy2 = particle_simulator._handle_collision(
        x1=0, y1=0, vx1=1, vy1=0, x2=2, y2=0, vx2=-1, vy2=0
    )
    # Verify velocities after an elastic collision
    assert (vx1, vy1, vx2, vy2) == (-1, 0, 1, 0)


def test_quadtree_insertion_and_query():
    """Verify that the Quadtree inserts and queries particles correctly."""
    simulator = CreateParticle(num_particles=3, x_max=100, y_max=100, radius=5, num_colors=3)
    simulator.particles = [(10, 10, 0, 0, 0), (50, 50, 0, 0, 1), (90, 90, 0, 0, 2)]
    simulator.update_quadtree()
    
    # Check the neighbors returned by the query
    queried_neighbors = simulator.quadtree.query(0, 0, 20, 20)
    assert len(queried_neighbors) == 1
    assert queried_neighbors[0][:2] == (10, 10)  # Nur Position prüfen


def test_large_particle_simulation():
    """Test simulation performance with many particles."""
    simulator = CreateParticle(num_particles=1000, x_max=500, y_max=500, radius=5, num_colors=5)
    simulator.generate_particles()
    
    # Confirm all particles are created
    assert len(simulator.particles) == 1000
    
    # Ensure positions are updated correctly
    simulator.update_positions()
