import pytest
import numpy as np
from particle_life_simulator.Class_Particle import Particle

def test_particle_initialization():
    """Test if a Particle object initializes without errors."""
    position = [0, 0]
    velocity = [1, 1]
    particle = Particle(
        position=position,
        velocity=velocity,
        particle_type="type1",
        color="red",
        interaction_strength=1.0,
        influence_radius=5.0,
        friction=0.1,
        random_motion=0.05,
    )

    # Verify attributes
    assert np.all(particle.position == np.array(position, dtype=float))
    assert np.all(particle.velocity == np.array(velocity, dtype=float))
    assert particle.type == "type1"
    assert particle.color == "red"
    assert particle.interaction_strength == 1.0
    assert particle.influence_radius == 5.0
    assert particle.friction == 0.1
    assert particle.random_motion == 0.05

def test_update_position():
    """Test the update_position method."""
    particle = Particle([0, 0], [2, 3], "type1", "blue", 1.0, 5.0, 0.1, 0.05)
    particle.update_position(1.0)
    assert np.all(particle.position == np.array([2, 3], dtype=float))

def test_apply_friction():
    """Test the apply_friction method."""
    particle = Particle([0, 0], [10, 10], "type1", "green", 1.0, 5.0, 0.1, 0.05)
    particle.apply_friction()
    assert np.allclose(particle.velocity, [9, 9])  # 10 * (1 - 0.1)

def test_randomize_movement():
    """Test the randomize_movement method."""
    particle = Particle([0, 0], [0, 0], "type1", "yellow", 1.0, 5.0, 0.1, 0.05)
    particle.randomize_movement()
    # The velocity should change, but the exact value is random
    assert not np.all(particle.velocity == np.array([0, 0]))
