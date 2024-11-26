from simulation import Simulation
from gui import Gui_Manager
from Class_Particle import Particle


if __name__ == "__main__":

    width, height, scale = 800, 600, 8

    simulation = Simulation(width=width, height=height, scale=scale, time_step=1)

    dot_x, dot_y = 10, 10
    velocity_x, velocity_y = 1, 1
    particle = Particle(
        position=[dot_x, dot_y],
        velocity=[velocity_x, velocity_y],
        particle_type='dot',
        color=(255, 255, 255),
        interaction_strength=0,
        influence_radius=0,
        friction=0.0,
        random_motion=0.0
    )

    simulation.add_particle(particle)

    gui_manager = Gui_Manager(simulation, width=width, height=height, scale=scale)

    gui_manager.run()
