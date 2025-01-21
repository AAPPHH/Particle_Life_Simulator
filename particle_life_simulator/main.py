from Class_GUI import GUI
from Class_Particle import CreateParticle
from Class_simulation import Simulation



def setup_simulation():
    """
    Sets up the simulation with particle generation and GUI.
    """
    particle_creator = CreateParticle(num_particles=1000, x_max=1920, y_max=1080, speed_range=(-1, 1), radius=8)
    particle_creator.generate_particles()

    interaction_matrix = [
        [-25, 0],
        [-25, 0],
    ]
    particle_creator.set_interaction_matrix(interaction_matrix)

    gui = GUI(window_width=1920, window_height=1080, particle_size=4)

    return Simulation(particle_creator, gui)


def main():
    """
    Main logic of the simulation.
    """
    simulation = setup_simulation()
    simulation.start()


if __name__ == "__main__":
    main()
