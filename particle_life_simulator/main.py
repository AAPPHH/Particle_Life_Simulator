from Class_simulation import Simulation
from Class_GUI import GUI
from Class_Particle import CreateParticle

def setup_simulation():
    """
    Setzt die Simulation mit Partikelerzeugung und GUI auf.
    """
    particle_creator = CreateParticle(
        num_particles=4000,
        x_max=1920,
        y_max=1080,
        speed_range=(-1, 1),
        radius=8
    )
    particle_creator.generate_particles()

    interaction_matrix = [
        [0, +5, -5,  0, +5],
        [+5, 0,  0, -5, -5],
        [-5, 0,  0, +5,  0],
        [0, -5, +5,  0, +5],
        [+5, -5, 0, +5,  0]
    ]
    particle_creator.set_interaction_matrix(interaction_matrix)

    gui = GUI(window_width=1920, window_height=1080, particle_size=4)

    return Simulation(particle_creator, gui)

def main():
    """
    Hauptlogik der Simulation.
    """
    simulation = setup_simulation()
    simulation.start()

if __name__ == "__main__":
    main()
