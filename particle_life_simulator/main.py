from Class_GUI import GUI
from Class_Particle import CreateParticle
from Class_simulation import Simulation
from vispy import app
def main():
    gui = GUI(window_width=1920, window_height=1080, particle_size=5)

    particle_creator = CreateParticle(
    num_particles=500,
    x_max=1920,
    y_max=1080,
    speed_range=(-1, 1),
    max_speed=1.0,
    radius=5,
    num_colors=5,
    interaction_strength=0.2,
)

    particle_creator.generate_particles()

    interaction_matrix = [
        [0, -1, 1, 0, 0],
        [-1, 0, -1, 1, 0],
        [1, -1, 0, 0, 1],
        [0, 1, 0, 0, -1],
        [0, 0, 1, -1, 0],
    ]
    particle_creator.set_interaction_matrix(interaction_matrix)

    simulation = Simulation(particle_creator=particle_creator, gui=gui)
    simulation.start()

    app.run()

if __name__ == "__main__":
    main()
    
