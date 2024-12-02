from Class_simulation import Simulation
from Class_GUI import GUI
from Class_Particle import CreateParticle

if __name__ == "__main__":
    particle_creator = CreateParticle(num_particles=100, x_max=1920, y_max=1080, speed_range=(-2, 2), radius=5)
    particle_creator.generate_particles()

    gui = GUI(window_width=1920, window_height=1080, particle_size=10)

    simulation = Simulation(particle_creator, gui)
    simulation.start()
