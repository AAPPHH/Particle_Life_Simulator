from Class_simulation import Simulation
from Class_GUI import GUI
from Class_Particle import CreateParticle

if __name__ == "__main__":
    particle_creator = CreateParticle(num_particles=1000, x_max=1920, y_max=1080, speed_range=(-1, 1), radius=7)
    particle_creator.generate_particles()

    gui = GUI(window_width=1920, window_height=1080, particle_size=8)

    simulation = Simulation(particle_creator, gui)
    simulation.start()
