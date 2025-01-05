import cProfile
from particle_life_simulator.Class_Particle import CreateParticle

def run_simulation():
    """Führt die Simulation aus und misst die Performance."""
    simulator = CreateParticle(num_particles=1000, x_max=500, y_max=500, radius=5)
    simulator.generate_particles()
    simulator.update_positions()

if __name__ == "__main__":
    cProfile.run('run_simulation()')