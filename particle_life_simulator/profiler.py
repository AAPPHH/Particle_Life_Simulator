import cProfile
import pstats

from Class_simulation import Simulation
from Class_GUI import GUI
from Class_Particle import CreateParticle

def main():
    particle_creator = CreateParticle(num_particles=1000, x_max=1920, y_max=1080, speed_range=(-2, 2), radius=5)
    particle_creator.generate_particles()

    gui = GUI(window_width=1920, window_height=1080, particle_size=10)

    simulation = Simulation(particle_creator, gui)
    simulation.start()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    with open("profiling_results.txt", "w") as text_file:
        stats = pstats.Stats(profiler, stream=text_file)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()

    profiler.dump_stats("profiling_results.prof")

    print("Profiling abgeschlossen. Ergebnisse:")
    print("- Lesbare Textdatei: profiling_results.txt")
    print("- Binäre Profildatei: profiling_results.prof (für SnakeViz oder andere Tools)")
