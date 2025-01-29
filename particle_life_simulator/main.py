from Class_GUI import GUI
from Class_Particle import CreateParticle
from Class_simulation import Simulation
from vispy import app
from tkinter import *

win= Tk()

win.geometry("650x250")

screen_width = win.winfo_screenwidth()
screen_height = win.winfo_screenheight()

def main():
    gui = GUI(window_width=screen_width, window_height=screen_height, particle_size=3)

    particle_creator = CreateParticle(
    num_particles=100,
    x_max=1920,
    y_max=1080,
    speed_range=(-1, 1),
    max_speed=1.0,
    radius=3,
    num_colors=5,
    interaction_strength=0.2,
)

    particle_creator.generate_particles()

    interaction_matrix = [
        [0, 0, 0, 0, 0], # Red interactions
        [0, 0, 0, 0, 0], # Blue interactions
        [0, 0, 0, 0, 0], # Green interactions
        [0, 0, 0, 0, 0], # Yellow interactions
        [0, 0, 0, 0, 0], # Magenta interactions
    ]
    particle_creator.set_interaction_matrix(interaction_matrix)

    simulation = Simulation(particle_creator=particle_creator, gui=gui)
    simulation.start()

    app.run()

if __name__ == "__main__":
    main()
    
