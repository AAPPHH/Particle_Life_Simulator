import time
from vispy import app
from Class_GUI import GUI
from Class_Particle import CreateParticle

class Simulation(app.Timer):
    def __init__(self, particle_creator, gui):
        super().__init__(interval=1 / 60, start=False)
        self.particle_creator = particle_creator
        self.gui = gui
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.connect(self.on_timer)

    def start(self):
        print("Simulation started!")
        super().start()

    def on_timer(self, event):
        """
        Updates the particle positions and redraws them on the GUI.
        """
        self.particle_creator.update_positions()
        particles = self.particle_creator.get_positions_and_colors()
        self.gui.draw_particles(particles)
