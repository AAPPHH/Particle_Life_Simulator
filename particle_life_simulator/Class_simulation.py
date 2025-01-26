import time
from vispy import app

class Simulation(app.Timer):
    def __init__(self, particle_creator, gui):
        super().__init__(interval=1 / 60, start=False)
        self.particle_creator = particle_creator
        self.gui = gui
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.connect(self.on_timer)

    def start(self):
        print("Simulation started!")
        super().start()

    def on_timer(self, event):
        """
        Updates the particle positions, redraws them on the GUI, and updates the FPS counter.
        """
        # Update particle positions
        self.particle_creator.update_positions()
        particles = self.particle_creator.get_positions_and_colors()
        self.gui.draw_particles(particles)

        # Calculate and update FPS
        self.frame_count += 1
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_time
        if elapsed_time > 1.0:  # Update FPS once per second
            fps = self.frame_count / elapsed_time
            self.gui.update_fps(fps)
            self.frame_count = 0
            self.last_time = current_time
