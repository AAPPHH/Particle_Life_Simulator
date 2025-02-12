import time
from vispy import app


class Simulation(app.Timer):
    def __init__(self, particle_creator, gui, benchmark_mode=True):
        super().__init__(interval=1 / 60, start=False)
        self.particle_creator = particle_creator
        self.gui = gui
        self.benchmark_mode = benchmark_mode
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.fps_list = []
        self.benchmark_duration = 60
        self.connect(self.on_timer)

    def start(self):
        print("Simulation started!")
        super().start()

    def on_timer(self, event):
        """
        Updates the particle positions, redraws them on the GUI, and updates the FPS counter.
        If benchmark mode is active, it calculates the average FPS over 60 seconds.
        """
        self.particle_creator.update_positions()
        self.gui.draw_particles(self.particle_creator.get_positions_and_colors())

        self.frame_count += 1
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_time

        if elapsed_time > 1.0:
            fps = self.frame_count / elapsed_time
            self.gui.update_fps(fps)
            self.fps_list.append(fps)
            self.frame_count = 0
            self.last_time = current_time

        if self.benchmark_mode:
            total_elapsed = current_time - self.start_time
            if total_elapsed >= self.benchmark_duration:
                avg_fps = sum(self.fps_list) / len(self.fps_list) if self.fps_list else 0
                print(f"Benchmark completed! Average FPS: {avg_fps:.2f}")
                self.stop()
