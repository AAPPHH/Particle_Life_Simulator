import time
import dearpygui.dearpygui as dpg
from Class_Particle import CreateParticle
from Class_GUI import GUI
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class Simulation:
    def __init__(self, particle_creator: CreateParticle, gui: GUI):
        """
        Initializes the Simulation with a particle creator and a GUI.

        Args:
            particle_creator (CreateParticle): An instance to manage particles.
            gui (GUI): An instance to handle the graphical interface.
        """
        self.particle_creator = particle_creator
        self.gui = gui
        self.running = True
        self.num_processes = multiprocessing.cpu_count()  # Use all available cores

    def start(self) -> None:
        """
        Starts the simulation, handling the main loop and rendering.
        """
        self.gui.setup_window()

        last_time = time.time()  # Start time for FPS calculation
        frame_count = 0          # Counter for rendered frames

        while dpg.is_dearpygui_running() and self.running:
            current_time = time.time()
            frame_count += 1

            if current_time - last_time >= 1.0:
                elapsed_time = current_time - last_time
                fps = frame_count / elapsed_time
                self.gui.update_fps(fps)
                frame_count = 0
                last_time = current_time

            self.update()
            dpg.render_dearpygui_frame()
            time.sleep(0.016)

        self.gui.cleanup()

    def update(self) -> None:
        """
        Updates the particle positions and redraws them.
        """
        positions = self.particle_creator.get_positions()
        self.run_parallel_updates(positions)
        self.gui.draw_particles(self.particle_creator.get_positions())

    def run_parallel_updates(self, positions):
        """
        Parallelize the position updates and interactions.
        """
        chunk_size = len(positions) // self.num_processes
        position_chunks = [
            positions[i:i + chunk_size]
            for i in range(0, len(positions), chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            results = list(executor.map(self.update_position_chunk, position_chunks))

        # Flatten the results back into a single list
        updated_positions = [pos for chunk in results for pos in chunk]
        self.particle_creator.set_positions(updated_positions)

    def update_position_chunk(self, position_chunk):
        """
        Update a chunk of positions, handling boundary logic or interactions if necessary.
        """
        updated_positions = []
        for x, y in position_chunk:
            x, y = self.particle_creator._handle_boundaries(x, y)
            updated_positions.append((x, y))
        return updated_positions
