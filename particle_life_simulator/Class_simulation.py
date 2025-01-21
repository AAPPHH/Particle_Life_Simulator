import time

import dearpygui.dearpygui as dpg

from Class_GUI import GUI
from Class_Particle import CreateParticle



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

    def start(self) -> None:
        """
        Starts the simulation, handling the main loop and rendering.
        """
        self.gui.setup_window()

        last_time = time.time()  # Start time for FPS calculation
        frame_count = 0  # Counter for rendered frames

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

        self.gui.cleanup()

    def update(self) -> None:
        """
        Updates the particle positions and redraws them.
        """
        self.particle_creator.update_positions()
        positions_and_colors = self.particle_creator.get_positions_and_colors()
        self.gui.clear_drawlist()
        self.gui.draw_particles(positions_and_colors)
