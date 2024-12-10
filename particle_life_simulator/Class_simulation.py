import time
import dearpygui.dearpygui as dpg
from Class_Particle import CreateParticle
from Class_GUI import GUI

class Simulation:
    def __init__(self, particle_creator: CreateParticle, gui: GUI, max_frames: int = 600):
        """
        Initializes the Simulation with a particle creator, GUI, and a maximum frame count.

        Args:
            particle_creator (CreateParticle): An instance to manage particles.
            gui (GUI): An instance to handle the graphical interface.
            max_frames (int): Maximum number of frames for the simulation.
        """
        self.particle_creator = particle_creator
        self.gui = gui
        self.running = True
        self.max_frames = max_frames  # Maximum frames to run

    def start(self) -> None:
        """
        Starts the simulation, handling the main loop and rendering.
        """
        self.gui.setup_window()

        frame_count = 0

        while dpg.is_dearpygui_running() and self.running:
            frame_count += 1

            # Stop simulation after max_frames
            if frame_count >= self.max_frames:
                print("Simulation beendet: Frame-Limit erreicht.")
                self.running = False
                break

            self.update()
            dpg.render_dearpygui_frame()
            time.sleep(0.016)

        self.gui.cleanup()

    def update(self) -> None:
        """
        Updates the particle positions and redraws them.
        """
        self.particle_creator.update_positions()
        positions = self.particle_creator.get_positions()
        self.gui.clear_drawlist()
        self.gui.draw_particles(positions)
