import dearpygui.dearpygui as dpg

class GUI:
    def __init__(self, window_width: int = 1920, window_height: int = 1080, particle_size: int = 10):
        """
        Initializes the GUI with specified window dimensions and particle size.

        Args:
            window_width (int): Width of the window.
            window_height (int): Height of the window.
            particle_size (int): Visual size of the particles.
        """
        self.window_width = window_width
        self.window_height = window_height
        self.particle_size = particle_size

    def setup_window(self) -> None:
        """
        Sets up the Dear PyGui window and viewport for the simulation.
        """
        dpg.create_context()

        with dpg.window(label="Particle Simulator", width=self.window_width, height=self.window_height, tag="main_window"):
            with dpg.drawlist(width=self.window_width, height=self.window_height, tag="drawlist"):
                pass

            # Add FPS display
            dpg.add_text("FPS: 0", tag="fps_label", pos=(self.window_width - 100, 30))

        dpg.create_viewport(title="Particle Simulator", width=self.window_width, height=self.window_height)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def update_fps(self, fps: float) -> None:
        """
        Updates the FPS label in the GUI.

        Args:
            fps (float): The current frames per second to display.
        """
        dpg.set_value("fps_label", f"FPS: {fps:.2f}")

    def clear_drawlist(self) -> None:
        """
        Clears all particles from the drawing area.
        """
        dpg.delete_item("drawlist", children_only=True)

    def draw_particles(self, particles: list) -> None:
        """
        Draws the particles in the drawing area.

        Args:
            particles (list): A list of dictionaries containing the x and y positions of each particle.
        """
        for particle in particles:
            x = particle['x']
            y = particle['y']
            adjusted_y = self.window_height - y
            dpg.draw_circle((x, adjusted_y),
                            radius=self.particle_size / 2,
                            color=(255, 255, 255), fill=(255, 255, 255),
                            parent="drawlist")

    def cleanup(self) -> None:
        """
        Destroys the Dear PyGui context and cleans up resources.
        """
        dpg.destroy_context()