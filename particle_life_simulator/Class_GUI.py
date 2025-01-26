from vispy import app, scene
from vispy.visuals import transforms
import numpy as np


class GUI:
    def __init__(self, window_width: int = 1920, window_height: int = 1080, particle_size: int = 10, color_lookup: dict = None):
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
        self.color_lookup = (
            color_lookup
            if color_lookup
            else {
                0: (1.0, 0.0, 0.0),  # Red
                1: (0.0, 0.0, 1.0),  # Blue
                2: (0.0, 1.0, 0.0),  # Green
                3: (1.0, 1.0, 0.0),  # Yellow
                4: (1.0, 0.0, 1.0),  # Magenta
            }
        )

        # VisPy canvas and visuals
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, size=(window_width, window_height))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(0, self.window_width), y=(0, self.window_height))
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.fps_label = scene.Label("FPS: 0", color='white', font_size=14, anchor_x='right', anchor_y='top')
        self.fps_label.transform = transforms.STTransform(translate=(self.window_width - 10, 10))
        self.view.add(self.fps_label)

    def update_fps(self, fps: float) -> None:
        """
        Updates the FPS label in the GUI.

        Args:
            fps (float): The current frames per second to display.
        """
        self.fps_label.text = f"FPS: {fps:.2f}"

    def draw_particles(self, particles: list) -> None:
        """
        Draws the particles in the drawing area.

        Args:
            particles (list): A list of tuples containing the x and y positions of each particle and their color index.
        """
        if not particles:
            print("No particles to draw.")
            return

        positions = np.array([[x, y] for x, y, _ in particles], dtype=np.float32)
        colors = np.array(
            [self.color_lookup.get(color_index, (1.0, 1.0, 1.0)) for _, _, color_index in particles], dtype=np.float32
        )

        self.scatter.set_data(positions, face_color=colors, size=self.particle_size)