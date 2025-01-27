from vispy import app, scene
from vispy.visuals import transforms
import numpy as np
from numba import njit


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

        dtype = [('x', np.float32), ('y', np.float32), ('color_index', np.int32)]
        particles_array = np.array(particles, dtype=dtype)

        color_lookup_keys = np.array(list(self.color_lookup.keys()), dtype=np.int32)
        color_lookup_values = np.array(list(self.color_lookup.values()), dtype=np.float32)

        positions, colors = process_positions_and_colors(particles_array, color_lookup_keys, color_lookup_values)

        self.scatter.set_data(positions, face_color=colors, size=self.particle_size)

@njit
def process_positions_and_colors(particles, color_lookup_keys, color_lookup_values):
    """
    Processes particle positions and colors using Numba for faster performance.

    Args:
        particles (array): Structured NumPy array with fields 'x', 'y', and 'color_index'.
        color_lookup_keys (array): Array of keys representing the color indices.
        color_lookup_values (array): Array of corresponding RGB color values.

    Returns:
        tuple: Two numpy arrays - positions and colors.
    """
    positions = np.empty((len(particles), 2), dtype=np.float32)
    colors = np.empty((len(particles), 3), dtype=np.float32)

    for i in range(len(particles)):
        x = particles[i]['x']
        y = particles[i]['y']
        color_index = particles[i]['color_index']

        positions[i, 0] = x
        positions[i, 1] = y

        idx = np.where(color_lookup_keys == color_index)[0]
        if len(idx) > 0:
            colors[i, :] = color_lookup_values[idx[0]]
        else:
            colors[i, :] = (1.0, 1.0, 1.0)

    return positions, colors

