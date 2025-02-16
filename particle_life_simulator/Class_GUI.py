from vispy import app, scene
from vispy.visuals import transforms
import numpy as np
from numba import prange
from numba import njit, typed, types
import tkinter as tk


class GUI:
    
    def __init__(
        self,
        window_width: int = None,
        window_height: int = None,
        particle_size: int = 10,
        color_lookup: dict = None,
    ):
        self.win = tk.Tk()
        self.win.geometry("650x250")

        screen_width = self.win.winfo_screenwidth()
        screen_height = self.win.winfo_screenheight()

        self.window_width = window_width if window_width else screen_width
        self.window_height = window_height if window_height else screen_height
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

        self.numba_color_lookup = create_numba_dict(self.color_lookup)

        # VisPy Setup
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, fullscreen=True, size=(window_width, window_height)
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(0, self.window_width), y=(0, self.window_height))
        self.scatter = scene.visuals.Markers(parent=self.view.scene)

        # FPS Label
        self.fps_label = scene.Label("FPS: 0", color="white", font_size=14, anchor_x="right", anchor_y="top")
        self.fps_label.transform = transforms.STTransform(translate=(self.window_width * 0.124, self.window_height * 0.86))
        self.view.add(self.fps_label)

        self.add_buttons()


    def add_buttons(self):
        # Dynamische Positionen basierend auf Bildschirmgröße
        self.button_width = self.window_width * 0.078
        self.button_height = self.window_height * 0.046
        self.button_x = self.window_width * 0.104
        self.button_y = self.window_height * 0.93

        self.stop_button = scene.visuals.Rectangle(
            center=(self.button_x, self.button_y),
            width=self.button_width,
            height=self.button_height,
            color=(0.4, 0.0, 0.0, 0.4),
            parent=self.view.scene,
        )

        self.stop_label = scene.Text(
            "STOP",
            color="white",
            font_size=int(self.window_width * 0.008),
            bold=True,
            parent=self.view.scene,
            pos=(self.button_x, self.button_y),
            anchor_x="center",
            anchor_y="center",
        )

        self.canvas.events.mouse_release.connect(self.on_mouse_release)


    def on_mouse_release(self, event):
        if event.pos is not None:
            x, y = event.pos
            mouse_x, mouse_y = self.view.camera.transform.imap((x, y))[:2]

            button_x_min = self.button_x - (self.button_width / 2)
            button_x_max = self.button_x + (self.button_width / 2)
            button_y_min = self.button_y - (self.button_height / 2)
            button_y_max = self.button_y + (self.button_height / 2)

            if button_x_min <= mouse_x <= button_x_max and button_y_min <= mouse_y <= button_y_max:
                self.stop_simulation()


    def stop_simulation(self):
        print("Simulation stopped.")
        self.canvas.close()
        app.quit()


    def update_fps(self, fps: float) -> None:
        self.fps_label.text = f"FPS: {fps:.2f}"
        self.fps_label.transform = transforms.STTransform(
            translate=(self.window_width * 0.124, self.window_height * 0.86)
        )


    def draw_particles(self, particles: np.ndarray, num_particles: int) -> None:
        """
        Draws the particles in the drawing area.

        Args:
            particles (np.ndarray): Structured NumPy array with 'x', 'y', and 'color' fields.
        """
        positions, colors = process_positions_and_colors(particles, self.numba_color_lookup, num_particles)

        self.scatter.set_data(positions, face_color=colors, size=self.particle_size)



def create_numba_dict(color_lookup):
    """
    Converts a Python dictionary to a numba.typed.Dict for fast lookup.

    Args:
        color_lookup (dict): Dictionary mapping color indices to RGB colors.

    Returns:
        numba.typed.Dict: Numba-optimized dictionary.
    """
    color_dict = typed.Dict.empty(key_type=types.int32, value_type=types.float32[:])

    for key, value in color_lookup.items():
        color_dict[key] = np.array(value, dtype=np.float32)

    return color_dict


@njit(parallel=True)
def process_positions_and_colors(particles, color_lookup_dict, num_particles):
    positions = np.empty((num_particles, 2), dtype=np.float32)
    colors = np.empty((num_particles, 3), dtype=np.float32)

    for i in prange(num_particles):
        x = particles[i][0]
        y = particles[i][1]
        color_index = int(particles[i][2])

        positions[i, 0] = x
        positions[i, 1] = y

        colors[i] = color_lookup_dict.get(color_index, np.array([1.0, 1.0, 1.0], dtype=np.float32))

    return positions, colors
