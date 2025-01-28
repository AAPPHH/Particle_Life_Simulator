from vispy import app, scene
from vispy.visuals import transforms
import numpy as np
from numba import njit
from tkinter import *

win= Tk()

win.geometry("650x250")

screen_width = win.winfo_screenwidth()
screen_height = win.winfo_screenheight()

class GUI:
    def __init__(self, window_width: screen_width, window_height: screen_height, particle_size: int = 10, color_lookup: dict = None):
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
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, fullscreen=True, size=(window_width, window_height))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(0, self.window_width), y=(0, self.window_height))
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.fps_label = scene.Label("FPS: 0", color='white', font_size=14, anchor_x='right', anchor_y='top')
        self.fps_label.transform = transforms.STTransform(translate=(self.window_width - 10, 10))
        self.view.add(self.fps_label)
        self.add_buttons()

    def add_buttons(self):

        self.stop_button = scene.visuals.Rectangle(
            center=(200, self.window_height - 75),
            width=150,
            height=50,
            color=(0.4, 0.0, 0.0, 0.4),
            parent=self.view.scene,
        )

        self.interaction_box = scene.visuals.Rectangle(
            center=(200, self.window_height - 450),
            width=350,
            height=600,
            color=(0.07, 0.07, 0.07, 0.4),
            parent=self.view.scene,
        )

       
        #Button Text
        self.stop_label = scene.Text(
            "STOP",
            color="white",
            font_size=15,
            bold=True,
            parent=self.view.scene,
            pos=(200, self.window_height - 75),
            anchor_x="center",
            anchor_y="center",
        )


        self.canvas.events.mouse_release.connect(self.on_mouse_release)

    def on_mouse_release(self, event):
        if event.pos is not None:
            x, y = event.pos
            #Map mouse click coordinates
            canvas_x, canvas_y = self.canvas.size[0], self.canvas.size[1]
            button_x_min = 200 - 75
            button_x_max = 200 + 125
            button_y_min = canvas_y - (self.window_height - 75 + 25)
            button_y_max = canvas_y - (self.window_height - 75 - 75)

            # Check if button clicked
            if button_x_min <= x <= button_x_max and button_y_min <= y <= button_y_max:
                self.stop_simulation()

    def stop_simulation(self):
        print("Simulation stopped.")
        self.canvas.close()
        app.quit()
     


    def update_fps(self, fps: float) -> None:

        self.fps_label.text = f"FPS: {fps:.2f}"
        #Position
        self.fps_label.transform = transforms.STTransform(translate=(130, self.window_height - 200))

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

