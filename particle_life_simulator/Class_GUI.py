from vispy import app, scene
from vispy.visuals import transforms
import numpy as np
from numba import prange, njit, typed, types
import tkinter as tk


class GUI:

    def __init__(
        self,
        window_width: int = None,
        window_height: int = None,
        particle_size: int = 10,
        color_lookup: dict = None,
    ):
        
        # tk to get screen size
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

        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, fullscreen=True, size=(self.window_width, self.window_height)
        )

        self.view = self.canvas.central_widget.add_view()
        # coordinates according to display size
        self.view.camera = scene.cameras.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(0, self.window_width), y=(0, self.window_height))
        self.scatter = scene.visuals.Markers(parent=self.view.scene)

        # FPS Label
        self.fps_label = scene.Label("FPS: 0", color="white", font_size=14, anchor_x="right", anchor_y="top")
        self.fps_label.transform = transforms.STTransform(translate=(self.window_width * 0.068, self.window_height * 0.815))
        self.view.add(self.fps_label)

        self.add_buttons()
        self.add_interaction_box()
        self.add_sliders()

    def add_buttons(self):

        # Stop Button
        self.button_width = self.window_width * 0.078
        self.button_height = self.window_height * 0.046
        self.button_x = self.window_width * 0.104
        self.button_y = self.window_height * 0.93

        self.stop_button = scene.visuals.Rectangle(
            center=(self.button_x, self.button_y),
            width=self.button_width,
            height=self.button_height,
            color=(0.4, 0.0, 0.0, 1),
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

    
    def add_interaction_box(self):

        # Box for Slider and FPS Label
        self.interaction_box = scene.visuals.Rectangle(
            center=(self.window_width * 0.104, self.window_height * 0.58),
            width=self.window_width * 0.182,
            height=self.window_height * 0.556,
            color=(0.07, 0.07, 0.07, 0.4),
            parent=self.view.scene,
        )

    
    def add_sliders(self):

        self.sliders = {}

        # Same Box as interaciton Box
        box_center = (self.window_width * 0.104, self.window_height * 0.58)
        box_width = self.window_width * 0.182
        box_height = self.window_height * 0.556
        box_left = box_center[0] - box_width / 2
        box_top = box_center[1] + box_height / 2

        # Slider position
        margin_x = 10
        slider_width = box_width - 2 * margin_x
        slider_height = 10
        slider_y_start = self.window_height * 0.75
        slider_spacing = 60


        def create_slider(name, min_val, max_val, default_val, pos_y):

            # Slider Track
            track_x = box_left + margin_x
            track_center = (track_x + slider_width / 2, pos_y)
            track = scene.visuals.Rectangle(
                center=track_center,
                width=slider_width,
                height=slider_height,
                color=(0.8, 0.8, 0.8, 1.0),
                parent=self.view.scene,
            )

            # Calculate handle position
            fraction = (default_val - min_val) / (max_val - min_val)
            handle_x = track_x + fraction * slider_width
            handle = scene.visuals.Rectangle(
                center=(handle_x, pos_y),
                width=10,
                height=slider_height + 4,
                color=(1, 0, 0, 1),
                parent=self.view.scene,
            )

            # Slider Label with value
            label = scene.Text(
                f"{name}: {default_val:.2f}",
                color="white",
                font_size=12,
                parent=self.view.scene,
                pos=(track_x, pos_y + slider_height + 20),
                anchor_x="left",
                anchor_y="bottom",
            )

            # Save track bounds for click detection
            track_bounds = (track_x, track_x + slider_width, pos_y - slider_height / 2, pos_y + slider_height / 2)
            print(f"{name} Slider bounds to {track_bounds}")
            return {
                "min": min_val,
                "max": max_val,
                "value": default_val,
                "track": track,
                "handle": handle,
                "label": label,
                "bounds": track_bounds,
            }


        # Create sliders with updated positions and spacing
        self.sliders["interaction_strength"] = create_slider(
            name="Interaction Strength", min_val=0.0, max_val=1.0, default_val=0.1, pos_y=slider_y_start
        )
        self.sliders["max_speed"] = create_slider(
            name="Max Speed", min_val=0.5, max_val=5.0, default_val=2.0, pos_y=slider_y_start - slider_spacing
        )

    
    def on_mouse_release(self, event):

        if event.pos is not None:
            x, y = event.pos
            # Coordinates
            mouse_x, mouse_y = self.view.camera.transform.imap((x, y))[:2]

            # Stop Button Check
            button_x_min = self.button_x - (self.button_width / 2)
            button_x_max = self.button_x + (self.button_width / 2)
            button_y_min = self.button_y - (self.button_height / 2)
            button_y_max = self.button_y + (self.button_height / 2)
            if button_x_min <= mouse_x <= button_x_max and button_y_min <= mouse_y <= button_y_max:
                self.stop_simulation()
                return

            # Sliders Check
            self.check_slider_click(mouse_x, mouse_y)


    def check_slider_click(self, x, y) -> bool:

        # Checking click
        for key, slider in self.sliders.items():
            x_min, x_max, y_min, y_max = slider["bounds"]

            if x_min <= x <= x_max and y_min <= y <= y_max:
                fraction = (x - x_min) / (x_max - x_min)
                new_value = slider["min"] + fraction * (slider["max"] - slider["min"])
                slider["value"] = new_value
                # update handle position and label
                new_handle_x = x_min + fraction * (x_max - x_min)
                slider["handle"].center = (new_handle_x, slider["handle"].center[1])
                slider["label"].text = f"{key.replace('_', ' ').title()}: {new_value:.2f}"
                print(f"{key}-Slider set to {new_value:.2f}")
                return True
        return False

    
    def stop_simulation(self):
        print("Simulation stopped.")
        self.canvas.close()
        app.quit()


    def update_fps(self, fps: float) -> None:
        self.fps_label.text = f"FPS: {fps:.2f}"
        self.fps_label.transform = transforms.STTransform(
            translate=(self.window_width * 0.068, self.window_height * 0.815)
        )


    def draw_particles(self, particles: np.ndarray) -> None:
        if particles is None or len(particles) == 0:
            print("No particles to draw.")
            return
        positions, colors = process_positions_and_colors(particles, self.numba_color_lookup)
        self.scatter.set_data(positions, face_color=colors, size=self.particle_size)


def create_numba_dict(color_lookup):
    color_dict = typed.Dict.empty(key_type=types.int32, value_type=types.float32[:])
    for key, value in color_lookup.items():
        color_dict[key] = np.array(value, dtype=np.float32)
    return color_dict


@njit(parallel=True)

def process_positions_and_colors(particles, color_lookup_dict):
    num_particles = len(particles)
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
