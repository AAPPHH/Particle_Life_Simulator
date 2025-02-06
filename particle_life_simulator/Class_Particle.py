import random
import math
import numpy as np
from numba import njit, prange
from numba.experimental import jitclass
from numba import int32, float32

spec = [
    ("num_particles", int32),
    ("x_max", int32),
    ("y_max", int32),
    ("speed_range", float32[:]),
    ("max_speed", float32),
    ("radius", int32),
    ("num_colors", int32),
    ("interaction_strength", float32),
    ("color_interaction", float32[:, :]),
    ("particles", float32[:, :]),
]

@jitclass(spec)
class CreateParticle:
    def __init__(
        self,
        num_particles: int = 1000,
        x_max: int = 1920,
        y_max: int = 1080,
        speed_range: tuple = (-2.0, 2.0),
        max_speed: float = 2.0,
        radius: int = 5,
        num_colors: int = 2,
        interaction_strength: float = 0.1,
    ):
        self.num_particles = num_particles
        self.x_max = x_max
        self.y_max = y_max
        self.speed_range = np.array(speed_range, dtype=np.float32)
        self.max_speed = max_speed
        self.radius = radius
        self.num_colors = num_colors
        self.interaction_strength = interaction_strength

        self.color_interaction = np.zeros((num_colors, num_colors), dtype=np.float32)

        self.particles = np.zeros((self.num_particles, 5), dtype=np.float32)

    def set_interaction_matrix(self, matrix: np.ndarray):
        """Sets a custom interaction matrix."""
        self.color_interaction[:, :] = matrix

    def generate_particles(self) -> None:
        self.particles[:, 0] = np.random.randint(self.radius, self.x_max - self.radius, self.num_particles).astype(np.float32)
        self.particles[:, 1] = np.random.randint(self.radius, self.y_max - self.radius, self.num_particles).astype(np.float32)
        self.particles[:, 2] = np.random.uniform(self.speed_range[0], self.speed_range[1], self.num_particles).astype(np.float32)
        self.particles[:, 3] = np.random.uniform(self.speed_range[0], self.speed_range[1], self.num_particles).astype(np.float32)
        self.particles[:, 4] = np.random.randint(0, self.num_colors, self.num_particles).astype(np.float32)


    def update_positions(self):
        old_particles = self.particles.copy()

        neighbor_lists = compute_neighbors_quadtree(self.particles, self.x_max, self.y_max, self.radius)

        self.particles = update_positions_numba(
            old_particles,
            self.particles,
            self.x_max,
            self.y_max,
            self.radius,
            self.color_interaction,
            self.interaction_strength,
            self.max_speed,
            neighbor_lists
        )

    def get_positions_and_colors(self) -> np.ndarray:
        """
        Returns the positions (x, y) and color index of particles as a NumPy array.
        Shape: (num_particles, 3) -> [[x1, y1, color1], [x2, y2, color2], ...]
        """
        return np.column_stack((self.particles[:, 0], self.particles[:, 1], self.particles[:, 4]))

@njit(parallel=True, fastmath=True)
def update_positions_numba(
    old_particles,
    new_particles,
    x_max,
    y_max,
    radius,
    interaction_matrix,
    interaction_strength,
    max_speed,
    neighbor_lists
):
    num_particles = len(old_particles)
    radius_sq = (2 * radius) * (2 * radius)

    for i in prange(num_particles):
        x, y, vx, vy, color = old_particles[i]

        x_new = (x + vx) % x_max
        y_new = (y + vy) % y_max

        speed_sq = vx * vx + vy * vy
        if speed_sq > max_speed * max_speed:
            scale = max_speed * fast_inv_sqrt(speed_sq)
            vx *= scale
            vy *= scale

        for j in neighbor_lists[i]:
            if j == -1:
                break

            dx = new_particles[j, 0] - x_new
            dy = new_particles[j, 1] - y_new
            dist_sq = dx * dx + dy * dy

            if dist_sq < radius_sq:
                dist = max(math.sqrt(dist_sq), 1e-8)
                overlap = 2 * radius - dist
                nx = dx / dist
                ny = dy / dist

                x_new -= 0.5 * overlap * nx
                y_new -= 0.5 * overlap * ny

        new_particles[i, 0] = x_new
        new_particles[i, 1] = y_new
        new_particles[i, 2] = vx
        new_particles[i, 3] = vy
        new_particles[i, 4] = color

    return new_particles


@njit(fastmath=True)
def fast_inv_sqrt(x):
    return 1.0 / math.sqrt(x)

@njit(parallel=True)
def compute_neighbors_quadtree(particles, x_max, y_max, radius):
    num_particles = len(particles)
    MAX_NEIGHBORS = 25
    neighbor_lists = np.full((num_particles, MAX_NEIGHBORS), -1, dtype=np.int32)

    quadtree = np.zeros((num_particles, 5), dtype=np.float32)

    def insert_into_quadtree(i, x, y, depth=0):
        if depth > 20:
            return
        quad_id = int((x // (x_max / 2)) + 2 * (y // (y_max / 2)))
        quadtree[i] = [x, y, quad_id, depth, 1]

    for i in prange(num_particles):
        insert_into_quadtree(i, particles[i, 0], particles[i, 1])

    for i in prange(num_particles):
        count = 0
        for j in range(num_particles):
            if i != j:
                dx = particles[i, 0] - particles[j, 0]
                dy = particles[i, 1] - particles[j, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (2 * radius) ** 2:
                    if count < MAX_NEIGHBORS:
                        neighbor_lists[i, count] = j
                        count += 1

    return neighbor_lists