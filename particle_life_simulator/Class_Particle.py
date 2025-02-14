import math
import numpy as np
from numba.experimental import jitclass
from numba import int32, float32
from numba import njit, prange

spec = [
    ("num_particles", int32),
    ("x_max", int32),
    ("y_max", int32),
    ("speed_range", float32[:]),
    ("max_speed", float32),
    ("min_speed", float32),
    ("radius", float32),
    ("radius_sq", float32),
    ("num_colors", int32),
    ("interaction_strength", float32),
    ("color_interaction", float32[:, :]),
    ("particles", float32[:, :]),
]

@jitclass(spec)
class CreateParticle:
    """
    Particle system with float-based radius and radius_sq.
    """

    def __init__(
        self,
        num_particles: int = 1000,
        x_max: int = 1920,
        y_max: int = 1080,
        speed_range: tuple = (-2.0, 2.0),
        max_speed: float = 2.0,
        min_speed: float = 0.1,
        radius: float = 5.0,
        num_colors: int = 5,
        interaction_strength: float = 0.1,
        radius_factor: float = 0.75,
    ):
        self.num_particles = num_particles
        self.x_max = x_max
        self.y_max = y_max
        self.speed_range = np.array(speed_range, dtype=np.float32)
        self.max_speed = max_speed
        self.min_speed = min_speed

        scaled_radius = radius * radius_factor
        if scaled_radius < 0.01:
            scaled_radius = 0.01
        self.radius = np.float32(scaled_radius)

        self.radius_sq = np.float32((2.0 * self.radius) * (2.0 * self.radius))

        self.num_colors = num_colors
        self.interaction_strength = interaction_strength

        self.color_interaction = np.zeros((num_colors, num_colors), dtype=np.float32)
        self.particles = np.zeros((self.num_particles, 5), dtype=np.float32)

    def set_interaction_matrix(self, matrix: np.ndarray):
        """
        Sets a custom color interaction matrix.
        The matrix shape must match (num_colors, num_colors).
        """
        if matrix.shape != (self.num_colors, self.num_colors):
            raise ValueError("Matrix has incorrect dimensions.")
        self.color_interaction[:, :] = matrix

    def generate_particles(self) -> None:
        """
        Randomly initializes particle positions, velocities, and colors.
        """
        self.particles[:, 0] = np.random.randint(
            0, self.x_max, self.num_particles
        ).astype(np.float32)
        self.particles[:, 1] = np.random.randint(
            0, self.y_max, self.num_particles
        ).astype(np.float32)
        self.particles[:, 2] = np.random.uniform(
            self.speed_range[0], self.speed_range[1], self.num_particles
        ).astype(np.float32)
        self.particles[:, 3] = np.random.uniform(
            self.speed_range[0], self.speed_range[1], self.num_particles
        ).astype(np.float32)
        self.particles[:, 4] = np.random.randint(
            0, self.num_colors, self.num_particles
        ).astype(np.float32)

    def update_positions(self):
        """
        Performs a single update step:
         1) Build neighbor lists
         2) Compute influence map
         3) Apply influence
         4) Update final positions with collisions, wrap-around, etc.
        """
        neighbor_lists = compute_neighbors_grid(
            self.particles, self.x_max, self.y_max, self.radius
        )
        influence_map = compute_influence_map(
            self.particles,
            self.color_interaction,
            neighbor_lists,
            self.radius,
            self.x_max,
            self.y_max,
            grid_size=100
        )
        apply_influence(
            self.particles,
            influence_map,
            self.radius,
            self.max_speed
        )
        self.particles = update_positions_numba(
            self.particles,
            self.num_particles,
            self.x_max,
            self.y_max,
            self.radius,
            self.radius_sq,
            self.color_interaction,
            self.interaction_strength,
            self.max_speed,
            self.min_speed,
            neighbor_lists
        )

    def get_positions_and_colors(self) -> np.ndarray:
        """
        Returns positions (x, y) and color for each particle in shape (num_particles, 3).
        """
        return np.column_stack((
            self.particles[:, 0],
            self.particles[:, 1],
            self.particles[:, 4]
        ))


@njit(parallel=True)
def compute_neighbors_grid(particles, x_max, y_max, radius):
    """
    Creates neighbor lists based on grid cells.
    """
    num_particles = particles.shape[0]

    MAX_NEIGHBORS = 20
    MAX_PARTICLES_PER_CELL = 20

    neighbor_lists = np.full((num_particles, MAX_NEIGHBORS), -1, dtype=np.int32)

    cell_size = int(2.0 * radius)
    if cell_size < 1:
        cell_size = 1
    
    grid_x = x_max // cell_size + 1
    grid_y = y_max // cell_size + 1

    grid = np.full((grid_x, grid_y, MAX_PARTICLES_PER_CELL), -1, dtype=np.int32)
    grid_counts = np.zeros((grid_x, grid_y), dtype=np.int32)

    for i in prange(num_particles):
        x = particles[i, 0]
        y = particles[i, 1]

        cx = max(0, min(int(x // cell_size), grid_x - 1))
        cy = max(0, min(int(y // cell_size), grid_y - 1))

        ccount = grid_counts[cx, cy]
        if ccount < MAX_PARTICLES_PER_CELL:
            grid[cx, cy, ccount] = i
            grid_counts[cx, cy] = ccount + 1

    max_dist_sq = (2.0 * radius) ** 2 

    for i in prange(num_particles):
        x = particles[i, 0]
        y = particles[i, 1]

        cx = max(0, min(int(x // cell_size), grid_x - 1))
        cy = max(0, min(int(y // cell_size), grid_y - 1))

        ncount = 0
        for gx in range(max(0, cx - 1), min(cx + 2, grid_x)):
            for gy in range(max(0, cy - 1), min(cy + 2, grid_y)):
                limit_count = grid_counts[gx, gy]
                for cidx in range(limit_count):
                    j = grid[gx, gy, cidx]
                    if j == -1 or j == i:
                        continue
                    dx = x - particles[j, 0]
                    dy = y - particles[j, 1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < max_dist_sq:
                        if ncount < MAX_NEIGHBORS:
                            neighbor_lists[i, ncount] = j
                            ncount += 1

    return neighbor_lists

@njit(parallel=True, fastmath=True)
def compute_influence_map(particles, interaction_matrix, neighbor_lists,
                          influence_scale, x_max, y_max, grid_size=100):
    """
    Builds a 2D influence map of shape (grid_size, grid_size, 2).
    """
    num_particles = particles.shape[0]
    influence_map = np.zeros((grid_size, grid_size, 2), dtype=np.float32)

    for i in prange(num_particles):
        x, y, _, _, color = particles[i]
        gx = int(x // influence_scale)
        gy = int(y // influence_scale)
        if gx < 0 or gx >= grid_size or gy < 0 or gy >= grid_size:
            continue

        for k in range(len(neighbor_lists[i])):
            j = neighbor_lists[i][k]
            if j == -1:
                break
            x2, y2, _, _, color2 = particles[j]
            dx = x2 - x
            dy = y2 - y
            dist_sq = dx * dx + dy * dy
            if dist_sq < (influence_scale * influence_scale) and dist_sq > 1e-5:
                dist = math.sqrt(dist_sq)
                inv_dist = 1.0 / dist

                influence = 0.0
                ncols = interaction_matrix.shape[0]
                for col in range(ncols):
                    influence += (interaction_matrix[int(color), col] *
                                  interaction_matrix[col, int(color2)])

                influence_map[gx, gy, 0] += influence * dx * inv_dist
                influence_map[gx, gy, 1] += influence * dy * inv_dist

    return influence_map

@njit(parallel=True, fastmath=True)
def apply_influence(particles, influence_map, influence_scale, max_speed):
    """
    Adjusts each particle's velocity based on the local influence map cell.
    """
    num_particles = particles.shape[0]

    for i in prange(num_particles):
        x, y, vx, vy, _ = particles[i]
        gx = int(x // influence_scale)
        gy = int(y // influence_scale)

        if gx < 0:
            gx = 0
        elif gx >= influence_map.shape[0]:
            gx = influence_map.shape[0] - 1
        if gy < 0:
            gy = 0
        elif gy >= influence_map.shape[1]:
            gy = influence_map.shape[1] - 1

        fx, fy = influence_map[gx, gy]

        vx += min(max(fx * 0.1, -max_speed / 2), max_speed / 2)
        vy += min(max(fy * 0.1, -max_speed / 2), max_speed / 2)
        vx, vy = limit_speed(vx, vy, max_speed, 0.1)

        particles[i, 2] = vx
        particles[i, 3] = vy

    return particles

@njit(parallel=True, fastmath=True)
def update_positions_numba(
    particles,
    num_particles,
    x_max,
    y_max,
    radius,
    radius_sq,
    interaction_matrix,
    interaction_strength,
    max_speed,
    min_speed,
    neighbor_lists,
):
    """
    Performs the final position update for all particles, including
    color-based forces, velocity limiting, wrap-around, and collision handling.
    """
    for i in prange(num_particles):
        x, y, vx, vy, color = particles[i]

        fx, fy = compute_forces_with_neighbors(
            i, particles, neighbor_lists, interaction_matrix,
            interaction_strength, radius_sq
        )

        vx += fx
        vy += fy
        vx, vy = limit_speed(vx, vy, max_speed, min_speed)

        x_new = x + vx
        y_new = y + vy

        x_new %= x_max
        y_new %= y_max

        x_new, y_new = handle_collisions(
            i, x_new, y_new, radius, radius_sq, particles, neighbor_lists
        )

        x_new %= x_max
        y_new %= y_max

        particles[i, 0] = x_new
        particles[i, 1] = y_new
        particles[i, 2] = vx
        particles[i, 3] = vy
        particles[i, 4] = color

    return particles

@njit(fastmath=True)
def compute_forces_with_neighbors(idx, particles, neighbor_lists,
                                  interaction_matrix, interaction_strength, radius_sq):
    """
    Same logic as compute_forces, but only over neighbor_lists.
    This saves iterating again over all particles and redundant distance computations.
    """
    fx, fy = 0.0, 0.0
    x, y, _, _, color = particles[idx]

    for j in neighbor_lists[idx]:
        if j == -1:
            break
        if j == idx:
            continue

        x2, y2, _, _, color2 = particles[j]
        dx = x2 - x
        dy = y2 - y
        dist_sq = dx * dx + dy * dy

        if 0.0 < dist_sq < radius_sq:
            dist = math.sqrt(dist_sq)
            force = interaction_matrix[int(color), int(color2)] * interaction_strength
            damping_factor = min(1.0, (dist / (math.sqrt(radius_sq) * 0.6)) ** 1.5)
            fx += force * (dx / dist) * damping_factor
            fy += force * (dy / dist) * damping_factor

    return fx, fy

@njit(fastmath=True)
def limit_speed(vx, vy, max_speed, min_speed):
    """
    Keeps velocity between min_speed and max_speed.
    """
    speed = math.sqrt(vx * vx + vy * vy)
    if speed > max_speed:
        vx = (vx / speed) * max_speed
        vy = (vy / speed) * max_speed
    elif 0.0 < speed < min_speed:
        vx = (vx / speed) * min_speed
        vy = (vy / speed) * min_speed
    return vx, vy

@njit(fastmath=True)
def handle_collisions(i, x_new, y_new, radius, radius_sq, particles, neighbor_lists):
    """
    Resolves collisions by shifting particle i away from overlapping neighbors.
    """
    for j in neighbor_lists[i]:
        if j == -1:
            break
        dx = particles[j, 0] - x_new
        dy = particles[j, 1] - y_new
        dist_sq = dx * dx + dy * dy
        if dist_sq < radius_sq:
            dist = max(math.sqrt(dist_sq), 1e-8)
            overlap = (2.0 * radius) - dist
            nx = dx / dist
            ny = dy / dist
            x_new -= overlap * nx
            y_new -= overlap * ny

    return x_new, y_new