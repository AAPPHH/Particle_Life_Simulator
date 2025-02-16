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
    Represents a particle system with color-based interactions.

    This class manages particle attributes (positions, velocities, and colors)
    and updates them according to various interaction rules. Interactions
    are partially determined by an interaction matrix that encodes how
    different colors influence each other.

    Attributes:
        num_particles (int32): Total number of particles.
        x_max (int32): Maximum x-dimension (e.g., screen width).
        y_max (int32): Maximum y-dimension (e.g., screen height).
        speed_range (float32[:]): Range from which initial velocities are drawn (min, max).
        max_speed (float32): Maximum allowed speed for any particle.
        min_speed (float32): Minimum speed below which velocities will be corrected upward.
        radius (float32): Radius used to determine the interaction neighborhood.
        radius_sq (float32): The square of an effective interaction diameter (used to speed up distance checks).
        num_colors (int32): Number of distinct colors in the system.
        interaction_strength (float32): Scaling factor for inter-particle color-based forces.
        color_interaction (float32[:, :]): A 2D matrix defining interaction coefficients between colors.
        particles (float32[:, :]): Array of shape (num_particles, 5) storing particle data:
            - [:, 0]: x-position
            - [:, 1]: y-position
            - [:, 2]: x-velocity
            - [:, 3]: y-velocity
            - [:, 4]: color index
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
        """
        Initializes the CreateParticle system and allocates memory for particles.

        Args:
            num_particles (int, optional): Number of particles to create.
            x_max (int, optional): Maximum x-dimension.
            y_max (int, optional): Maximum y-dimension.
            speed_range (tuple, optional): Range for initial random velocities (min, max).
            max_speed (float, optional): Maximum allowed speed.
            min_speed (float, optional): Minimum allowed speed.
            radius (float, optional): Base radius for interaction neighborhoods.
            num_colors (int, optional): Number of distinct colors.
            interaction_strength (float, optional): Scaling factor for color-based forces.
            radius_factor (float, optional): Factor to scale the interaction radius.

        Raises:
            ValueError: If scaled_radius becomes too small (less than 0.01).
        """
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

        # The effective squared "diameter" is used for collision checks
        self.radius_sq = np.float32((2.0 * self.radius) * (2.0 * self.radius))

        self.num_colors = num_colors
        self.interaction_strength = interaction_strength

        self.color_interaction = np.zeros((num_colors, num_colors), dtype=np.float32)
        self.particles = np.zeros((self.num_particles, 5), dtype=np.float32)

    def set_interaction_matrix(self, matrix: np.ndarray):
        """
        Sets a custom color interaction matrix.

        Args:
            matrix (np.ndarray): A 2D matrix of shape (num_colors, num_colors)
                that specifies color interaction coefficients.

        Raises:
            ValueError: If the provided matrix does not have the shape (num_colors, num_colors).
        """
        if matrix.shape != (self.num_colors, self.num_colors):
            raise ValueError("Matrix has incorrect dimensions.")
        self.color_interaction[:, :] = matrix

    def generate_particles(self) -> None:
        """
        Initializes each particle with random positions, velocities, and colors.
        
        Notes:
            - Positions are randomly chosen in [0, x_max) for x and in [0, y_max) for y.
            - Velocities are sampled from the provided speed_range.
            - Colors are randomly assigned from 0 to num_colors-1.
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
        Performs a single update step on all particles:
        
        1) Constructs neighbor lists using a grid-based approach.
        2) Computes an influence map over the domain (based on local interactions).
        3) Applies the influence to modify particle velocities.
        4) Updates final positions with collision handling and wrap-around at borders.
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
        Returns a view of each particle's (x, y) position and color.

        Returns:
            np.ndarray: An array of shape (num_particles, 3) where each row is (x, y, color).
        """
        return np.column_stack((
            self.particles[:, 0],
            self.particles[:, 1],
            self.particles[:, 4]
        ))


@njit(parallel=True)
def compute_neighbors_grid(particles, x_max, y_max, radius):
    """
    Generates neighbor lists for each particle based on a grid partition.

    Cells are sized by 2*radius. Each particle is placed in a cell
    depending on its (x, y) coordinates. Only nearby cells (in a 3x3 region
    around each particle's cell) are searched to find potential neighbors.

    Args:
        particles (np.ndarray): Particle array of shape (N, 5).
        x_max (int): Maximum x-dimension (width).
        y_max (int): Maximum y-dimension (height).
        radius (float): The neighborhood radius.

    Returns:
        np.ndarray: A 2D array (N, max_neighbors) that stores the indices of each particle's neighbors.
                    Unused neighbor slots are filled with -1.
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

    # Place particles in the appropriate grid cell
    for i in prange(num_particles):
        x = particles[i, 0]
        y = particles[i, 1]

        cx = max(0, min(int(x // cell_size), grid_x - 1))
        cy = max(0, min(int(y // cell_size), grid_y - 1))

        ccount = grid_counts[cx, cy]
        if ccount < MAX_PARTICLES_PER_CELL:
            grid[cx, cy, ccount] = i
            grid_counts[cx, cy] = ccount + 1

    # Find neighbors in adjacent cells
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
    Computes a coarse "influence map" over a grid of size (grid_size x grid_size).

    Influence is calculated by summing interaction forces between each
    particle and its neighbors in a small local region. The map is used to
    guide velocity adjustments in a subsequent step.

    Args:
        particles (np.ndarray): Array of shape (N, 5) containing particle data.
        interaction_matrix (np.ndarray): The color interaction matrix (num_colors x num_colors).
        neighbor_lists (np.ndarray): A 2D array of neighbor indices for each particle.
        influence_scale (float): The spatial scaling for the influence grid cells.
        x_max (int): Maximum x-dimension (width).
        y_max (int): Maximum y-dimension (height).
        grid_size (int, optional): The resolution of the influence map.

    Returns:
        np.ndarray: A float32 3D array (grid_size, grid_size, 2), where
                    [:, :, 0] is the x-influence component and
                    [:, :, 1] is the y-influence component.
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
            dist_sq = dx*dx + dy*dy
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
    Adjusts each particle's velocity based on the local influence map.

    Args:
        particles (np.ndarray): Array of shape (N, 5) containing particle data.
        influence_map (np.ndarray): The influence map from `compute_influence_map()`.
        influence_scale (float): Cell size for accessing the map.
        max_speed (float): Maximum allowed speed for any particle.

    Returns:
        np.ndarray: The modified `particles` array with updated velocities.
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
    Finalizes the update of particle positions and velocities, including:
    - Color-based force application.
    - Speed limiting.
    - Wrap-around at the domain edges.
    - Collision handling (pushing particles apart if overlapping).

    Args:
        particles (np.ndarray): Particle data array of shape (N, 5).
        num_particles (int): Total number of particles.
        x_max (int): Maximum x-dimension (width).
        y_max (int): Maximum y-dimension (height).
        radius (float): Interaction radius for collisions.
        radius_sq (float): Square of the collision interaction diameter.
        interaction_matrix (np.ndarray): Color interaction coefficients.
        interaction_strength (float): Global scaling for interaction forces.
        max_speed (float): Maximum velocity magnitude.
        min_speed (float): Minimum velocity magnitude.
        neighbor_lists (np.ndarray): Neighbor indices for each particle.

    Returns:
        np.ndarray: Updated `particles` array after applying interactions and constraints.
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

        # Wrap-around
        x_new %= x_max
        y_new %= y_max

        # Collision handling
        x_new, y_new = handle_collisions(
            i, x_new, y_new, radius, radius_sq, particles, neighbor_lists
        )

        # Wrap-around again after collision adjustments
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
    Computes the net force on a given particle from its neighbors.

    The force is calculated based on color interactions and distance,
    scaled by `interaction_strength`. Only particles within `neighbor_lists[idx]`
    are considered, significantly reducing complexity.

    Args:
        idx (int): Index of the particle for which to compute the force.
        particles (np.ndarray): Particle data array of shape (N, 5).
        neighbor_lists (np.ndarray): Array of neighbor indices for each particle.
        interaction_matrix (np.ndarray): Color interaction matrix (num_colors x num_colors).
        interaction_strength (float): Global scale for color forces.
        radius_sq (float): Squared diameter for collision/interaction checks.

    Returns:
        Tuple[float, float]: (fx, fy), the total force in x and y directions on the particle.
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
    Clamps the speed of a velocity vector to be within [min_speed, max_speed].

    Args:
        vx (float): Velocity in x-direction.
        vy (float): Velocity in y-direction.
        max_speed (float): Maximum speed allowed.
        min_speed (float): Minimum speed allowed.

    Returns:
        Tuple[float, float]: (vx, vy), the corrected velocity.
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
    Resolves collisions by pushing overlapping particles apart.

    For the particle at index `i`, checks all neighbors in `neighbor_lists[i]`
    and adjusts the new position (x_new, y_new) to avoid overlap.

    Args:
        i (int): Index of the current particle.
        x_new (float): Proposed new x-position for this particle.
        y_new (float): Proposed new y-position for this particle.
        radius (float): Particle collision radius.
        radius_sq (float): Squared collision diameter for overlap checks.
        particles (np.ndarray): Array of particle data.
        neighbor_lists (np.ndarray): Array of neighbor indices for each particle.

    Returns:
        Tuple[float, float]: (x_corr, y_corr), the potentially corrected position.
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
