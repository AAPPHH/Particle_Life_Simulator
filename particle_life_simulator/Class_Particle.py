import random
import math

class CreateParticle:
    def __init__(self, num_particles: int = 1000, x_max: int = 1920, y_max: int = 1080,
                 speed_range: tuple = (-2, 2), radius: int = 5):
        """
        Initializes the CreateParticle class with the given parameters.

        Args:
            num_particles (int): Number of particles to generate.
            x_max (int): Maximum x-coordinate for the particles.
            y_max (int): Maximum y-coordinate for the particles.
            speed_range (tuple): Range of initial speeds for the particles.
            radius (int): Radius of each particle.
        """
        self.num_particles = num_particles
        self.x_max = x_max
        self.y_max = y_max
        self.speed_range = speed_range
        self.particles = []
        self.radius = radius

    def generate_particles(self) -> None:
        """
        Generates particles with random positions and velocities, ensuring they don't overlap initially.
        """
        self.particles = []
        while len(self.particles) < self.num_particles:
            x = random.randint(self.radius, self.x_max - self.radius)
            y = random.randint(self.radius, self.y_max - self.radius)
            vx = random.uniform(*self.speed_range)
            vy = random.uniform(*self.speed_range)

            # Check overlap
            if all(self._distance(x, y, p['x'], p['y']) >= 2 * self.radius for p in self.particles):
                self.particles.append({'x': x, 'y': y, 'vx': vx, 'vy': vy})

    def update_positions(self) -> None:
        """
        Updates the positions of all particles, handles collisions and boundary conditions.
        """
        for particle in self.particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']

            # Handle boundaries
            if particle['x'] - self.radius < 0:
                particle['x'] = self.x_max - self.radius
            elif particle['x'] + self.radius > self.x_max:
                particle['x'] = self.radius

            if particle['y'] - self.radius < 0:
                particle['y'] = self.y_max - self.radius
            elif particle['y'] + self.radius > self.y_max:
                particle['y'] = self.radius

            # Check collisions
            for other in self.particles:
                if particle is not other and self._distance(particle['x'], particle['y'], other['x'], other['y']) < 2 * self.radius:
                    self._handle_collision(particle, other)

    def get_positions(self) -> list:
        """
        Retrieves the current positions of all particles.

        Returns:
            list: A list of dictionaries containing the x and y positions of each particle.
        """
        return [{'x': p['x'], 'y': p['y']} for p in self.particles]

    def _handle_collision(self, p1: dict, p2: dict) -> None:
        """
        Handles elastic collision between two particles.

        Args:
            p1 (dict): First particle.
            p2 (dict): Second particle.
        """
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        distance = math.hypot(dx, dy)

        if distance == 0:
            return

        nx = dx / distance
        ny = dy / distance

        dvx = p1['vx'] - p2['vx']
        dvy = p1['vy'] - p2['vy']

        dot_product = dvx * nx + dvy * ny

        if dot_product > 0:
            return

        # Adjust velocities
        p1['vx'] -= dot_product * nx
        p1['vy'] -= dot_product * ny
        p2['vx'] += dot_product * nx
        p2['vy'] += dot_product * ny

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculates the Euclidean distance between two points.

        Args:
            x1 (float): X-coordinate of the first point.
            y1 (float): Y-coordinate of the first point.
            x2 (float): X-coordinate of the second point.
            y2 (float): Y-coordinate of the second point.

        Returns:
            float: The distance between the two points.
        """
        return math.hypot(x2 - x1, y2 - y1)
