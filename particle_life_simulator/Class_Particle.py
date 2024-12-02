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

            if all(self._distance(x, y, px, py) >= 2 * self.radius for px, py, _, _ in self.particles):
                self.particles.append((x, y, vx, vy))

    def update_positions(self) -> None:
        """
        Updates the positions of all particles, handles collisions and boundary conditions.
        """
        updated_particles = []

        for i, (x1, y1, vx1, vy1) in enumerate(self.particles):
            x1 += vx1
            y1 += vy1

            # Handle horizontal boundary conditions (wrap around)
            if x1 - self.radius < 0:
                x1 = self.x_max - self.radius

            if x1 + self.radius > self.x_max:
                x1 = self.radius

            # Handle vertical boundary conditions (wrap around)
            if y1 - self.radius < 0:
                y1 = self.y_max - self.radius

            if y1 + self.radius > self.y_max:
                y1 = self.radius

            # Check for collisions with other particles
            for j, (x2, y2, vx2, vy2) in enumerate(self.particles):
                if i != j and self._distance(x1, y1, x2, y2) < 2 * self.radius:
                    vx1, vy1, vx2, vy2 = self._handle_collision(x1, y1, vx1, vy1, x2, y2, vx2, vy2)

                    # Resolve overlap
                    overlap = 2 * self.radius - self._distance(x1, y1, x2, y2)
                    if overlap > 0:
                        distance = self._distance(x1, y1, x2, y2)
                        if distance == 0:
                            distance = 0.1
                        separation_vector_x = (x1 - x2) / distance
                        separation_vector_y = (y1 - y2) / distance
                        x1 += separation_vector_x * overlap / 2
                        y1 += separation_vector_y * overlap / 2
                        x2 -= separation_vector_x * overlap / 2
                        y2 -= separation_vector_y * overlap / 2

                    self.particles[j] = (x2, y2, vx2, vy2)

            updated_particles.append((x1, y1, vx1, vy1))

        self.particles = updated_particles

    def get_positions(self) -> list:
        """
        Retrieves the current positions of all particles.

        Returns:
            list: A list of tuples containing the x and y positions of each particle.
        """
        return [(x, y) for x, y, _, _ in self.particles]

    def _handle_collision(self, x1: float, y1: float, vx1: float, vy1: float,
                          x2: float, y2: float, vx2: float, vy2: float) -> tuple:
        """
        Calculates new velocities after an elastic collision between two particles.

        Args:
            x1 (float): X-coordinate of the first particle.
            y1 (float): Y-coordinate of the first particle.
            vx1 (float): Velocity in x-direction of the first particle.
            vy1 (float): Velocity in y-direction of the first particle.
            x2 (float): X-coordinate of the second particle.
            y2 (float): Y-coordinate of the second particle.
            vx2 (float): Velocity in x-direction of the second particle.
            vy2 (float): Velocity in y-direction of the second particle.

        Returns:
            tuple: Updated velocities (vx1, vy1, vx2, vy2) after collision.
        """
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)

        if distance == 0:
            return vx1, vy1, vx2, vy2

        nx = dx / distance
        ny = dy / distance

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        dot_product = dvx * nx + dvy * ny

        if dot_product > 0:
            return vx1, vy1, vx2, vy2

        vx1 -= dot_product * nx
        vy1 -= dot_product * ny
        vx2 += dot_product * nx
        vy2 += dot_product * ny

        return vx1, vy1, vx2, vy2

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
