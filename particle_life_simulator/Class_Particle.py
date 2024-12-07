import random
import math

from Class_Quadtree import Quadtree

class CreateParticle:
    def __init__(self, num_particles: int = 1000, x_max: int = 1920, y_max: int = 1080,
                 speed_range: tuple = (-2, 2), radius: int = 5):
        self.num_particles = num_particles
        self.x_max = x_max
        self.y_max = y_max
        self.speed_range = speed_range
        self.particles = []
        self.radius = radius
        self.quadtree = Quadtree(0, 0, x_max, y_max)

    def generate_particles(self) -> None:
        self.particles = []
        while len(self.particles) < self.num_particles:
            x = random.randint(self.radius, self.x_max - self.radius)
            y = random.randint(self.radius, self.y_max - self.radius)
            vx = random.uniform(*self.speed_range)
            vy = random.uniform(*self.speed_range)

            if all(self._distance(x, y, px, py) >= 2 * self.radius for px, py, _, _ in self.particles):
                self.particles.append((x, y, vx, vy))
        self.update_quadtree()

    def update_positions(self) -> None:
        updated_particles = []

        for i, (x1, y1, vx1, vy1) in enumerate(self.particles):
            x1 += vx1
            y1 += vy1

            x1, y1 = self._handle_boundaries(x1, y1)

            nearby_particles = self.quadtree.query(
                x1 - 2 * self.radius, y1 - 2 * self.radius,
                x1 + 2 * self.radius, y1 + 2 * self.radius
            )

            for (x2, y2, vx2, vy2) in nearby_particles:
                if self._distance(x1, y1, x2, y2) < 2 * self.radius:
                    vx1, vy1, vx2, vy2 = self._handle_collision(x1, y1, vx1, vy1, x2, y2, vx2, vy2)

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

            updated_particles.append((x1, y1, vx1, vy1))

        self.particles = updated_particles
        self.update_quadtree()

    def update_quadtree(self):
        self.quadtree = Quadtree(0, 0, self.x_max, self.y_max)
        for particle in self.particles:
            self.quadtree.insert(particle)

    def _handle_boundaries(self, x: float, y: float) -> tuple:
        if x - self.radius < 0:
            x = self.x_max - self.radius
        elif x + self.radius > self.x_max:
            x = self.radius

        if y - self.radius < 0:
            y = self.y_max - self.radius
        elif y + self.radius > self.y_max:
            y = self.radius

        return x, y

    def _handle_collision(self, x1: float, y1: float, vx1: float, vy1: float,
                          x2: float, y2: float, vx2: float, vy2: float) -> tuple:
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
        return math.hypot(x2 - x1, y2 - y1)

    def get_positions(self) -> list:
        return [(x, y) for x, y, _, _ in self.particles]