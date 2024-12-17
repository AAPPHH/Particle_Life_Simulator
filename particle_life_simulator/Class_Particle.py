import random
import math
from Class_Quadtree import Quadtree


class CreateParticle:
    def __init__(self, num_particles: int = 1000, x_max: int = 1920, y_max: int = 1080,
                 speed_range: tuple = (-2, 2), radius: int = 5):
        """
        Initialisiere die Partikel-Simulation.
        """
        self.num_particles = num_particles
        self.x_max = x_max
        self.y_max = y_max
        self.speed_range = speed_range
        self.particles = []  # Liste der Partikel [(x, y, vx, vy)]
        self.radius = radius
        self.quadtree = Quadtree(0, 0, x_max, y_max)  # Quadtree zur Optimierung

    def generate_particles(self) -> None:
        """
        Generiere zufällige Partikel, die sich nicht überlappen.
        """
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
        """
        Aktualisiere die Positionen der Partikel.
        - Abstandsabhängiger Einfluss
        - Collision-Handling bei direkten Kollisionen
        """
        updated_particles = []

        for i, (x1, y1, vx1, vy1) in enumerate(self.particles):
            # Bewege das Partikel entsprechend seiner Geschwindigkeit
            x1 += vx1
            y1 += vy1

            # Stelle sicher, dass das Partikel innerhalb der Grenzen bleibt
            x1, y1 = self._handle_boundaries(x1, y1)

            # Finde benachbarte Partikel im Quadtree
            nearby_particles = self.quadtree.query(
                x1 - 2 * self.radius, y1 - 2 * self.radius,
                x1 + 2 * self.radius, y1 + 2 * self.radius
            )

            total_dx, total_dy = 0, 0  # Summe der Einflüsse initialisieren

            for (x2, y2, vx2, vy2) in nearby_particles:
                if (x1, y1) != (x2, y2):  # Vermeide Einfluss von sich selbst
                    dist = self._distance(x1, y1, x2, y2)

                    # Abstandsabhängiger Einfluss berechnen
                    if dist < 2 * self.radius:
                        influence = 1 / (dist + 0.01)  # Je näher, desto stärker der Einfluss
                        dx = (x1 - x2) * influence
                        dy = (y1 - y2) * influence

                        total_dx += dx
                        total_dy += dy

                        # Collision-Handling bei Überlappung
                        if dist < 2 * self.radius:
                            vx1, vy1, vx2, vy2 = self._handle_collision(x1, y1, vx1, vy1, x2, y2, vx2, vy2)

            # Aktualisiere Position basierend auf Einflüssen
            x1 += total_dx * 0.1  # Dämpfung des Einflusses
            y1 += total_dy * 0.1

            updated_particles.append((x1, y1, vx1, vy1))

        self.particles = updated_particles
        self.update_quadtree()

    def update_quadtree(self):
        """
        Aktualisiere den Quadtree mit den neuen Positionen.
        """
        self.quadtree = Quadtree(0, 0, self.x_max, self.y_max)
        for particle in self.particles:
            self.quadtree.insert(particle)

    def _handle_boundaries(self, x: float, y: float) -> tuple:
        """
        Stelle sicher, dass Partikel innerhalb der Grenzen bleiben.
        """
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
        """
        Berechne elastische Kollision zwischen zwei Partikeln.
        """
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)

        if distance == 0:
            return vx1, vy1, vx2, vy2

        # Berechne die Normalenrichtung
        nx = dx / distance
        ny = dy / distance

        # Berechne relative Geschwindigkeit
        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Skalarprodukt der Geschwindigkeit in Richtung der Normalen
        dot_product = dvx * nx + dvy * ny

        if dot_product > 0:
            return vx1, vy1, vx2, vy2

        # Passe Geschwindigkeiten an (Impulsübertragung)
        vx1 -= dot_product * nx
        vy1 -= dot_product * ny
        vx2 += dot_product * nx
        vy2 += dot_product * ny

        return vx1, vy1, vx2, vy2

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Berechne die euklidische Distanz zwischen zwei Punkten.
        """
        return math.hypot(x2 - x1, y2 - y1)

    def get_positions(self) -> list:
        """
        Gib die Positionen der Partikel zurück.
        """
        return [(x, y) for x, y, _, _ in self.particles]
