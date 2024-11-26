import numpy as np

class Particle:
    def __init__(self, position, velocity, particle_type, color, interaction_strength, influence_radius, friction, random_motion):
        """
        Initialisiert die Attribute des Partikels mit den gegebenen Parametern.
       """
        self.position = np.array(position, dtype=float)  # Position of the particle
        self.velocity = np.array(velocity, dtype=float)  # Velocity of the particle
        self.type = particle_type  # Type of the particle
        self.color = color  # Color of the particle
        self.interaction_strength = interaction_strength  # Strength of interactions
        self.influence_radius = influence_radius  # Influence radius for interactions
        self.friction = friction  # Friction coefficient
        self.random_motion = random_motion  # Random motion intensity for unpredictability

    def update_position(self, delta_time):
        """
        Aktualisiert die Position des Partikels basierend auf der Geschwindigkeit und der Zeitdifferenz.
        """
        self.position += self.velocity * delta_time

    def apply_interaction(self, other_particle):
        """
        Berechnet die Kraft zwischen diesem Partikel und einem anderen und passt die Geschwindigkeit an.
        """
        # Placeholder logic for interaction (e.g., gravitational force between two particles)
        distance = np.linalg.norm(self.position - other_particle.position)
        if distance < self.influence_radius:
            force = self.interaction_strength / (distance ** 2)
            direction = (other_particle.position - self.position) / distance  # Unit vector towards the other particle
            self.velocity += force * direction

    def apply_friction(self):
        """
        Reduziert die Geschwindigkeit basierend auf der Reibung.
        """
        self.velocity *= (1 - self.friction)

    def randomize_movement(self):
        """
        Fügt der Geschwindigkeit zufällige Bewegungen hinzu, um natürliche Zufälligkeit zu simulieren.
        """
        random_velocity = (np.random.rand(2) - 0.5) * self.random_motion
        self.velocity += random_velocity

    @staticmethod
    def resolve_collisions(particles):
        """
        Überprüft, ob sich mehr als zwei Partikel an derselben Position befinden,
        und verschiebt sie, falls notwendig.
        """
        # Simple collision detection (can be extended later)
        for i, p1 in enumerate(particles):
            for j, p2 in enumerate(particles):
                if i != j and np.array_equal(p1.position, p2.position):
                    # Resolve the collision by slightly moving the particles apart
                    p1.position += np.random.rand(2) * 0.1
                    p2.position += np.random.rand(2) * 0.1

    def __str__(self):
        """
        Return a string representation of the particle for debugging purposes.
        """
        return f"Particle(Type: {self.type}, Position: {self.position}, Velocity: {self.velocity}, Color: {self.color})"

# Example usage:

# Create a particle
particle1 = Particle(position=[1.0, 2.0],
                     velocity=[0.5, 0.5],
                     particle_type="TypeA",
                     color="red",
                     interaction_strength=10,
                     influence_radius=5,
                     friction=0.01,
                     random_motion=0.2)

# Update the particle's position
particle1.update_position(delta_time=0.1)

# Apply random movement
particle1.randomize_movement()

# Apply friction
particle1.apply_friction()

# Print particle state
print(particle1)
