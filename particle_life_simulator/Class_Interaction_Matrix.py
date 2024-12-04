class Interaction_Matrix:
    def __init__(self, particles_df, max_radius):
        self.particles_df = particles_df
        self.max_radius = max_radius

    def calculate(self):
        positions = self.particles_df[["x", "y"]].to_numpy()
        deltas = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.linalg.norm(deltas, axis=2)

        # Interaction matrix logic
        interaction_matrix = np.zeros((len(self.particles_df), len(self.particles_df)))
        for i in range(len(self.particles_df)):
            for j in range(len(self.particles_df)):
                if i != j and distances[i, j] < self.max_radius:
                    interaction_matrix[i, j] = 1 / distances[i, j]  # Example: Inverse distance
        return interaction_matrix

    
import pandas as pd
import numpy as np

# Example DataFrame with particle attributes
num_particles = 5
data = {
    "ID": np.arange(num_particles),
    "x": np.random.uniform(0, 800, num_particles),
    "y": np.random.uniform(0, 600, num_particles),
    "vx": np.random.uniform(-2, 2, num_particles),
    "vy": np.random.uniform(-2, 2, num_particles),
    "radius": np.random.uniform(10, 30, num_particles),
    "mass": np.random.uniform(1, 5, num_particles),
    "color": [tuple(np.random.randint(0, 256, 3)) for _ in range(num_particles)],
}
particles_df = pd.DataFrame(data)

print(particles_df)




interaction = Interaction_Matrix(particles_df, max_radius=100)
interaction_matrix = interaction.calculate()
print(interaction_matrix)