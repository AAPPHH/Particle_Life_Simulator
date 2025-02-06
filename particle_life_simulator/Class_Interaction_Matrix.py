import random
import numpy as np
class InteractionMatrix:
    def __init__(self, num_colors: int, default_value: int = 0):
        """
        Creates a square interaction matrix for particle colors.

        Args:
            num_colors (int): Number of colors (matrix dimension).
            default_value (int): Default value for interactions (e.g., 0 for no interaction).
        """
        self.num_colors = num_colors
        self.matrix = [[default_value for _ in range(num_colors)] for _ in range(num_colors)]

    def set_interaction(self, color1: int, color2: int, value: int):
        """
        Sets the interaction value between two colors.

        Args:
            color1 (int): Index of the first color.
            color2 (int): Index of the second color.
            value (int): Interaction value (e.g., -1 for repulsion, +1 for attraction).
        """
        self.matrix[color1][color2] = value
        self.matrix[color2][color1] = value  # Symmetric interaction

    def get_interaction(self, color1: int, color2: int) -> int:
        """
        Returns the interaction value between two colors.

        Args:
            color1 (int): Index of the first color.
            color2 (int): Index of the second color.
            Returns:
                int: Interaction value.
        """
        return self.matrix[color1][color2]

    def randomize_interactions(self, values: list):
        """
        Initializes the matrix with random values from a given list.

        Args:
            values (list): List of possible interaction values (e.g., [-1, 0, +1]).
        """
        for i in range(self.num_colors):
            for j in range(i, self.num_colors):
                value = random.choice(values)
                self.set_interaction(i, j, value)

    def get_matrix(self):
        """Returns the full interaction matrix as a NumPy array."""
        return np.array(self.matrix, dtype=np.float32)

    
    
    
