import random

class InteractionMatrix:
    def __init__(self, num_colors: int, default_value: int = 0):
        """
        Creates a square interaction matrix for particle colors.

        :param num_colors: Number of colors (matrix dimension).
        :param default_value: Default value for interactions (e.g., 0 for no interaction).
        """
        self.num_colors = num_colors
        self.matrix = [
            [default_value for _ in range(num_colors)]
            for _ in range(num_colors)
        ]

    def set_interaction(self, color1: int, color2: int, value: int):
        """
        Sets the interaction value between two colors.

        :param color1: Index of the first color.
        :param color2: Index of the second color.
        :param value: Interaction value (e.g., -1 for repulsion, +1 for attraction).
        """
        self.matrix[color1][color2] = value
        self.matrix[color2][color1] = value  # Symmetric interaction

    def set_full_matrix(self, matrix: list):
        """
        Sets the entire interaction matrix.

        :param matrix: A list of lists representing the interactions.
        """
        if len(matrix) != self.num_colors or any(len(row) != self.num_colors for row in matrix):
            raise ValueError("The matrix must be square and match the number of colors.")
        self.matrix = matrix

    def get_interaction(self, color1: int, color2: int) -> int:
        """
        Returns the interaction value between two colors.

        :param color1: Index of the first color.
        :param color2: Index of the second color.
        :return: Interaction value.
        """
        return self.matrix[color1][color2]

    def randomize_interactions(self, values: list):
        """
        Initializes the matrix with random values from a given list.

        :param values: List of possible interaction values (e.g., [-1, 0, +1]).
        """
        for i in range(self.num_colors):
            for j in range(i, self.num_colors):
                value = random.choice(values)
                self.set_interaction(i, j, value)

    def display(self):
        """Displays the interaction matrix on the console."""
        print("Interaction Matrix:")
        for row in self.matrix:
            print(" ".join(f"{val:+}" for val in row))
