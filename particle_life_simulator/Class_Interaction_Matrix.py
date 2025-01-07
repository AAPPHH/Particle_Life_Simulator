import random

class InteractionMatrix:
    def __init__(self, num_colors: int, default_value: int = 0):
        """
        Erstellt eine quadratische Interaktionsmatrix für Partikel-Farben.

        :param num_colors: Anzahl der Farben (Matrix-Dimension).
        :param default_value: Standardwert für Interaktionen (z. B. 0 für keine Interaktion).
        """
        self.num_colors = num_colors
        self.matrix = [
            [default_value for _ in range(num_colors)]
            for _ in range(num_colors)
        ]

    def set_interaction(self, color1: int, color2: int, value: int):
        """
        Setzt den Interaktionswert zwischen zwei Farben.

        :param color1: Index der ersten Farbe.
        :param color2: Index der zweiten Farbe.
        :param value: Interaktionswert (z. B. -1 für Abstoßung, +1 für Anziehung).
        """
        self.matrix[color1][color2] = value
        self.matrix[color2][color1] = value  # Symmetrische Interaktion

    def set_full_matrix(self, matrix: list):
        """
        Setzt die gesamte Interaktionsmatrix.

        :param matrix: Eine Liste von Listen, die die Interaktionen darstellt.
        """
        if len(matrix) != self.num_colors or any(len(row) != self.num_colors for row in matrix):
            raise ValueError("Die Matrix muss quadratisch sein und der Anzahl der Farben entsprechen.")
        self.matrix = matrix

    def get_interaction(self, color1: int, color2: int) -> int:
        """
        Gibt den Interaktionswert zwischen zwei Farben zurück.

        :param color1: Index der ersten Farbe.
        :param color2: Index der zweiten Farbe.
        :return: Interaktionswert.
        """
        return self.matrix[color1][color2]

    def randomize_interactions(self, values: list):
        """
        Initialisiert die Matrix mit zufälligen Werten aus einer Liste.

        :param values: Liste möglicher Interaktionswerte (z. B. [-1, 0, +1]).
        """
        for i in range(self.num_colors):
            for j in range(i, self.num_colors):
                value = random.choice(values)
                self.set_interaction(i, j, value)

    def display(self):
        """Zeigt die Interaktionsmatrix auf der Konsole an."""
        print("Interaction Matrix:")
        for row in self.matrix:
            print(" ".join(f"{val:+}" for val in row))
