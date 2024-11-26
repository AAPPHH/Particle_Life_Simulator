import numpy as np
import pygame
class Gui_Manager:
    def __init__(self, simulation, width=800, height=600, scale=8):
        """
        Initialize the GUI manager with the simulation instance.
        """
        self.simulation = simulation
        self.width = width
        self.height = height
        self.scale = scale
        self.array_shape = (height // scale, width // scale)
        self.bg_color = (30, 30, 30)
        self.screen = None
        self.clock = pygame.time.Clock()

    def initialize(self):
        """
        Initialize Pygame and create the window.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Numpy Array Visualization")

    def render(self):
        """
        Render the simulation onto the screen.
        """
        array = np.zeros(self.array_shape, dtype=np.uint8)

        for particle in self.simulation.particles:
            x, y = int(particle.position[0]), int(particle.position[1])
            if 0 <= x < self.array_shape[1] and 0 <= y < self.array_shape[0]:
                array[y, x] = 255

        self.screen.fill(self.bg_color)

        for y in range(self.array_shape[0]):
            for x in range(self.array_shape[1]):
                color_value = array[y, x]
                color = (color_value, color_value, color_value)
                rect = pygame.Rect(x * self.scale, y * self.scale, self.scale, self.scale)
                self.screen.fill(color, rect)

        for particle in self.simulation.particles:
            x, y = particle.position * self.scale
            rect = pygame.Rect(x, y, self.scale, self.scale)
            pygame.draw.rect(self.screen, particle.color, rect)

        pygame.display.flip()
        self.clock.tick(60)

    def handle_events(self):
        """
        Handle Pygame events such as quitting the application.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.simulation.is_running = False
                return False
        return True

    def run(self):
        """
        Main loop of the GUI manager.
        """
        self.initialize()
        self.simulation.start()

        running = True
        while running and self.simulation.is_running:
            running = self.handle_events()
            self.simulation.update()
            self.render()

        pygame.quit()
