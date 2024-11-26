class Simulation:
    def __init__(self, width, height, scale, time_step=0.1):
        """
        Initialize the simulation with dimensions and time step.
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.particles = []  # List of particles in the simulation
        self.time_step = time_step  
        self.is_running = False 
        self.frame_count = 0

    def add_particle(self, particle):
        """
        Add a particle to the simulation.
        """
        self.particles.append(particle)

    def start(self):
        """
        Start the simulation.
        """
        self.is_running = True

    def pause(self):
        """
        Pause the simulation.
        """
        self.is_running = False

    def reset(self):
        """
        Reset the simulation to its initial state.
        """
        self.particles = []  # Clear the list of particles
        self.frame_count = 0  # Reset frame count
        self.is_running = False

    def update(self):
        """
        Update the simulation by one time step.
        """
        if not self.is_running:
            return

        for particle in self.particles:
            # Update particle position
            particle.update_position(self.time_step)
            # Apply friction
            particle.apply_friction()
            # Add random movement
            particle.randomize_movement()

            # Boundary checks
            max_x = (self.width // self.scale) - 1
            max_y = (self.height // self.scale) - 1

            if particle.position[0] <= 0 or particle.position[0] >= max_x:
                particle.velocity[0] *= -1  # Reverse X velocity
                particle.position[0] = max(min(particle.position[0], max_x), 0)
            if particle.position[1] <= 0 or particle.position[1] >= max_y:
                particle.velocity[1] *= -1  # Reverse Y velocity
                particle.position[1] = max(min(particle.position[1], max_y), 0)

        self.frame_count += 1
        
