class Simulation:

    def __init__(self, time_step=0.1):
        """
        Initialize the list of particles, time step, and simulation state.
        """
        self.particles = []  # List of particles in the simulation
        self.time_step = time_step  
        self.is_running = False 
        self.frame_count = 0
    
    def start(self):
        """
         a method to start the simulation
        """
        self.is_running = True
    
    def pause(self):
        """
        a method to pause the simulation.
        """
        self.is_running = False
    
    def reset(self):
        """
        a method to reset the simulation to its initial state.
        """
        self.particles = []  # Clear the list of particles
        self.frame_count = 0  # Reset frame count
        self.is_running = False
        
