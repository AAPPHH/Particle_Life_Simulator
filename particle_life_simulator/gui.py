
import dearpygui


class Gui_Manager():
    
    def __init__(self, window_width: 800, window_height: 600):
        #UI window
        self.window_width = window_width
        self.window_height = window_height
        self.particles = []
        self.running = False

    def render_particles(self):
        clear_drawing("ParticleCanvas")
        for particle in self.particles:
            position = particle.get("position", (0, 0))
            #color?
            #draw particle

    def user_input(self, sender, data):
        self.running = not self.running
        set_value("Play/Pause Button", "Pause ⏸" if self.running else "Play ▷")

    def setup_gui(self):
        with window("Particle Simulator", width=self.window_width, height=self.window_height):
            
            add_drawing("Particle Canvas", width=self.window_width, height=self.window_height - 50)
            add_button("Play/Pause Button", label="Play ▷", callback=self.handle_user_input)
            add_text("Status", default_value = "Pause")
            add_button("ClearParticles Button", label = "Clear Particles", callback = lambda s, d: self.clear_particles())
            #lambda to pass self.clear_particles() method as callback,  s = sender, d = data

    def add_particle(self, position, color):
        self.particles.append({"position": position, "color": color})
        self.render_particles()


    def clear_particles(self):
        self.particles.clear()
        self.render_particles()

    def start(self):
        self.setup_gui()
        start_dearpygui()


#▶ ▷ ⏯ ⏸ ⏸ ► ▶️
'''
from gui import GUI

if __name__ == "__main__":
    # Initialize the GUI class
    gui = GUI(window_width=800, window_height=600)
    
    # Start the GUI application
    gui.start()
'''










