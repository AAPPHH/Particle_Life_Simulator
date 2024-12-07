class Quadtree:
    def __init__(self, x_min, y_min, x_max, y_max, capacity=100):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.capacity = capacity
        self.particles = []
        self.subtrees = None

    def insert(self, particle):
        x, y = particle[:2]
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return False

        if len(self.particles) < self.capacity and self.subtrees is None:
            self.particles.append(particle)
            return True

        if self.subtrees is None:
            self.subdivide()

        for subtree in self.subtrees:
            if subtree.insert(particle):
                return True

        return False

    def subdivide(self):
        mid_x = (self.x_min + self.x_max) / 2
        mid_y = (self.y_min + self.y_max) / 2
        self.subtrees = [
            Quadtree(self.x_min, self.y_min, mid_x, mid_y, self.capacity),
            Quadtree(mid_x, self.y_min, self.x_max, mid_y, self.capacity),
            Quadtree(self.x_min, mid_y, mid_x, self.y_max, self.capacity),
            Quadtree(mid_x, mid_y, self.x_max, self.y_max, self.capacity)
        ]

    def query(self, x_min, y_min, x_max, y_max):
        results = []
        if not self.intersects(x_min, y_min, x_max, y_max):
            return results

        for particle in self.particles:
            x, y = particle[:2]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                results.append(particle)

        if self.subtrees is not None:
            for subtree in self.subtrees:
                results.extend(subtree.query(x_min, y_min, x_max, y_max))

        return results

    def intersects(self, x_min, y_min, x_max, y_max):
        return not (x_max < self.x_min or x_min > self.x_max or
                    y_max < self.y_min or y_min > self.y_max)
