# quadtree.pyx
# Example Quadtree implementation in Cython (Python 3)
# cython: language_level=3

cdef class Quadtree:
    cdef public double x_min, y_min, x_max, y_max
    cdef public int capacity
    cdef list particles
    cdef object subtrees  # Can be None or a list of Quadtrees

    def __init__(self, double x_min, double y_min, double x_max, double y_max, int capacity=10):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.capacity = capacity
        self.particles = []
        self.subtrees = None

    cpdef bint insert(self, object particle):
        cdef double x = particle[0]
        cdef double y = particle[1]

        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return False

        if len(self.particles) < self.capacity and self.subtrees is None:
            self.particles.append(particle)
            return True

        if self.subtrees is None:
            self.subdivide()

        cdef Quadtree subtree
        for subtree in self.subtrees:
            if subtree.insert(particle):
                return True

        return False

    cpdef void subdivide(self):
        cdef double mid_x = (self.x_min + self.x_max) / 2.0
        cdef double mid_y = (self.y_min + self.y_max) / 2.0

        # Erstelle die vier Unterbäume
        self.subtrees = [
            Quadtree(self.x_min, self.y_min, mid_x,      mid_y,      self.capacity),
            Quadtree(mid_x,      self.y_min, self.x_max, mid_y,      self.capacity),
            Quadtree(self.x_min, mid_y,      mid_x,      self.y_max, self.capacity),
            Quadtree(mid_x,      mid_y,      self.x_max, self.y_max, self.capacity)
        ]

        # Verschiebe vorhandene Partikel in die Unterbäume
        for particle in self.particles:
            for subtree in self.subtrees:
                if subtree.insert(particle):
                    break

        # Leere die Partikelliste der aktuellen Region
        self.particles = []

    cpdef list query(self, double x_min, double y_min, double x_max, double y_max):
        cdef list results = []
        if not self.intersects(x_min, y_min, x_max, y_max):
            return results

        cdef double x, y
        for particle in self.particles:
            x = particle[0]
            y = particle[1]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                results.append(particle)

        cdef Quadtree subtree
        if self.subtrees is not None:
            for subtree in self.subtrees:
                results.extend(subtree.query(x_min, y_min, x_max, y_max))

        return results

    cpdef bint intersects(self, double x_min, double y_min, double x_max, double y_max):
        return not (
            x_max < self.x_min or
            x_min > self.x_max or
            y_max < self.y_min or
            y_min > self.y_max
        )
