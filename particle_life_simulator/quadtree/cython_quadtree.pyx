# quadtree.pyx
# Example Quadtree implementation in Cython (Python 3)
# cython: language_level=3

cdef class Quadtree:
    cdef public double x_min, y_min, x_max, y_max
    cdef public int capacity, depth, max_depth
    cdef list particles
    cdef object subtrees  # Can be None or a list of Quadtrees

    def __init__(self, double x_min, double y_min, double x_max, double y_max, int capacity=50, int depth=0, int max_depth=100):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.capacity = capacity
        self.depth = depth
        self.max_depth = max_depth
        self.particles = []
        self.subtrees = None

    cpdef bint insert(self, object particle):
        cdef double x = particle[0]
        cdef double y = particle[1]

        # Check if the particle lies within the current region
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return False

        # If capacity is not exceeded and there are no subtrees, add the particle
        if len(self.particles) < self.capacity and self.subtrees is None:
            self.particles.append(particle)
            return True

        # Subdivide only if the maximum depth is not reached
        if self.subtrees is None and self.depth < self.max_depth:
            self.subdivide()

        # Insert the particle into the appropriate subtree
        cdef Quadtree subtree
        if self.subtrees is not None:
            for subtree in self.subtrees:
                if subtree.insert(particle):
                    return True

        # If the maximum depth is reached, keep the particle in the current region
        self.particles.append(particle)
        return True

    cpdef void subdivide(self):
        if self.depth >= self.max_depth:
            return  # Further subdivision not allowed, maximum depth reached

        cdef double mid_x = (self.x_min + self.x_max) / 2.0
        cdef double mid_y = (self.y_min + self.y_max) / 2.0

        # Create the four subtrees with an increased depth of 1
        self.subtrees = [
            Quadtree(self.x_min, self.y_min, mid_x,      mid_y,      self.capacity, self.depth + 1, self.max_depth),
            Quadtree(mid_x,      self.y_min, self.x_max, mid_y,      self.capacity, self.depth + 1, self.max_depth),
            Quadtree(self.x_min, mid_y,      mid_x,      self.y_max, self.capacity, self.depth + 1, self.max_depth),
            Quadtree(mid_x,      mid_y,      self.x_max, self.y_max, self.capacity, self.depth + 1, self.max_depth)
        ]

        # Move existing particles into the subtrees
        for particle in self.particles:
            for subtree in self.subtrees:
                if subtree.insert(particle):
                    break

        # Clear the particle list of the current region
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
