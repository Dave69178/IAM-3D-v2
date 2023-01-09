import numpy as np


class Boid:
    def __init__(self, index, position=np.array([0, 0, 0]), velocity=np.array([0, 0, 0])):
        self._index = index
        self._position = position
        self._velocity = velocity

    def __repr__(self):
        return f"Boid({self._index},{self._position},{self._velocity})"

    def __str__(self):
        return f"Index: {self._index}, Position: {self._position}, Velocity: {self._velocity}"

    @property
    def index(self):
        return self._index

    @index.getter
    def index(self):
        return self._index

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position

    @position.getter
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @velocity.getter
    def velocity(self):
        return self._velocity

    def get_distances_to_boids(self, env):
        return np.array([np.linalg.norm(i) for i in (env.get_boid_positions() - self.position)])
