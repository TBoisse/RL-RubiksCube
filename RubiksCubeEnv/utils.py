from numpy.random import randint

class ActionSpace():
    def __init__(self, n):
        self.n = n

    def sample(self, size = 1):
        return randint(self.n, size=size)
    
    def in_bound(self, action):
        return action >= 0 and action < self.n 

    def __repr__(self):
        return f"Discrete({self.n})"
    
class ObservationSpace():
    def __init__(self, low, high, length, obs_type):
        self.low = low
        self.high = high
        self.shape = (length,)
        self.obs_type = obs_type

    def __repr__(self):
        return f"Vector({self.low}, {self.high}, {self.shape}, {self.obs_type.__name__})"