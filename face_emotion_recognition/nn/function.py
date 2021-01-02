import numpy as np
EPSILON = 1e-7


class Function:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def gradient_factor(self):
        pass

    def summary(self):
        pass


class Sigmoid(Function):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return 1 / (1 + np.exp(-x + EPSILON))

    def gradient_factor(self):
        return ((1 + np.exp(-self.input + EPSILON)) ** 2) / (np.exp(-self.input + EPSILON) + EPSILON)

    def summary(self):
        return 'Sigmoid\n'


class Relu(Function):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return x * (x > 0)

    def gradient_factor(self):
        return self.input * (self.input > 0)

    def summary(self):
        return 'Relu\n'

