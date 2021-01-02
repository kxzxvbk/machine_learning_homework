import numpy as np
EPSILON = 1e-7


class Loss:
    def __init__(self):
        pass

    def get_loss(self, x, y):
        pass

    def get_gradient(self, x, y):
        pass


class MeanSquareLoss(Loss):
    def __init__(self):
        super().__init__()

    def get_loss(self, x, y):
        return ((y - x) ** 2).sum()

    def get_gradient(self, x, y):
        return 2 * (y - x)


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def get_loss(self, x, y):
        return -(y * np.log(x + EPSILON) + (1 - y) * np.log(1 - x + EPSILON))

    def get_gradient(self, x, y):
        return -1 / (-y / (x + EPSILON) + (1 - y) / (1 - x + EPSILON))
