import numpy as np
from nn.model import Sequential, LinearLayer
import nn.function as function
import nn.loss as loss
EPOCHS = 1000


if __name__ == '__main__':
    X = np.random.rand(8)
    y = np.array([1, 100])

    seq = Sequential()
    seq.add(LinearLayer(8, 4))
    seq.add(function.Relu())
    seq.add(LinearLayer(4, 2))
    print(seq.summary())

    for i in range(EPOCHS):
        pred = seq.forward(X)
        print('PRED: ' + str(pred))
        los = loss.MeanSquareLoss()
        print('LOSS: ' + str(los.get_loss(pred, y)))
        grad = los.get_gradient(pred, y)
        seq.step(grad)
