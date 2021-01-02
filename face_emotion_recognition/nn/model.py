import numpy as np
import math
from functools import reduce
import nn.function as function
LEARNING_RATE = 0.01


class Model:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, gradient):
        pass

    def zero_grad(self):
        pass

    def apply_gradient(self):
        pass

    def summary(self):
        pass


class Sequential(Model):
    def __init__(self):
        super().__init__()
        self.layers = []

    def add(self, layer):
        assert isinstance(layer, (function.Function, Model))
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient):
        index = len(self.layers) - 1
        temp_gradient = gradient
        while index >= 0:
            if isinstance(self.layers[index], Model):
                self.layers[index].backward(temp_gradient)
                temp_gradient = np.dot(self.layers[index].gradient_factor, temp_gradient)
            elif isinstance(self.layers[index], function.Function):
                temp_gradient = self.layers[index].gradient_factor() * temp_gradient
            index = index - 1

    def apply_gradient(self):
        index = len(self.layers) - 1
        while index >= 0:
            if isinstance(self.layers[index], Model):
                self.layers[index].apply_gradient()
            index = index - 1

    def zero_grad(self):
        index = len(self.layers) - 1
        while index >= 0:
            if isinstance(self.layers[index], Model):
                self.layers[index].zero_grad()
            index = index - 1

    def step(self, gradient):
        self.backward(gradient)
        # self.apply_gradient()
        # self.zero_grad()

    def summary(self):
        ans = 'Sequential: \n'
        for layer in self.layers:
            ans += layer.summary()
        return ans


class LinearLayer(Model):
    def __init__(self, input_size, output_size, has_bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = has_bias

        self.w = np.random.rand(output_size, input_size)
        self.b = np.zeros([output_size, 1])
        self.input = np.zeros([input_size])
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)
        self.gradient_factor = self.w.transpose()

    # x -> 1-dim vector
    def forward(self, x):
        x = x.reshape([x.shape[0], 1])
        self.input = x
        assert x.shape[0] == self.input_size
        temp = np.dot(self.w, x) + self.b
        temp = temp.squeeze()
        return temp

    # gradient -> 1-dim vector
    def backward(self, gradient):
        self.dw = gradient.reshape([gradient.shape[0], 1]) * self.input.transpose()
        self.db = gradient.reshape([gradient.shape[0], 1])
        self.apply_gradient()
        self.zero_grad()

    def apply_gradient(self):
        global LEARNING_RATE
        self.w += LEARNING_RATE * self.dw
        self.b += LEARNING_RATE * self.db
        self.gradient_factor = self.w.transpose()

    def zero_grad(self):
        self.dw, self.db = 0, 0

    def summary(self):
        return 'LinearLayer({}, {})\n'.format(self.input_size, self.output_size)


class Conv2D(Model):
    # reference:https://zhuanlan.zhihu.com/p/102119808
    def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
        # parameters
        super().__init__()
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.method = method

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        if method == 'VALID':
            self.eta = np.zeros((shape[0], (shape[1] - ksize + 1) // self.stride, (shape[1] - ksize + 1) // self.stride,
                                 self.output_channels))

        if method == 'SAME':
            self.eta = np.zeros((shape[0], shape[1] / self.stride, shape[2] / self.stride, self.output_channels))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        if (shape[1] - ksize) % stride != 0:
            print('input tensor width can\'t fit stride')
        if (shape[2] - ksize) % stride != 0:
            print('input tensor height can\'t fit stride')

    def zero_grad(self):
        self.pre_grad = np.zeros(self.input.shape)
        self.w_gradient = np.zeros(self.w_gradient.shape)
        self.b_gradient = np.zeros(self.b_gradient.shape)

    def forward(self, x):
        # 一旦进入正传，必须先zero_grad
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                       'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = self.im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        print(self.col_image.shape)
        return conv_out

    def backward(self, eta):
        # post_grad只上一层的回传梯度
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        print(self.col_image)
        print(col_eta)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array(
            [self.im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def update(self):
        self.weights -= LEARNING_RATE * self.weights_grad
        self.bias -= LEARNING_RATE * self.bias_grad

    def apply_gradient(self):
        self.update()

    def im2col(self, image, ksize, stride):
        # flatten

        N, C, H, W = image.shape
        image_col = []

        # i，j枚举卷积核可选的左上角起点
        for i in range(0, H - ksize + 1, stride):
            for j in range(0, W - ksize + 1, stride):
                # 展平成每一列append进去
                col = image[:, :, i:i + ksize, j:j + ksize].reshape(N, -1)
                image_col.append(col)

        image_col = np.array(image_col).transpose((1, 0, 2)).reshape(-1, C * ksize * ksize)
        return image_col
