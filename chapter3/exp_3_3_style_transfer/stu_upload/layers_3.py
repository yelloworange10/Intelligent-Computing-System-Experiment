import numpy as np
import struct
import os
import scipy.io
import time

class ContentLossLayer(object):
    def __init__(self):
        print('\tContent loss layer.')
    def forward(self, input_layer, content_layer):
        # TODO： 计算风格迁移图像和目标内容图像的内容损失
        self.diff = input_layer - content_layer
        loss = 0.5 * np.sum(self.diff ** 2)
        return loss
    def backward(self, input_layer, content_layer):
        # TODO： 计算内容损失的反向传播
        bottom_diff = self.diff
        return bottom_diff

class StyleLossLayer(object):
    def __init__(self):
        print('\tStyle loss layer.')
    def forward(self, input_layer, style_layer):
        # TODO： 计算风格迁移图像和目标风格图像的Gram 矩阵
        style_layer_reshape = np.reshape(style_layer, [style_layer.shape[0], style_layer.shape[1], -1])
        self.gram_style = np.matmul(style_layer_reshape, style_layer_reshape.transpose((0, 2, 1)))
        self.input_layer_reshape = np.reshape(input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        self.gram_input = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[1]])
        for idxn in range(input_layer.shape[0]):
            self.gram_input[idxn, :, :] = np.matmul(self.input_layer_reshape[idxn], self.input_layer_reshape[idxn].T)
        M = input_layer.shape[2] * input_layer.shape[3]
        N = input_layer.shape[1]
        self.div = M * M * N * N
        # TODO： 计算风格迁移图像和目标风格图像的风格损失
        style_diff = self.gram_input - self.gram_style
        loss = np.sum(self.style_diff ** 2) / self.div
        return loss
    def backward(self, input_layer, style_layer):
        bottom_diff = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[2]*input_layer.shape[3]])
        for idxn in range(input_layer.shape[0]):
            # TODO： 计算风格损失的反向传播
            M = input_layer.shape[2] * input_layer.shape[3]
            N = input_layer.shape[1]
            factor = (1.0 / (N * N * M * M))
            bottom_diff[idxn, :, :] = factor * np.matmul(self.style_diff[idxn], self.input_layer_reshape[idxn]).reshape((N, M * M))
        bottom_diff = np.reshape(bottom_diff, input_layer.shape)
        return bottom_diff
