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
        # 计算两个特征图之间的平方差并取平均
        loss = np.square(input_layer - content_layer).sum() / (2 * input_layer.shape[0] * input_layer.shape[1] * input_layer.shape[2] * input_layer.shape[3])
        return loss
    def backward(self, input_layer, content_layer):
        # TODO： 计算内容损失的反向传播
        bottom_diff = (input_layer - content_layer) / (input_layer.shape[0] * input_layer.shape[1] * input_layer.shape[2] * input_layer.shape[3])

        return bottom_diff

class StyleLossLayer(object):
    def __init__(self):
        print('\tStyle loss layer.')
    def forward(self, input_layer, style_layer):
        # TODO： 计算风格迁移图像和目标风格图像的Gram 矩阵
        style_layer_reshape = np.reshape(style_layer, [style_layer.shape[0], style_layer.shape[1], -1])
        # 计算风格图像的Gram矩阵
        self.gram_style = np.matmul(style_layer_reshape, np.transpose(style_layer_reshape, [0, 2, 1]))

        self.input_layer_reshape = np.reshape(input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        self.gram_input = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[1]])
        for idxn in range(input_layer.shape[0]):
            self.gram_input[idxn, :, :] = np.matmul(self.input_layer_reshape[idxn], np.transpose(self.input_layer_reshape[idxn], [1, 0]))

        M = input_layer.shape[2] * input_layer.shape[3]
        N = input_layer.shape[1]
        self.div = M * M * N * N
        # TODO： 计算风格迁移图像和目标风格图像的风格损失
        style_diff = ((self.gram_input - self.gram_style)**2).sum(axis=(1, 2))  # ((self.gram_input - self.gram_style)**2).sum() / (input_layer.shape[0] * self.div * 4)
        loss = style_diff.sum() / (input_layer.shape[0] * self.div * 4)
        return loss
    def backward(self, input_layer, style_layer):
        # 计算风格图的Gram矩阵
        style_layer_reshape = np.reshape(style_layer, [style_layer.shape[0], style_layer.shape[1], -1])
        gram_style = np.matmul(style_layer_reshape, np.transpose(style_layer_reshape, [0, 2, 1]))

        # 计算输入图的Gram矩阵
        input_layer_reshape = np.reshape(input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        gram_input = np.matmul(input_layer_reshape, np.transpose(input_layer_reshape, [0, 2, 1]))

        # 计算每个元素的损失梯度
        M = input_layer.shape[2] * input_layer.shape[3]
        N = input_layer.shape[1]
        factor = 4.0 * (N ** 2) * (M ** 2)

        # 计算Gram矩阵之差
        gram_diff = gram_input - gram_style

        # 初始化梯度矩阵
        bottom_diff = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[2]*input_layer.shape[3]])
        for idxn in range(input_layer.shape[0]):
            # TODO： 计算风格损失的反向传播
            dG = gram_diff[idxn] / (N * M)
            bottom_diff[idxn, :, :] = (dG @ input_layer_reshape[idxn]).reshape(input_layer[idxn].shape)
        bottom_diff = np.reshape(bottom_diff, input_layer.shape)
        return bottom_diff
