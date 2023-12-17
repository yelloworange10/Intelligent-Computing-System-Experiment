from turtle import forward
import torch
import torch.nn as nn
import torch.jit as jit

from typing import Any

# TODO: 请补充自定义算子库的名称
from mlu_custom_ext import *


class sigmoid_function(torch.autograd.Function):
    """
    sigmoid for autograd
    """

    @staticmethod
    def forward(ctx, x):
        # TODO: 请补充自定义算子的python接口函数名
        # 自定义算子的python接口函数名，根据之前的cpp文件中的pybind11模块推断
        y = active_sigmoid_mlu(x)
        ctx.save_for_backward(*[x, y])
        return y

    @staticmethod
    def backward(ctx: Any, d_r: Any) -> Any:
        d_r = d_r.contiguous()
        x, y = ctx.saved_tensors
        dx = y * (1 - y) * d_r
        return dx


@jit.ignore
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    sigmoid for forward
    """
    return sigmoid_function.apply(x)
