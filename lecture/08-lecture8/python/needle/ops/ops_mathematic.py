"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar-1)

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad  / b,  -out_grad / b**2 * a 


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, -2, -1)
        else:
            return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)
            

def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        origin_shape = node.inputs[0].shape
        return reshape(out_grad, origin_shape)

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape
        print(f'\ninput shape: {input_shape}')
        print(f'output shape: {output_shape}')

        # 先生成一个和output_shape长度一样的zero list，然后把input shape放在最后
        input_shape_adjusted = [0] * len(output_shape)
        if len(input_shape) > 0:
            input_shape_adjusted[-len(input_shape):] = input_shape[:]
        print(f'adjusted input shape: {input_shape_adjusted}')
        # 找到 input_shape_adjusted 和 output_shape不同的位置，这些位置就是需要reduce的维度
        axes_to_reduce = [
                            axis for axis, (in_dim, out_dim) 
                            in enumerate(zip(input_shape_adjusted, output_shape)) 
                            if in_dim != out_dim
                        ]
        print(f'axes to reduce: {axes_to_reduce}')
        grad = summation(out_grad, tuple(axes_to_reduce))
        print(f'grad shape: {grad.shape}')
        if len(input_shape) > 0:  # 这里是可能存在shape不对应的情况(5,4) -> (5,4,1)
            grad = reshape(grad, input_shape)
        return grad



def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        reduce_shape = list(node.inputs[0].shape)
        # print(f'\nreduce_shape 1: {reduce_shape}')
        if self.axes is not None:
            if not isinstance(self.axes, tuple):
                self.axes = tuple(self.axes)
            for axis in self.axes:
                reduce_shape[axis] = 1
            # print(f'reduce_shape 2: {reduce_shape}')
            # print(f'out_grad shape:{out_grad.shape}')
            grad = reshape(out_grad, reduce_shape)
            # print(f'out_grad reshape:{out_grad.shape}')
        else:
            grad = out_grad
        return broadcast_to(grad, node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs # a (B, m, k) b (k, n) -> output (B, m, n)
        grad_a = matmul(out_grad, transpose(b)) # grad_a: (B,m,n)@(n,k) -> (B,m,k)
        grad_b = matmul(transpose(a), out_grad) # grad_b: (B,k,m)@(B,m,n) ->(B,k,n)

        # 如果需要的话 调整shape（如果是BMM形式）
        if grad_a.shape != a.shape:
            grad_a = summation(grad_a, tuple(range(len(grad_a.shape) - len(a.shape))))
        if grad_b.shape != b.shape:
            grad_b = summation(grad_b, tuple(range(len(grad_b.shape) - len(b.shape))))

        return grad_a, grad_b

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -1 * a

    def gradient(self, out_grad, node):
        return -1 * out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return divide(out_grad, node.inputs[0])

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        res = a.copy()
        res[res < 0] = 0
        return res

    def gradient(self, out_grad, node):
        out_grad_data = out_grad.realize_cached_data()
        input_data = node.inputs[0].realize_cached_data()
        out_grad_data[input_data < 0] = 0
        return Tensor(out_grad_data)



def relu(a):
    return ReLU()(a)
