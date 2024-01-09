"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(self.weight)
        if bias:
            bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = ops.reshape(bias, (1, out_features))
            self.bias = Parameter(self.bias)
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        out = X @ self.weight
        if self.bias is not None:
            bias = self.bias.broadcast_to(out.shape)
            out = out + bias
        return out


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        return ops.reshape(X, (batch_size, -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        log_sum = ops.logsumexp(logits, axes=1)
        y_one_hot = init.one_hot(logits.shape[1], y)
        z_y = ops.summation(logits * y_one_hot, axes=1)
        return ops.summation(log_sum - z_y) / logits.shape[0]
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim))
        self.bias = Parameter(init.zeros(1, self.dim))
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)
        ### END YOUR SOLUTION


    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            X_mean = ops.summation(X, axes=(0,)) / X.shape[0]
            X_sub = X - ops.broadcast_to(ops.reshape(X_mean, (1, X.shape[1])), X.shape)
            X_sub_sq = ops.power_scalar(X_sub, 2)
            X_sub_sq_sum = ops.summation(X_sub_sq, axes=(0,))
            X_var = X_sub_sq_sum / X.shape[0]
            X_var_eps = X_var + self.eps
            X_sigma = ops.power_scalar(X_var_eps, 1/2)
            X_norm = X_sub / ops.broadcast_to(ops.reshape(X_sigma, (1, X.shape[1])), X.shape)
            result = X_norm * ops.broadcast_to(self.weight, X.shape)
            result += ops.broadcast_to(self.bias, result.shape)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * X_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * X_var.detach()
        else:
            X_mean = self.running_mean.detach()
            X_var = self.running_var.detach()
            X_sub = X - ops.broadcast_to(ops.reshape(X_mean, (1, X.shape[1])), X.shape)
            X_var_eps = X_var + self.eps
            X_sigma = ops.power_scalar(X_var_eps, 1/2)
            X_norm = X_sub / ops.broadcast_to(ops.reshape(X_sigma, (1, X.shape[1])), X.shape)
            result = X_norm * ops.broadcast_to(self.weight, X.shape)
            result += ops.broadcast_to(self.bias, result.shape)
        return result
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(1, dim, device=device, dtype=dtype)
        self.bias = init.zeros(1, dim, device=device, dtype=dtype)
        self.weight = Parameter(self.weight)
        self.bias = Parameter(self.bias)

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_mean = ops.summation(X, axes=(1,)) / X.shape[1]
        X_mean = ops.reshape(X_mean, (X.shape[0], 1))
        X_sub = X - ops.broadcast_to(X_mean, X.shape)
        X_sub_sq = ops.power_scalar(X_sub, 2)
        X_sub_sq_sum = ops.summation(X_sub_sq, axes=(1,))
        X_var = X_sub_sq_sum / X.shape[1]
        X_var_eps = X_var + self.eps
        X_sigma = ops.power_scalar(X_var_eps, 1/2)
        X_sigma = ops.reshape(X_sigma, (X.shape[0], 1))
        X_norm = X_sub / ops.broadcast_to(X_sigma, X.shape)
        result = X_norm * ops.broadcast_to(self.weight, X_norm.shape)
        if self.bias:
            result += ops.broadcast_to(self.bias, result.shape)
        return result
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            B = init.randb(*x.shape, p=(1-self.p))
            x_norm = x / (1 - self.p)
            result = x_norm * B
        else:
            result = x
        return result
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
