import needle as ndl
from needle import backend_ndarray as nd

x = ndl.Tensor([1,2,3], device=nd.cuda(), dtype="float32")
y = ndl.Tensor([1,2,3], device=nd.cuda(), dtype="float32")
z = x / y
print(z)