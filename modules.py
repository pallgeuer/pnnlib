# PNN Library: Model modules

# Imports
import warnings
import torch
import torch.nn as nn
from typing import Optional

#
# Data normalisation module
#

class DataNorm(nn.Module):

	DEFAULT_MOMENTUM_SAMPLES = 10000
	DEFAULT_EPS = 1e-5

	__constants__ = ['expected_input_dim', 'num_channels', 'momentum_samples', 'inplace', 'eps', '_reduce_dims']

	def __init__(self, expected_input_dim, num_channels, momentum_samples=DEFAULT_MOMENTUM_SAMPLES, inplace=False, eps=DEFAULT_EPS):
		super().__init__()

		self.expected_input_dim = expected_input_dim
		self.num_channels = num_channels
		self.momentum_samples = momentum_samples
		self.inplace = inplace
		self.eps = eps
		self._reduce_dims = tuple(index for index in range(self.expected_input_dim) if index != 1)

		broadcast_dims = tuple(self.num_channels if index == 1 else 1 for index in range(self.expected_input_dim))
		self.register_buffer('running_count', torch.tensor(0, dtype=torch.long))
		self.register_buffer('running_mean', torch.zeros(broadcast_dims))
		self.register_buffer('running_var', torch.ones(broadcast_dims))

	def reset_running_stats(self):
		self.running_count.zero_()
		self.running_mean.zero_()
		self.running_var.fill_(1)

	def extra_repr(self):
		return f"{self.num_channels}, momentum_samples={self.momentum_samples}, eps={self.eps}"

	def forward(self, batch_in):
		if batch_in.dim() != self.expected_input_dim:
			raise ValueError(f"Expected {self.expected_input_dim}D input (got {batch_in.dim()}D input)")

		if self.training:
			with torch.inference_mode():
				sample_count = batch_in.shape[0]
				old_count = min(self.running_count.item(), self.momentum_samples)
				new_count = old_count + sample_count
				ratio = sample_count / new_count
				self.running_count.fill_(min(new_count, self.momentum_samples))

				channel_var_tmp, channel_mean_tmp = torch.var_mean(batch_in, self._reduce_dims, keepdim=True, unbiased=False)
				delta_mean = channel_mean_tmp.sub_(self.running_mean)
				delta_var = channel_var_tmp.sub_(self.running_var)
				self.running_mean.add_(delta_mean, alpha=ratio)
				self.running_var.add_(delta_var, alpha=ratio).addcmul_(delta_mean, delta_mean, value=ratio * (1 - ratio))

		batch_out = batch_in.sub_(self.running_mean) if self.inplace else batch_in.sub(self.running_mean)
		batch_out.div_(self.running_var.add(self.eps).sqrt_())

		return batch_out

class DataNorm1d(DataNorm):

	def __init__(self, num_channels, momentum_samples=DataNorm.DEFAULT_MOMENTUM_SAMPLES, inplace=False, eps=DataNorm.DEFAULT_EPS):
		super().__init__(3, num_channels, momentum_samples=momentum_samples, inplace=inplace, eps=eps)

class DataNorm2d(DataNorm):

	def __init__(self, num_channels, momentum_samples=DataNorm.DEFAULT_MOMENTUM_SAMPLES, inplace=False, eps=DataNorm.DEFAULT_EPS):
		super().__init__(4, num_channels, momentum_samples=momentum_samples, inplace=inplace, eps=eps)

class DataNorm3d(DataNorm):

	def __init__(self, num_channels, momentum_samples=DataNorm.DEFAULT_MOMENTUM_SAMPLES, inplace=False, eps=DataNorm.DEFAULT_EPS):
		super().__init__(5, num_channels, momentum_samples=momentum_samples, inplace=inplace, eps=eps)

#
# Activation functions
#

class ScaledSigmoid(nn.Module):

	__constants__ = ['delta']
	delta: float

	def __init__(self, delta: float = 0.14):
		super().__init__()
		self.delta = delta

	def extra_repr(self):
		return f"delta={self.delta}"

	def forward(self, batch_in):
		return torch.sigmoid(batch_in).mul(1 + 2 * self.delta).sub_(self.delta)

#
# Dual log softmax
#

# noinspection PyMethodOverriding, PyAbstractClass
class LogSoftmaxFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, inp, dim):
		ctx.set_materialize_grads(False)
		ctx.dim = dim
		max_inp = inp.amax(dim=dim, keepdim=True)
		stable_inp = inp.sub(max_inp)
		exp = stable_inp.exp()
		sumexp = exp.sum(dim=dim, keepdim=True)
		ctx.save_for_backward(exp, sumexp)
		return stable_inp.sub(sumexp.log())

	@staticmethod
	@torch.autograd.function.once_differentiable
	def backward(ctx, grad_logsoft):
		if grad_logsoft is None or not ctx.needs_input_grad[0]:
			return None, None
		exp, sumexp = ctx.saved_tensors
		return grad_logsoft.sub(grad_logsoft.sum(dim=ctx.dim, keepdim=True).div_(sumexp).mul(exp)), None

# noinspection PyMethodOverriding, PyAbstractClass
class LogCompSoftmaxFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, inp, dim):
		ctx.set_materialize_grads(False)
		ctx.dim = dim
		max_inp = inp.amax(dim=dim, keepdim=True)
		stable_inp = inp.sub(max_inp)
		temp_exp = stable_inp.exp()
		neg_softmax = temp_exp.div_(temp_exp.sum(dim=dim, keepdim=True).neg_())
		ctx.save_for_backward(neg_softmax)
		return neg_softmax.log1p()

	@staticmethod
	@torch.autograd.function.once_differentiable
	def backward(ctx, grad_logcompsoft):
		if grad_logcompsoft is None or not ctx.needs_input_grad[0]:
			return None, None
		neg_softmax, = ctx.saved_tensors
		grad_scaled = grad_logcompsoft.mul(neg_softmax).div_(neg_softmax.add(1))
		return grad_scaled.add_(grad_scaled.sum(dim=ctx.dim, keepdim=True).mul(neg_softmax)), None

# noinspection PyMethodOverriding, PyAbstractClass
class DualLogSoftmaxFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, inp, dim):
		ctx.set_materialize_grads(False)
		ctx.dim = dim
		max_inp = inp.amax(dim=dim, keepdim=True)
		stable_inp = inp.sub(max_inp)
		temp_exp = stable_inp.exp()
		temp_sumexp = temp_exp.sum(dim=dim, keepdim=True)
		logsumexp = temp_sumexp.log()
		neg_softmax = temp_exp.div_(temp_sumexp.neg_())
		ctx.save_for_backward(neg_softmax)
		return stable_inp.sub(logsumexp), neg_softmax.log1p()

	@staticmethod
	@torch.autograd.function.once_differentiable
	def backward(ctx, grad_logsoft, grad_logcompsoft):
		if not ctx.needs_input_grad[0]:
			return None, None
		neg_softmax, = ctx.saved_tensors
		if grad_logsoft is not None:
			grad_inp_logsoft = grad_logsoft.add(grad_logsoft.sum(dim=ctx.dim, keepdim=True).mul(neg_softmax))
		else:
			grad_inp_logsoft = None
		if grad_logcompsoft is not None:
			grad_scaled = grad_logcompsoft.mul(neg_softmax).div_(neg_softmax.add(1))
			grad_inp_logcompsoft = grad_scaled.add_(grad_scaled.sum(dim=ctx.dim, keepdim=True).mul(neg_softmax))
		else:
			grad_inp_logcompsoft = None
		if grad_inp_logsoft is not None and grad_inp_logcompsoft is not None:
			return grad_inp_logsoft + grad_inp_logcompsoft, None
		elif grad_inp_logsoft is not None:
			return grad_inp_logsoft, None
		elif grad_inp_logcompsoft is not None:
			return grad_inp_logcompsoft, None
		else:
			return None, None

def log_softmax(inp: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
	if dim is None:
		dim = _get_softmax_dim("log_softmax", inp.ndim())
	return LogSoftmaxFunction.apply(inp, dim)

def log_comp_softmax(inp: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
	if dim is None:
		dim = _get_softmax_dim("log_comp_softmax", inp.ndim())
	return LogCompSoftmaxFunction.apply(inp, dim)

def dual_log_softmax(inp: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
	if dim is None:
		dim = _get_softmax_dim("dual_log_softmax", inp.ndim())
	return DualLogSoftmaxFunction.apply(inp, dim)

def _get_softmax_dim(name: str, ndim: int) -> int:
	warnings.warn(f"Implicit dimension choice for {name} has been deprecated. Change the call to include dim=X as an argument.")
	return 0 if ndim == 0 or ndim == 1 or ndim == 3 else 1

class LogSoftmaxBase(nn.Module):

	__constants__ = ['dim']
	dim: Optional[int]

	def __init__(self, dim: Optional[int] = None) -> None:
		super().__init__()
		self.dim = dim

	def extra_repr(self):
		return f"dim={self.dim}"

class LogSoftmax(LogSoftmaxBase):

	def forward(self, inp: torch.Tensor) -> torch.Tensor:
		return log_softmax(inp, self.dim)

class LogCompSoftmax(LogSoftmaxBase):

	def forward(self, inp: torch.Tensor) -> torch.Tensor:
		return log_comp_softmax(inp, self.dim)

class DualLogSoftmax(LogSoftmaxBase):

	def forward(self, inp: torch.Tensor) -> torch.Tensor:
		return dual_log_softmax(inp, self.dim)
# EOF
