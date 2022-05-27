#!/usr/bin/env python3
# Test pnnlib.modules

# Imports
import sys
import timeit
import torch
import torch.nn.functional as F
from pnnlib import modules

#
# Test dual log softmax
#

# Test numerical consistency of the dual log softmax implementations
def test_dual_log_softmax():
	inp = torch.normal(0.0, 3.0, (3, 4, 5, 6), requires_grad=True)
	dim = torch.randint(4, ()).item()

	inp.grad = None
	logsofta = modules.LogSoftmaxFunction.apply(inp, dim)
	logsofta[0, 1, 2, 3].backward()
	logsoftagrad = inp.grad

	inp.grad = None
	logcompsofta = modules.LogCompSoftmaxFunction.apply(inp, dim)
	logcompsofta[0, 1, 2, 3].backward()
	logcompsoftagrad = inp.grad

	duallogsoftagrad = logsoftagrad + logcompsoftagrad

	inp.grad = None
	logsoftb, logcompsoftb = modules.DualLogSoftmaxFunction.apply(inp, dim)
	(logsoftb[0, 1, 2, 3] + logcompsoftb[0, 1, 2, 3]).backward()
	duallogsoftbgrad = inp.grad

	inp.grad = None
	logsoftc = F.log_softmax(inp, dim=dim)
	logsoftc[0, 1, 2, 3].backward()
	logsoftcgrad = inp.grad

	if not __debug__:
		print("Please enable debug mode otherwise the assertions will not be executed")

	assert torch.equal(modules.log_softmax(inp, dim=dim), logsofta)
	assert torch.equal(modules.log_comp_softmax(inp, dim=dim), logcompsofta)
	templogsoft, templogcompsoft = modules.dual_log_softmax(inp, dim=dim)
	assert torch.equal(templogsoft, logsoftb)
	assert torch.equal(templogcompsoft, logcompsoftb)

	assert torch.equal(modules.LogSoftmax(dim=dim)(inp), logsofta)
	assert torch.equal(modules.LogCompSoftmax(dim=dim)(inp), logcompsofta)
	templogsoft, templogcompsoft = modules.DualLogSoftmax(dim=dim)(inp)
	assert torch.equal(templogsoft, logsoftb)
	assert torch.equal(templogcompsoft, logcompsoftb)

	print("Showing max differences:")
	print(torch.abs(logsofta - logsoftb).max())
	print(torch.abs(logcompsofta - logcompsoftb).max())
	print(torch.abs(logsofta - logsoftc).max())
	print(torch.abs(logsoftb - logsoftc).max())
	print(torch.abs(logsoftagrad - logsoftcgrad).max())
	print(torch.abs(duallogsoftagrad - duallogsoftbgrad).max())
	print()

	assert torch.allclose(logsofta, logsoftb)
	assert torch.allclose(logcompsofta, logcompsoftb)
	assert torch.allclose(logsofta, logsoftc, atol=1e-6)
	assert torch.allclose(logsoftb, logsoftc, atol=1e-6)
	assert torch.allclose(logsoftagrad, logsoftcgrad, atol=1e-7)
	assert torch.allclose(duallogsoftagrad, duallogsoftbgrad)

	print("Showing sum differences:")
	print((logsofta.exp() + logcompsofta.exp() - 1).abs().max())
	print((logsoftb.exp() + logcompsoftb.exp() - 1).abs().max())
	print((logsoftc.exp() + logcompsofta.exp() - 1).abs().max())
	print()

	assert torch.allclose(logsofta.exp() + logcompsofta.exp(), torch.ones_like(logsofta))
	assert torch.allclose(logsoftb.exp() + logcompsoftb.exp(), torch.ones_like(logsoftb))
	assert torch.allclose(logsoftc.exp() + logcompsofta.exp(), torch.ones_like(logsoftc))

	assert torch.autograd.gradcheck(lambda x: F.log_softmax(x, dim=dim), torch.normal(0.0, 3.0, (3, 4, 5, 6), requires_grad=True, dtype=torch.double), check_grad_dtypes=True, check_batched_grad=True)
	assert torch.autograd.gradcheck(lambda x: modules.LogSoftmaxFunction.apply(x, dim), torch.normal(0.0, 3.0, (3, 4, 5, 6), requires_grad=True, dtype=torch.double), check_grad_dtypes=True, check_batched_grad=True)
	assert torch.autograd.gradcheck(lambda x: modules.LogCompSoftmaxFunction.apply(x, dim), torch.normal(0.0, 3.0, (3, 4, 5, 6), requires_grad=True, dtype=torch.double), check_grad_dtypes=True, check_batched_grad=True)
	assert torch.autograd.gradcheck(lambda x: modules.DualLogSoftmaxFunction.apply(x, dim), torch.normal(0.0, 3.0, (3, 4, 5, 6), requires_grad=True, dtype=torch.double), check_grad_dtypes=True, check_batched_grad=True)

# Test speed of the dual log softmax implementations
def test_dual_log_softmax_speed():
	inp = torch.normal(0.0, 3.0, (3, 4, 5, 6), requires_grad=True, device='cuda')
	dim = torch.randint(4, ()).item()

	F.log_softmax(inp, dim=dim)
	modules.log_softmax(inp, dim=dim)
	modules.log_comp_softmax(inp, dim=dim)
	modules.dual_log_softmax(inp, dim=dim)

	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	t_start = timeit.default_timer()
	start.record()

	for _ in range(100000):
		# F.log_softmax(inp, dim=dim)
		# modules.log_softmax(inp, dim=dim)
		modules.log_comp_softmax(inp, dim=dim)
		# modules.dual_log_softmax(inp, dim=dim)

	end.record()
	torch.cuda.synchronize()
	t_stop = timeit.default_timer()
	print(f"Duration: {start.elapsed_time(end):.0f}ms or {1000 * (t_stop - t_start):.0f}ms")

#
# Main
#

# Main function
def main():
	test_dual_log_softmax()
	test_dual_log_softmax_speed()

# Run main function
if __name__ == "__main__":
	sys.exit(main())
# EOF
