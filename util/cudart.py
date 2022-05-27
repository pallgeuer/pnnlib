# PNN Library: CUDA runtime API utilities

# Imports
import torch

# Note: At some point between versions 1.4.1 and 1.9.0, PyTorch stopped using Python ctypes to load the CUDA Runtime Library, and instead just
#       exposed the few functions that are required in Python manually from the C sources. This means that it is no longer possible to manually
#       call arbitrary CUDA Runtime functions via ctypes, hence why functions are disabled in this file. You should use PyCUDA instead.

# Call a CUDA runtime API C function
# Note: This initialises the entire CUDA runtime if it is not already initialised, resulting in large allocations of GPU memory on the default device!
#       This is okay when you know the CUDA runtime was/will be initialised anyway, but if not, using the PyCUDA device interface can avoid these allocations.
def call_cudart_func_DISABLED(func, *func_args, cudaerror=True):
	# func = String function name (e.g. 'cudaDriverGetVersion'), or direct CUDA runtime function pointer (e.g. torch.cuda.cudart().cudaDriverGetVersion)
	# func_args = Arguments to pass to func (using ctypes)
	# cudaerror = Whether to treat the return value as a CUDA error (cudaError_t), automatically clear the CUDA error state if appropriate, and return None or Tuple(error_name, error_desc) instead

	cudart = None

	if not callable(func):
		if not cudart:
			cudart = torch.cuda.cudart()
		func = getattr(cudart, func)

	ret = func(*func_args)

	if cudaerror:
		if ret != 0:
			if not cudart:
				cudart = torch.cuda.cudart()
			cuda_error = cudart.cudaGetLastError()  # Note: It is very important that this clears the CUDA error state!
			cuda_error_name = cudart.cudaGetErrorName(cuda_error).decode('utf-8')
			cuda_error_desc = cudart.cudaGetErrorString(cuda_error).decode('utf-8')
			return cuda_error_name, cuda_error_desc
		else:
			return None

	return ret
# EOF
