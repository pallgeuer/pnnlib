# PNN Library: Torch tensor utilities

# Imports
import math
import numpy as np
import PIL.Image
import cv2
import torch
import torchvision.transforms.functional

# Torch <-> Numpy dtype mappings
torch_dtype_from_numpy_dict = {
	np.bool: torch.bool,
	np.uint8: torch.uint8,
	np.int8: torch.int8,
	np.int16: torch.int16,
	np.int32: torch.int32,
	np.int64: torch.int64,
	np.float16: torch.float16,
	np.float32: torch.float32,
	np.float64: torch.float64,
	np.complex64: torch.complex64,
	np.complex128: torch.complex128
}
numpy_dtype_from_torch_dict = {value: key for (key, value) in torch_dtype_from_numpy_dict.items()}

# Converter function from numpy.dtype to torch.dtype
def torch_dtype_from_numpy(numpy_dtype):
	# numpy_dtype = Numpy dtype
	# Return equivalent torch dtype
	return torch_dtype_from_numpy_dict[numpy_dtype]

# Converter function from torch.dtype to numpy.dtype
def numpy_dtype_from_torch(torch_dtype):
	# torch_dtype = Torch dtype
	# Return equivalent numpy dtype
	return numpy_dtype_from_torch_dict[torch_dtype]

# Converter function for torch.dtype
def parse_torch_dtype(string):
	# string = Torch dtype name
	# Return equivalent torch.dtype
	try:
		dtype = getattr(torch, string.lower())
	except AttributeError:
		dtype = None
	if not isinstance(dtype, torch.dtype):
		raise LookupError(f"Failed to convert string to torch.dtype: '{string}'")
	return dtype

# Stringify an object with some special handling for arrays and tensors
def brief_repr(obj, size_threshold=None):
	if isinstance(obj, tuple):
		return f"({', '.join(brief_repr(o) for o in obj)})"
	elif isinstance(obj, list):
		return f"[{', '.join(brief_repr(o) for o in obj)}]"
	if size_threshold is None:
		size_threshold = -1
	if (isinstance(obj, np.ndarray) and obj.size > size_threshold) or (isinstance(obj, torch.Tensor) and obj.numel() > size_threshold):
		return f"{type(obj).__name__}({' x '.join(str(d) for d in obj.shape)})"
	else:
		return str(obj)

# Unnormalise a tensor given the means and standard deviations
def unnormalise_tensor(tensor, mean, stddev, inplace=False):
	# tensor = Tensor to unnormalise (should be DxHxW or BxDxHxW)
	# mean = Iterable of mean values for each channel (length D)
	# stddev = Iterable of standard deviation values for each channel (length D)
	# inplace = Whether to unnormalise the tensor in-place or create a copy

	if not torch.is_tensor(tensor):
		raise TypeError(f"Input is not a torch tensor: {type(tensor)}")

	D = len(mean)
	if len(stddev) != D:
		raise ValueError(f"Supplied mean and standard deviation do not have the same number of elements: {len(mean)} vs {len(stddev)}")

	N = tensor.ndim
	if N < 3:
		raise ValueError(f"Tensor must have at least 3 dimensions: Has {N}")
	if tensor.shape[N - 3] != D:
		raise ValueError(f"Tensor does not have the expected depth: {tensor.shape[N - 3]} (should be {D})")

	if not inplace:
		tensor = tensor.clone()

	dtype = tensor.dtype
	mean_tensor = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
	std_tensor = torch.as_tensor(stddev, dtype=dtype, device=tensor.device)

	tensor.mul_(std_tensor[:, None, None]).add_(mean_tensor[:, None, None])

	return tensor

# Create a PIL image from a normalised tensor
def image_from_normed_tensor(tensor, mean, stddev, mode):
	# tensor = Tensor to unnormalise and convert to a PIL image (should be DxHxW)
	# mean = Iterable of mean values for each channel (length D)
	# stddev = Iterable of standard deviation values for each channel (length D)
	# mode = PIL image mode to interpret the unnormalised tensor data as (e.g. 'RGB', 'L')
	if mode is None:
		raise ValueError("PIL image mode must be explicitly specified")
	tensor = unnormalise_tensor(tensor, mean, stddev, inplace=False)
	return torchvision.transforms.functional.to_pil_image(tensor, mode=mode)

# Convert PIL image to numpy image array
def numpyify_pil(image):
	# image = PIL image to numpyify (without any normalization)
	# Return the resulting image array
	default_float_dtype = torch.get_default_dtype()
	return np.array(image, dtype=numpy_dtype_from_torch(default_float_dtype), copy=False)

# Convert numpy image array to tensor
def tensorize_numpy_image(image):
	# image = Numpy image array
	# Return the resulting image tensor
	tensor = torch.from_numpy(image)
	tensor = tensor.permute((2, 0, 1)) if tensor.dim() >= 3 else tensor.unsqueeze(0)
	return tensor.contiguous()

# Tensorize PIL image
def tensorize_pil(image):
	# image = PIL image to tensorize (without any normalization)
	# Return the resulting image tensor
	return tensorize_numpy_image(numpyify_pil(image))

# Convert tensor to numpy image array
def numpyify_tensor(image, dtype=None, contiguous=True):
	# image = Image tensor
	# dtype = Desired numpy dtype (None = Default based on tensor dtype)
	# contiguous = Whether to ensure the resulting numpy array is contiguous
	# Return the resulting numpy image array
	numpy_image = image.detach().cpu().numpy()
	if numpy_image.ndim >= 3:
		if numpy_image.shape[0] == 1:
			numpy_image = numpy_image.squeeze(0)
		else:
			numpy_image = np.transpose(numpy_image, (1, 2, 0))
	if dtype is not None and numpy_image.dtype != dtype:
		numpy_image = numpy_image.astype(dtype, copy=True)
	elif contiguous:
		numpy_image = np.ascontiguousarray(numpy_image)  # May or may not make a copy depending on whether numpy_image is already contiguous
	return numpy_image

# Convert numpy image array to PIL image
DEFAULT_PIL_MODE = {1: 'L', 2: 'LA', 3: 'RGB', 4: 'RGBA'}
def pilify_numpy_image(image, mode=None):
	# image = Numpy image array
	# mode = PIL image mode to use
	# Return the resulting PIL image
	if mode is None:
		mode = DEFAULT_PIL_MODE[image.shape[-1] if image.ndim >= 3 else 1]
	if image.dtype != np.uint8 and image.dtype != np.int8 and mode != 'I' and mode != 'F':
		image = image.astype(dtype=np.uint8)  # Note: This fixes floating point values (rounds them towards zero)
	return PIL.Image.fromarray(image, mode=mode)

# Convert tensor to PIL image
def pilify_tensor(image, mode=None):
	# image = Image tensor
	# mode = PIL image mode to use
	# Return the resulting PIL image
	return pilify_numpy_image(numpyify_tensor(image, contiguous=False), mode=mode)

# Linearly stretch a tensor
def linear_stretch(tensor, in_range, out_range, clip=False):
	# tensor = Tensor to linearly stretch input range -> output range
	# in_range = Pair of floats specifying input range
	# out_range = Pair of floats specifying output range
	# clip = Whether to clip the result to the output range

	a, b = in_range
	c, d = out_range

	if a == b:
		if tensor.dtype.is_floating_point:
			return tensor.new_full(tensor.shape, 0.5 * (c + d))
		else:
			return tensor.new_full(tensor.shape, 0.5 * (c + d), dtype=float)

	if tensor.dtype.is_floating_point:
		copied = False
	else:
		tensor = tensor.to(float, copy=True)
		copied = True

	if a == c and b == d:
		if not copied:
			tensor = tensor.clone()
	else:
		m = (d - c) / (b - a)
		tensor = tensor.mul_(m) if copied else tensor.mul(m)
		offset = c - m * a
		if offset != 0:
			tensor.add_(offset)

	if clip:
		tensor.clip_(min=c, max=d)

	return tensor

# Linearly stretch a tensor based on its quantiles
DEFAULT_QUANTILES = torch.Tensor((0.0005, 0.9995))
def quantile_stretch(tensor, out_range, clip=True, quantiles=None):
	# tensor = Floating point tensor to linearly stretch quantile range -> output range
	# out_range = Pair of floats specifying output range
	# clip = Whether to clip the result to the output range
	# quantiles = Quantile range as torch tensor with 2 elements
	return linear_stretch(tensor, tensor.quantile(quantiles if quantiles is not None else DEFAULT_QUANTILES).tolist(), out_range, clip=clip)

# Generate a 2D Gaussian tensor
def gaussian_tensor(stddev, cutoff=0.01):
	# stddev = Standard deviation of the required Gaussian (equal in x and y directions, can be floating point)
	# mean_offset = Offset to apply to the mean of the Gaussian (intended for subpixel adjustments)
	# cutoff = Threshold of the Gaussian beyond which to cut it off
	# Return required Gaussian as a square torch tensor, index offset (plus/minus) specifying tensor size
	if not 0 < cutoff < 1:
		raise ValueError(f"Cutoff needs to be in the range (0, 1): {cutoff}")
	N = math.floor(stddev * math.sqrt(-2 * math.log(cutoff)))
	xyvec = torch.arange(-N, N + 1).square_().div(-2 * stddev * stddev)
	G = torch.exp(xyvec[None, :] + xyvec[:, None])
	G[G < cutoff] = 0
	return G, N

# Find maxima in a 2D tensor/array
# noinspection PyUnresolvedReferences
def find_maxima(array, min_value, min_area):
	# array = 2D tensor/array to find maxima in (3D tensor/array may be passed if the first dimension is size 1)
	# min_value = Threshold below which to ignore values in the tensor/array when finding maxima
	# min_area = Minimum area of regions to consider the local maxima for (pixel count)
	# Return List[Tuple[float, float]] of maxima locations
	if torch.is_tensor(array):
		array = array.detach().cpu().numpy()
	if array.ndim == 3:
		array = array.squeeze(0)
	if array.ndim != 2:
		raise ValueError(f"Find maxima expects a 2D tensor/array, got {array.ndim}D")
	if array.size <= 1:
		return []
	binary_array = cv2.compare(array, min_value, cv2.CMP_GE)
	contours, hierarchy = cv2.findContours(binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	maxima = []
	for contour in contours:
		if contour.shape[0] >= 3:  # Need at least a triangle in order to have non-zero area
			moments = cv2.moments(contour)
			contour_area = moments['m00']
			if contour_area >= min_area:
				maxima.append((moments['m10'] / contour_area + 0.5, moments['m01'] / contour_area + 0.5))  # Note: Stems must be in axis coordinates and the contours are in pixel coordinates, hence +0.5
	return maxima
# EOF
