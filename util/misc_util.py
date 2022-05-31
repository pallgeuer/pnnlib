# PNN Library: Miscellaneous utilities

# Imports
import random
import numpy as np
import torch
import torch.random
import torch.backends.cudnn
from ppyutil.print import printc, print_hor_line

# Constants
header_color = 'cyan'

# Print a header
def print_header(header, color=header_color):
	print_hor_line(color)
	printc(header, color)
	print()

# Update the determinism of the code
def update_determinism(deterministic, manual_seed=0, cudnn_benchmark_mode=False):
	# Causes of indeterminism:
	#  - Some PyTorch functions (https://pytorch.org/docs/stable/notes/randomness.html#pytorch)
	#    For example, ConvTranspose2d causes non-determinism!
	#  - Python hashing (PYTHONHASHSEED) => Would need to be set in containing env prior to invoking python
	#  - Multithreading, mutexes, file locking or the spawning of child processes
	#  - Anything that relies on time or speed of execution in a conditional way
	if deterministic:
		random.seed(manual_seed)
		np.random.seed(manual_seed)
		torch.manual_seed(manual_seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:
		random.seed()
		np.random.seed(None)
		# noinspection PyUnresolvedReferences
		torch.manual_seed(torch.random.default_generator.seed() & ((1 << 63) - 1))
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = cudnn_benchmark_mode
# EOF
