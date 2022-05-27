# PNN Library: System information utilities

# Imports
import os
import sys
import json
from typing import Any
import datetime
import platform
import cpuinfo
import numpy as np
import matplotlib
import PIL
import pycuda.driver
import torch
# noinspection PyUnresolvedReferences, PyProtectedMember, PyPep8Naming
import torch._C as torchC
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mkldnn
import torch.backends.openmp
import torch.cuda.nccl
import torch.utils.collect_env
import torch.version
import torchvision
from util.print import print_warn
import util.nvsmi
from pnnlib.util import device_util

# Get a summary of the system information
# noinspection PyProtectedMember
def get_system_info_summary(cpu_only=False) -> Any:

	def version_int_to_str(v, minor):
		if isinstance(v, tuple):
			major_version = v[0]
			minor_version = v[1]
			patch_version = v[2] if len(v) >= 3 else 0
		else:
			major_version = v // 1000
			remainder = v % 1000
			minor_version = remainder // minor
			patch_version = remainder % minor
		if patch_version:
			return f"{major_version}.{minor_version}.{patch_version}", (major_version, minor_version, patch_version)
		else:
			return f"{major_version}.{minor_version}", (major_version, minor_version, patch_version)

	def version_str_to_tup(v, parts=3):
		v_list = v.split('.')[:parts]
		v_list.extend([0] * (parts - len(v_list)))
		try:
			v_tup = tuple(int(part) for part in v_list)
		except (ValueError, TypeError):
			v_tup = None
		return v_tup

	info = {
		'timestamp': datetime.datetime.now(),
		'cpu_only': cpu_only,
	}

	python_version = platform.python_version()
	info['python'] = {
		'version': python_version,
		'version_tup': platform.python_version_tuple(),
		'version_details': ' '.join(s.strip() for s in sys.version.split()),
		'versions': {
			'Python': python_version,
			'NumPy': np.__version__,
			'PIL': PIL.__version__,
			'Matplotlib': matplotlib.__version__,
		},
	}

	uname = os.uname()
	try:
		cpu_info = cpuinfo.get_cpu_info()  # Note: This sometimes fails in debug mode in PyCharm due to the use of an internal subprocess, leading to an empty string being passed to the JSON decoder...
	except json.JSONDecodeError:
		cpu_info = None
	gcc_version = torch.utils.collect_env.run_and_parse_first_match(torch.utils.collect_env.run, 'gcc --version', r'gcc(?:\s+\(.*\))?\s+(\S+)')
	kernel_parts = uname.release.split('-')
	kernel_parts[:1] = kernel_parts[0].split('.')
	info['system'] = {
		'os': torch.utils.collect_env.get_os(run_lambda=torch.utils.collect_env.run),
		'os_details': uname.version,
		'kernel': uname.release,
		'kernel_tup': tuple(kernel_parts),           # Note: The length of this tuple may vary based on the number of parts in the 'kernel' string
		'hostname': uname.nodename,
		'hosttype': f'{uname.sysname} {uname.machine}',
		'gcc': gcc_version,                          # Note: The strict format of this cannot be relied upon (there may be additional text to the version number)
		'gcc_tup': version_str_to_tup(gcc_version),  # Note: This may be None if 'gcc' is not of a recognised format
		'cpu': {
			'have_info': bool(cpu_info),
			'brand': cpu_info['brand_raw'] if cpu_info else 'Unknown',
			'bits': cpu_info['bits'] if cpu_info else 0,
			'count': cpu_info['count'] if cpu_info else 0,
			'hz_current': cpu_info['hz_actual_friendly'] if cpu_info else '0 GHz',
		},
	}

	gpu_count = 0 if cpu_only else torch.cuda.device_count()
	info['gpu'] = {
		'have_info': not cpu_only,
		'count': gpu_count,
		'list': [{'index': i, 'name': torch.cuda.get_device_name(i), 'pci_id': device_util.get_device_pci_bus_id_drv(i)} for i in range(gpu_count)],
	}

	driver_max_cuda = (None, None)
	if not cpu_only:
		# noinspection PyUnresolvedReferences
		driver_max_cuda = version_int_to_str(pycuda.driver.get_driver_version(), 10)

	nvsmi = util.nvsmi.NvidiaSMI()
	driver_version = nvsmi.DeviceQuery('driver_version')['driver_version']
	info['gpu']['driver'] = {
		'version': driver_version,
		'version_tup': version_str_to_tup(driver_version),  # Note: This may be None if 'version' is not of a recognised format
		'max_cuda': driver_max_cuda[0],                     # Note: This may be None even if cpu_only is False (if retrieval of this information failed)
		'max_cuda_tup': driver_max_cuda[1],                 # Note: This may be None even if cpu_only is False (if retrieval of this information failed)
		'details': {},
	}

	with open('/proc/driver/nvidia/version') as f:
		for line in f.readlines():
			infoo = line.split(':', 1)
			if len(infoo) == 2:
				info['gpu']['driver']['details'][infoo[0].strip()] = ' '.join(infoo[1].split())

	pytorch_cuda_ver = torch.version.cuda
	pytorch_cudnn = version_int_to_str(torch.backends.cudnn.version(), 100)
	pytorch_nccl = version_int_to_str(torch.cuda.nccl.version(), 100)
	info['pytorch'] = {
		'version': torch.version.__version__,
		'commit': torch.version.git_version,
		'debug': torch.version.debug,
		'compiled_with': {
			'cuda': pytorch_cuda_ver,
			'cuda_tup': version_str_to_tup(pytorch_cuda_ver),  # Note: This may be None if 'cuda' is not of a recognised format
			'cudnn': pytorch_cudnn[0],
			'cudnn_tup': pytorch_cudnn[1],
			'nccl': pytorch_nccl[0],
			'nccl_tup': pytorch_nccl[1],
		},
		'backends': {
			'cuDNN': torch.backends.cudnn.is_available(),
			'OpenMP': torch.backends.openmp.is_available(),
			'MKL': torch.backends.mkl.is_available(),
			'MKL-DNN': torch.backends.mkldnn.is_available(),
		},
	}

	torchvision_cuda = version_int_to_str(torchvision.version.cuda, 10)
	info['torchvision'] = {
		'version': torchvision.__version__,
		'commit': torchvision.version.git_version,
		'compiled_with': {
			'cuda': torchvision_cuda[0],
			'cuda_tup': torchvision_cuda[1],
		},
		'image_backend': torchvision.get_image_backend(),
		'video_backend': torchvision.get_video_backend(),
	}

	return info

# Print a summary of the system information
def print_system_info_summary(cpu_only=False):

	info = get_system_info_summary(cpu_only=cpu_only)

	print("System information:")
	print(f"  Info time = {info['timestamp']}")

	python = info['python']
	print("  Python:")
	print(f"    Version = {python['version']}")
	print(f"    Version details = {python['version_details']}")
	for package, version in sorted(python['versions'].items(), key=lambda t: t[0].casefold()):
		if package != 'Python':
			print(f"    {package} version = {version.lower()}")

	system = info['system']
	print("  System:")
	print(f"    OS = {system['os']}")
	print(f"    Kernel = {system['kernel']}")
	print(f"    Machine = {system['hostname']} ({system['hosttype']})")
	cpu = system['cpu']
	if cpu['have_info']:
		print(f"    CPU = {cpu['brand']}")
		print(f"    CPU details = {cpu['bits']}-bit, {cpu['count']} logical cores, currently at {cpu['hz_current']}")
	else:
		print("    CPU = Unknown")
		print("    CPU details = Unknown")
	print(f"    OS details = {system['os_details']}")
	print(f"    GCC version = {system['gcc']}")

	gpu = info['gpu']
	if gpu['have_info']:

		print("  GPU devices:")
		for gpu_info in gpu['list']:
			print(f"    GPU {gpu_info['index']} = {gpu_info['name']} ({gpu_info['pci_id']})")
		driver = gpu['driver']

		print("  Nvidia driver:")
		print(f"    Driver version = {driver['version']}")
		print(f"    Supports CUDA versions (up to) = {driver['max_cuda']}")
		for detail, value in sorted(driver['details'].items(), key=lambda t: t[0].casefold()):
			print(f"    {detail} = {value}")

	pytorch = info['pytorch']
	print("  PyTorch:")
	print(f"    Version = {pytorch['version']}")
	print(f"    Git commit = {pytorch['commit']}")
	print(f"    Release build = {not pytorch['debug']}")
	if pytorch['debug']:
		print_warn("PyTorch was compiled in DEBUG mode")
	compiled_with = pytorch['compiled_with']
	print(f"    Compiled with CUDA = {compiled_with['cuda']}")
	print(f"    Compiled with cuDNN = {compiled_with['cudnn']}")
	print(f"    Compiled with NCCL = {compiled_with['nccl']}")
	for backend, available in sorted(pytorch['backends'].items(), key=lambda t: t[0].casefold()):
		print(f"    {backend} available = {available}")

	vision = info['torchvision']
	print("  TorchVision:")
	print(f"    Version = {vision['version']}")
	print(f"    Git commit = {vision['commit']}")
	compiled_with = vision['compiled_with']
	print(f"    Compiled with CUDA = {compiled_with['cuda']}")
	print(f"    Image backend = {vision['image_backend']}")
	print(f"    Video backend = {vision['video_backend']}")

	print()

	return info
# EOF
