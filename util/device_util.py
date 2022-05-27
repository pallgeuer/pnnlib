# PNN Library: Torch device utilities

# Imports
import os
import re
import time
import ctypes
import random
import contextlib
from enum import Enum, auto
from typing import Union, Optional, List
import pycuda.driver
import torch
import util.contextman
import util.execlock
from util.execlock import SYSLOCK_PATH, DEFAULT_TIMEOUT, DEFAULT_CHECK_INTERVAL
from pnnlib.util import cudadrv, cudart

# Resolve a device specification to a torch.device
def resolve_device(device: Union[None, int, str, torch.device], default: Union[None, int, str, torch.device] = None, default_type: str = 'cuda') -> torch.device:
	# device = Device specification to resolve (None => Use default, int => Device index, str => Torch device string or CUDA PCI bus ID (abbreviations possible), torch.device => Torch device)
	# default = If device is None, default device specification to resolve instead (int/str/torch.device as above, None => No default)
	# default_type = Device string to use if resolution of device/default leads to a plain int device index
	# Return the required resolved torch.device

	if device is None:
		device = default

	if isinstance(device, torch.device):
		return device
	elif isinstance(device, str):
		match = re.fullmatch(r'(([0-9A-Fa-f]{1,8}:)?[0-9A-Fa-f])?[0-9A-Fa-f]:[0-9A-Fa-f]{2}(\.[0-9A-Fa-f])?', device)
		if match:
			device_lower = device.lower()
			for device_index in range(torch.cuda.device_count()):
				pci_bus_id = get_device_pci_bus_id_drv(device_index).lower()
				if pci_bus_id.endswith(device_lower, 0, -2 if match.group(3) is None else None):
					break
			else:
				raise LookupError(f"PCI bus ID not found (or not an available GPU): {device}")
			return torch.device(type='cuda', index=device_index)
		else:
			return torch.device(device)  # Note: Raises RuntimeError if device is not a valid device string
	elif isinstance(device, int) and device >= 0:
		return torch.device(type=default_type, index=device)  # Note: Raises RuntimeError if default_type is not a valid device string
	else:
		raise TypeError(f"Invalid torch device specification: {device}")

# Resolve a device specification to a device index
def resolve_device_index(device: Union[None, int, str, torch.device], default: Union[None, int, str, torch.device] = None, enforce_type: Optional[str] = None) -> int:
	# device = Device specification to resolve (None => Use default, int => Device index, str => Torch device string, torch.device => Torch device)
	# default = If device is None, the default device specification to resolve instead (int/str/torch.device as above, None => No default)
	# enforce_type = Device type to check for and enforce, e.g. 'cuda' (None => Allow all types)
	# Return the required resolved device index (int)

	if device is None:
		device = default

	if isinstance(device, str):
		device = torch.device(device)  # Note: Raises RuntimeError if device is not a valid device string

	if isinstance(device, torch.device):
		if enforce_type is not None and device.type != enforce_type:
			raise TypeError(f"Resolved device is of type '{device.type}', but '{enforce_type}' is required and being enforced")
		if device.index is None:
			if device.type == 'cpu':
				return 0
			elif device.type == 'cuda':
				# noinspection PyUnresolvedReferences
				return torch.cuda.current_device() if torch.cuda.is_initialized() else 0  # If CUDA is not initialised, then no call to torch.cuda.set_device() can have happened so far, meaning that the current device must be the CUDA default of 0 still
			else:
				raise TypeError(f"Cannot resolve a 'None' device index for devices of type '{device.type}'")
		else:
			return device.index
	elif isinstance(device, int) and device >= 0:
		return device
	else:
		raise TypeError(f"Invalid torch device specification: {device}")

# Get the PCI bus ID of a particular CUDA device (using CUDA driver API)
# noinspection PyUnresolvedReferences
def get_device_pci_bus_id_drv(device: Union[None, int, str, torch.device]) -> str:
	# device = CUDA device to get the PCI bus ID of (see resolve_device_index() function)
	# Return a string representing the PCI bus ID of the specified CUDA device

	device_index = resolve_device_index(device, enforce_type='cuda')

	cudadrv.ensure_initialised()

	try:
		drv_device = pycuda.driver.Device(device_index)
		# noinspection PyArgumentList
		pci_bus_id = drv_device.pci_bus_id()
	except pycuda.driver.Error:
		raise LookupError(f"Failed to retrieve PCI bus ID for device index {device_index}")

	return pci_bus_id

# Get the PCI bus ID of a particular CUDA device (using CUDA runtime API)
def get_device_pci_bus_id_rt_DISABLED(device: Union[None, int, str, torch.device]) -> str:  # Note: Disabled as cudart.call_cudart_func() was disabled
	# device = CUDA device to get the PCI bus ID of (see resolve_device_index() function)
	# Return a string representing the PCI bus ID of the specified CUDA device

	device_index = resolve_device_index(device, enforce_type='cuda')

	c_pci_bus_id = ctypes.create_string_buffer(16)
	error = cudart.call_cudart_func_DISABLED('cudaDeviceGetPCIBusId', c_pci_bus_id, 16, device_index)
	if error:
		raise LookupError(f"Failed to retrieve PCI bus ID for device index {device_index} ({error[0]}: {error[1]})")

	return c_pci_bus_id.value.decode('utf-8')

# Helper function for torch device locks
def device_lock_path(device: Union[None, str, torch.device], relative_to=SYSLOCK_PATH):
	# device = Torch device to get the lock path for (torch.device or 'autoselect' or None)
	# relative_to = Path relative to which to resolve the device lock  (None => None)
	# Return the torch device lock path
	if device is None:
		return None
	elif device == 'autoselect':
		lock_file = 'cuda_autoselect.lock'
	elif device.type == 'cpu':
		lock_file = 'cpu.lock'
	elif device.type == 'cuda':
		pci_bus_id = get_device_pci_bus_id_drv(device)
		lock_file = f'cuda_{pci_bus_id}.lock'
	else:
		raise TypeError(f"Unsupported torch device type: {device.type}")
	return os.path.join(relative_to, 'device', lock_file)

# Context manager that allows system-wide locking of a torch device (only protects from simultaneous device accesses by other processes that use this context manager as well)
class DeviceLock(util.execlock.ExecutionLock):

	def __init__(self, device, blocking=True, timeout=DEFAULT_TIMEOUT, check_interval=DEFAULT_CHECK_INTERVAL, lock_delay=0):
		# device = Torch device to lock (see resolve_device() function, or None)
		# blocking, timeout, check_interval, lock_delay = See ExecutionLock
		self._device = None
		super().__init__(None, relative_to=SYSLOCK_PATH, makedirs=True, blocking=blocking, timeout=timeout, check_interval=check_interval, lock_delay=lock_delay)
		self.set_device(device)

	def set_device(self, device):
		if device is not None:
			device = resolve_device(device)
		lock_abspath = device_lock_path(device, relative_to=SYSLOCK_PATH)
		self.set_lock_path(lock_abspath, relative_to=SYSLOCK_PATH, makedirs=True)
		self._device = device

	@property
	def device(self):
		return self._device

# Context manager that wraps an existing DeviceLock instance in a verbose fashion
class VerboseDeviceLock:

	def __init__(self, device_lock, file=None, newline=True):
		# device_lock = Existing DeviceLock instance to wrap in a verbose fashion
		# file = File to print all verbose output to
		# newline = Whether the verbose output should include a trailing empty line
		self.device_lock = device_lock
		self.file = file
		self.newline = newline

	def __enter__(self):
		with contextlib.ExitStack() as stack:
			print(f"Acquiring device lock: {self.device_lock.lock_path}", file=self.file)
			already_locked = self.device_lock.locked
			result = stack.enter_context(self.device_lock)
			if already_locked:
				print("Lock acquired (device was already locked by us)!", file=self.file)
			else:
				print("Lock acquired!", file=self.file)
			if self.newline:
				print(file=self.file)
			stack.pop_all()
		return result

	def __exit__(self, exc_type, exc_val, exc_tb):
		print(f"Releasing device lock: {self.device_lock.lock_path}", file=self.file)
		suppress = self.device_lock.__exit__(exc_type, exc_val, exc_tb)
		if self.device_lock.locked:
			print("Lock released (but device is still locked by us from elsewhere)!", file=self.file)
		else:
			print("Lock released!", file=self.file)
		if self.newline:
			print(file=self.file)
		return suppress

# GPU level enumeration
class GPULevel(Enum):
	Unlocked = 0
	BaseLock = auto()
	LowMemNoExec = auto()
	NormalMemLowExec = auto()
	NormalMemHighExec = auto()
	HighMemHighExec = auto()

# GPU level execution locking class
class GPULevelLock(util.execlock.RunLevelLock):

	def __init__(self, device, level_counts, running_thres=GPULevel.NormalMemLowExec, solo_thres=GPULevel.NormalMemHighExec, check_interval=DEFAULT_CHECK_INTERVAL, lock_delay=0, autoselect_cb=None):
		# device = Torch device to lock (see resolve_device() function, or None)
		# level_counts = Iterable of maximum counts to assign to the run levels defined in GPULevel (ignoring the first/base level)
		# running_thres, solo_thres, check_interval, lock_delay = See RunLevelLock
		# autoselect_cb = Optional callback taking a single torch device as an argument, called when GPU auto-selection has successfully selected a device

		self._device = None
		self._autoselect_cb = autoselect_cb

		# noinspection PyTypeChecker
		run_levels = dict(zip(list(GPULevel)[2:], level_counts))
		super().__init__(None, GPULevel.Unlocked, GPULevel.BaseLock, run_levels, running_thres=running_thres, solo_thres=solo_thres, relative_to=SYSLOCK_PATH, makedirs=True, check_interval=check_interval, lock_delay=lock_delay)
		self.set_device(device, makedirs=True)

		autoselect_lock_abspath = device_lock_path('autoselect', relative_to=SYSLOCK_PATH)
		self._autoselect_lock = util.execlock.ExecutionLock(autoselect_lock_abspath, relative_to=SYSLOCK_PATH, makedirs=True, dir_mode=self._dir_mode, file_mode=self._file_mode, umask=self._umask, blocking=True, check_interval=check_interval, lock_delay=0)

	def set_device(self, device, makedirs: Optional[bool] = True):
		if device is not None:
			device = resolve_device(device)
		lock_abspath = device_lock_path(device, relative_to=SYSLOCK_PATH)
		self.set_lock_path(lock_abspath, relative_to=SYSLOCK_PATH, makedirs=makedirs)
		self._device = device

	@property
	def device(self):
		return self._device

	def __exit__(self, exc_type, exc_val, exc_tb):
		suppress = super().__exit__(exc_type, exc_val, exc_tb)
		self._autoselect_lock.ensure_locked(False)
		return suppress

	def set_autoselect_cb(self, autoselect_cb):
		self._autoselect_cb = autoselect_cb

	def _lock_invalid_cb(self, done):
		if done:
			self._autoselect_lock.ensure_locked(False)
		elif self._device is None:
			self._autoselect_device()
			if callable(self._autoselect_cb):
				self._autoselect_cb(self._device)

	def _autoselect_device(self):

		self._autoselect_device_cb(False)

		# noinspection PyUnresolvedReferences
		if torch.cuda.is_initialized():
			raise util.execlock.ExecLockError(f"Device auto-selection should occur PRIOR to CUDA initialization")

		max_ilevel = max(self._running_ilevel, self._solo_ilevel)
		device_list = [torch.device(type='cuda', index=device_index) for device_index in range(torch.cuda.device_count())]
		device_status: List[Optional[util.execlock.RunLockStatus]] = [None] * len(device_list)
		orig_makedirs = self._makedirs
		try:
			while True:

				self._autoselect_lock.ensure_locked(True)

				for i, device in enumerate(device_list):
					self.set_device(device, makedirs=None)
					device_status[i] = self._lock_status(max_ilevel=max_ilevel)
				self._makedirs = False

				available_devices = []
				for i, status in enumerate(device_status):
					num_processes = len(status.processes)
					lowest_lock_status = status.lock[2]
					max_processes = lowest_lock_status.max_count if lowest_lock_status is not None else 0
					if not status.base_lockable or num_processes >= max_processes:
						continue
					free_counts = [0 if cstat is None else cstat.free_count for cstat in reversed(status.lock)]
					available_devices.append((-num_processes, *free_counts, status.solo_lockable, random.random(), i))

				if available_devices:
					best_device = max(available_devices)
					self.set_device(device_list[best_device[-1]], makedirs=None)
					break

				self._autoselect_lock.ensure_locked(False)
				time.sleep(len(device_list) * self.check_interval * (1 + random.random()))

		finally:
			self._makedirs = orig_makedirs

		self._autoselect_device_cb(True)

	def _autoselect_device_cb(self, done):
		pass

# GPU level execution locking class (verbose)
class GPULevelLockV(GPULevelLock):

	def __init__(self, *args, verbose=True, newline=True, file=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.verbose = verbose
		self.newline = newline
		self.file = file
		self._line_count = 0
		self._sticky_config = None
		self._in_progress = False

	# noinspection PyProtectedMember
	class ConfigCM(metaclass=util.contextman.ReentrantMeta):

		def __init__(self, run_lock, verbose=None, newline=None, block_newline=None, file=False, restore_to_init=False, sticky=False):
			self._run_lock = run_lock
			self._verbose = verbose
			self._newline = newline
			self._block_newline = not self._newline if block_newline is None else block_newline
			self._file = file
			self._restore_to_init = restore_to_init
			self._sticky = sticky
			self._prev_config = self._run_lock._current()
			self._prev_count = 0

		def __enter__(self):
			if not self._restore_to_init:
				self._prev_config = self._run_lock._current()
			self._run_lock._configure(verbose=self._verbose, newline=self._newline, file=self._file, sticky=self._sticky)
			self._prev_count = self._run_lock._line_count
			return self._run_lock

		def __exit__(self, exc_type, exc_val, exc_tb):
			if self._block_newline and self._run_lock._line_count != self._prev_count:
				print(file=self._run_lock.file)
			self._run_lock._restore(self._prev_config, sticky=self._sticky)
			return False

	def config(self, verbose=None, newline=None, block_newline=None, file=False, restore_to_init=False, sticky=False):
		return self.ConfigCM(self, verbose=verbose, newline=newline, block_newline=block_newline, file=file, restore_to_init=restore_to_init, sticky=sticky)

	def _current(self):
		return self.verbose, self.newline, self.file

	def _configure_raw(self, verbose, newline, file):
		if verbose is not None:
			self.verbose = verbose
		if newline is not None:
			self.newline = newline
		if file is not False:
			self.file = file

	def _configure(self, verbose, newline, file, sticky):
		self._configure_raw(verbose, newline, file)
		if sticky:
			self._sticky_config = (verbose, newline, file)
		elif self._sticky_config is not None:
			self._configure_raw(*self._sticky_config)

	def _restore(self, prev_config, sticky):
		self.verbose, self.newline, self.file = prev_config
		if sticky:
			self._sticky_config = None
		elif self._sticky_config is not None:
			self._configure_raw(*self._sticky_config)

	def _print(self, *args, **kwargs):
		print(*args, file=self.file, **kwargs)
		self._line_count += 1

	def __enter__(self):
		if not self.verbose:
			return super().__enter__()
		with contextlib.ExitStack() as stack:
			self._print(f"Starting GPU run level context management: {'Will auto-select' if self._lock_path is None else self._lock_path}")
			result = super().__enter__()
			stack.push(super().__exit__)
			self._print(f"Current GPU run level: {self.current_level.name}")
			if self.newline:
				print(file=self.file)
			stack.pop_all()
		return result

	def __exit__(self, exc_type, exc_val, exc_tb):
		if not self.verbose:
			return super().__exit__(exc_type, exc_val, exc_tb)
		try:
			self._print(f"Current GPU run level: {self.current_level.name}")
		finally:
			with self.config(newline=False, block_newline=False):
				suppress = super().__exit__(exc_type, exc_val, exc_tb)
		self._print(f"Stopping GPU run level context management: {'Was going to auto-select' if self._lock_path is None else self._lock_path}")
		if self.newline:
			print(file=self.file)
		return suppress

	def _set_ilevel_cb(self, cur_ilevel, new_ilevel, done):
		if cur_ilevel == new_ilevel:
			return
		if done:
			if self.newline:
				print(file=self.file)
		else:
			self._print(f"{'Lowering' if new_ilevel < cur_ilevel else 'Elevating'} GPU run level to: {self._level_list[new_ilevel].name}")
		self._in_progress = not done

	def _go_solo_cb(self, solo, done):
		if done:
			if self.newline:
				print(file=self.file)
		else:
			self._print(f"{'Entering' if solo else 'Exiting'} solo GPU execution mode")
		self._in_progress = not done

	def _yield_solo_cb(self, done):
		if done:
			self._print("Resuming execution of this process")
			if self.newline:
				print(file=self.file)
		else:
			self._print("Yielding to solo GPU execution request made by another process")
		self._in_progress = not done

	def _autoselect_device_cb(self, done):
		if done:
			self._print(f"Auto-selected device: {repr(self._device)}")
			if self.newline and not self._in_progress:
				print(file=self.file)
		else:
			self._print(f"Searching for a free CUDA device...")
# EOF
