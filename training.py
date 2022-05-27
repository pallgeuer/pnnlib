# PNN Library: Neural network training utilities

# Imports
import os
import math
import timeit
import warnings
from typing import Dict, Any
import torch
import torch.optim.lr_scheduler
import util.nvsmi
import util.math
from collections import deque
from pnnlib import dataset
from pnnlib.util import device_util

# Global constants
loss_fmt = '#.4g'
rloss_fmt = '.2%'
lr_fmt = '.3g'

#
# Stopper classes
#

# Stopper base class
class StopperBase:

	def __init__(self):
		pass

	def __repr__(self):
		return f"{self.__class__.__name__}(stop=False)"

	def should_stop(self, epoch, epoch_losses, best_losses, min_losses, initial_lr, current_lr):
		# epoch = Epoch number that just finished (1-based)
		# epoch_losses = DatasetTuple of the losses of the just-finished epoch (the test loss may be None if it is not available)
		# best_losses = DatasetTuple of the losses of the 'best' epoch so far (the test loss may be None if it is not available)
		# min_losses = DatasetTuple of the absolute minimum losses seen so far, independent for each dataset type (the test loss may be None if it is not available)
		# initial_lr = List of initial learning rate(s)
		# current_lr = List of current learning rate(s)
		# Return whether the training process should stop, and a short string reason why stopping did or did not happen (may be None)
		return False, None

# Train/valid patience stopper class
class TVPatienceStopper(StopperBase):

	def __init__(self, train_patience_epochs=0, valid_patience_epochs=0):
		super().__init__()
		self.train_patience_epochs = train_patience_epochs
		self.valid_patience_epochs = valid_patience_epochs
		self.train_min = math.inf
		self.valid_min = math.inf
		self.train_patience = 0
		self.valid_patience = 0

	def __repr__(self):
		return (
			f"{self.__class__.__name__}(\n"
			f"  train_patience_epochs={self.train_patience_epochs},\n"
			f"  valid_patience_epochs={self.valid_patience_epochs},\n"
			f")"
		)

	def should_stop(self, epoch, epoch_losses, best_losses, min_losses, initial_lr, current_lr):

		if min_losses.train == self.train_min:
			self.train_patience += 1
		else:
			self.train_min = min_losses.train
			self.train_patience = 0
		train_exceeded = self.train_patience >= self.train_patience_epochs

		if min_losses.valid == self.valid_min:
			self.valid_patience += 1
		else:
			self.valid_min = min_losses.valid
			self.valid_patience = 0
		valid_exceeded = self.valid_patience >= self.valid_patience_epochs

		consider_train = self.train_patience_epochs > 0
		consider_valid = self.valid_patience_epochs > 0

		if consider_train or consider_valid:
			should_stop = (train_exceeded or not consider_train) and (valid_exceeded or not consider_valid)
			reason = "Reached " if should_stop else "Have "
			reason_losses = []
			if consider_train:
				reason_losses.append(f"{self.train_patience}/{self.train_patience_epochs} non-minimal training losses")
			if consider_valid:
				reason_losses.append(f"{self.valid_patience}/{self.valid_patience_epochs} non-minimal validation losses")
			reason += ', '.join(reason_losses)
			return should_stop, reason
		else:
			return False, None

#
# Learning rate schedulers
#

# Convert a list of optimizer learning rates to string form
def lr_string(lrs, brackets=None, sep=', '):
	# lrs = List of learning rates, e.g. [group['lr'] for group in optimizer.param_groups]
	# brackets = Whether to include square brackets in the string (False = Never, True = Always, None = Only if >1 elements)
	# sep = Separator string to use between learning rate entries
	# Return the required string representation of the list of learning rates
	if brackets is None:
		brackets = len(lrs) > 1
	if brackets:
		return f"[{sep.join(f'{lr:{lr_fmt}}' for lr in lrs)}]"
	else:
		return f"{sep.join(f'{lr:{lr_fmt}}' for lr in lrs)}"

# Reduce learning rate on plateau minimum class
# noinspection PyUnresolvedReferences, PyAttributeOutsideInit, PyProtectedMember
class ReduceLROnPlateauMin(torch.optim.lr_scheduler.ReduceLROnPlateau):

	def __init__(self, optimizer, grace=6, threshold_scale=1, **kwargs):

		self.grace = grace
		self.grace_skip = util.math.secretary_problem_soln(self.grace)
		self.grace_test = self.grace - self.grace_skip
		self.threshold_scale = threshold_scale

		self.cycbuf = deque(maxlen=self.grace_skip)
		self.grace_value = None
		self.grace_counter = 0
		self.threshold_crossed = False
		self.event = None

		super().__init__(optimizer, **kwargs)

		if self.patience < 1:
			raise ValueError(f"Patience must be at least 1: {self.patience}")
		if self.threshold_scale <= 0:
			raise ValueError(f"Threshold scale must be positive: {self.threshold_scale}")

	def __repr__(self):
		return f"{self.__class__.__name__}(factor={self.factor}, cooldown={self.cooldown}, patience={self.patience}, grace_skip={self.grace_skip}, grace_test={self.grace_test}, threshold_scale={self.threshold_scale})"

	def _reset(self):
		super()._reset()
		self.cycbuf.clear()
		self.grace_value = None
		self.grace_counter = 0
		self.threshold_crossed = False
		self.event = None

	@property
	def in_grace(self):
		return self.grace_value is not None

	def status_string(self):
		status = f"Have {self.num_bad_epochs}/{self.patience} patience epochs"
		if self.in_cooldown:
			status += f", {self.cooldown_counter} cooldown epochs left"
		if self.in_grace:
			status += f", {self.grace_counter} grace epochs left to beat {self.grace_value:{loss_fmt}}"
		if not self.threshold_crossed:
			status += f", step-enabling threshold not yet crossed"
		return status

	def event_string(self):
		return self.event

	def step(self, metrics, metric_threshold=None, epoch=None):

		current_lr = [group['lr'] for group in self.optimizer.param_groups]

		current = float(metrics)
		threshold = float(metric_threshold) if metric_threshold is not None else math.inf
		if current <= self.threshold_scale * threshold:
			self.threshold_crossed = True

		if epoch is None:
			epoch = self.last_epoch + 1
		else:
			warnings.warn(torch.optim.lr_scheduler.EPOCH_DEPRECATION_WARNING, DeprecationWarning)
		self.last_epoch = epoch

		reduce_lr = False
		reduce_reason = ''
		if self.grace_counter > 0:
			self.grace_counter -= 1
		if self.grace_value is not None and (self.is_better(current, self.grace_value) or self.grace_counter <= 0):
			reduce_lr = True
			reduce_reason = f' with grace {self.grace_counter} left' if self.grace_counter > 0 else ' with grace exceeded'

		if self.is_better(current, self.best):
			self.best = current
			self.num_bad_epochs = 0
		else:
			self.num_bad_epochs += 1

		if self.in_cooldown:
			self.cooldown_counter -= 1
			self.num_bad_epochs = 0

		if not self.threshold_crossed:
			self.num_bad_epochs = 0

		if self.num_bad_epochs >= self.patience and self.grace_value is None:
			if not self.cycbuf:
				grace_value = self.mode_worse
			elif self.mode == 'min':
				grace_value = min(self.cycbuf)
			else:
				grace_value = max(self.cycbuf)
			if self.is_better(current, grace_value):
				reduce_lr = True
				reduce_reason = f' immediately'
			else:
				self.grace_value = grace_value
				self.grace_counter = self.grace_test

		if reduce_lr:
			self._reduce_lr(epoch)
			self.cooldown_counter = self.cooldown
			self.num_bad_epochs = 0
			self.grace_value = None
			self.grace_counter = 0

		self.cycbuf.appendleft(current)

		self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

		self.event = f"Reduced learning rate {lr_string(current_lr, brackets=True)} --> {lr_string(self._last_lr, brackets=True)} at metric {current:{loss_fmt}} (best {self.best:{loss_fmt}}){reduce_reason}" if reduce_lr else None

#
# GPU status functions
#

# Reset the peak PyTorch GPU memory status of a CUDA device
def reset_gpu_memory_status_peaks(device, query=False, show=False, file=None):
	# device = CUDA device to reset the GPU memory status peaks of
	# query = Whether to query the latest GPU memory status right before resetting the peaks
	# show = Whether to show the latest GPU memory status right before resetting the peaks (implies query = True)
	# file = File to which to print the GPU memory status (if show is True)
	# Return the queried GPU memory status, if applicable (None otherwise)
	gpu_mem_status = None
	if device.type == 'cuda':
		if query or show:
			if show:
				gpu_mem_status = show_gpu_memory_status(device, file=file)
			else:
				gpu_mem_status = get_gpu_memory_status(device)
		# noinspection PyUnresolvedReferences
		torch.cuda.reset_peak_memory_stats(device=device)
	return gpu_mem_status

# Get the current PyTorch GPU memory status of a CUDA device
def get_gpu_memory_status(device):
	# device = CUDA device to get the GPU memory status of
	# Return the queried GPU memory status as a dict (None if not using GPU)
	if device.type == 'cuda':
		# noinspection PyUnresolvedReferences
		stats = torch.cuda.memory_stats(device=device)
		alloc_cur_mib = round(stats['allocated_bytes.all.current'] / 1048576)
		alloc_peak_mib = round(stats['allocated_bytes.all.peak'] / 1048576)
		reserved_cur_mib = round(stats['reserved_bytes.all.current'] / 1048576)
		reserved_peak_mib = round(stats['reserved_bytes.all.peak'] / 1048576)
		return {
			'alloc': {'cur': alloc_cur_mib, 'peak': alloc_peak_mib},
			'reserved': {'cur': reserved_cur_mib, 'peak': reserved_peak_mib},
		}
	return None

# Print the current PyTorch GPU memory status of a CUDA device
def show_gpu_memory_status(device, file=None):
	# device = CUDA device to show the GPU memory status of
	# file = File to which to print the GPU memory status
	# Return the queried GPU memory status as a dict (None if not using GPU)
	gpu_mem_status = get_gpu_memory_status(device)
	if gpu_mem_status:
		alloc = gpu_mem_status['alloc']
		reserved = gpu_mem_status['reserved']
		print(f"GPU memory: Tensors {alloc['cur']}MiB (peak {alloc['peak']}MiB), Reserved {reserved['cur']}MiB (peak {reserved['peak']}MiB)", file=file)
	return gpu_mem_status

# Print the current NVIDIA SMI status of a CUDA device
def show_nvidia_smi_status(device, nvsmi, file=None):
	# device = Device to show the NVIDIA SMI status for
	# nvsmi = A util.nvsmi.NvidiaSMI object (governs the lifetime of individual guaranteed access to the NVML library)
	# file = File to which to print the NVIDIA SMI status
	# Return the retrieved Nvidia SMI information as a dict (or None if information retrieval failed in some way)

	if device.type == 'cuda':
		try:
			pci_bus_id = device_util.get_device_pci_bus_id_drv(device)
			nvsmi_data = nvsmi.DeviceQuery('pci.bus_id, memory.used, memory.total, utilization.gpu, utilization.memory, pstate, temperature.gpu, compute-apps')
			data = next((gpu_data for gpu_data in nvsmi_data['gpu'] if gpu_data['id'].endswith(pci_bus_id) or pci_bus_id.endswith(gpu_data['id'])), None)
			if data is None:
				print(f"Nvidia SMI: Failed to find PCI bus ID '{pci_bus_id}' in list of CUDA devices returned by NVML library", file=file)
			else:

				memory = data['fb_memory_usage']
				temp = data['temperature']
				utilise = data['utilization']
				processes = data['processes']

				smi_info: Dict[str, Any] = {'mem_us': None}
				our_memory = ''
				if processes:
					our_pid = os.getpid()
					our_proc = [proc for proc in processes if proc['pid'] == our_pid]
					if our_proc:
						proc_memory = sum(proc['used_memory'] for proc in our_proc)
						other_memory = max(memory['used'] - proc_memory, 0)
						smi_info['mem_us'] = round(proc_memory)
						our_memory = f" ({smi_info['mem_us']} us, {round(other_memory)} other)"

				try:
					smi_info['pstate'] = int(data['performance_state'][1:])
				except ValueError:
					smi_info['pstate'] = None

				smi_info['temp'] = round(temp['gpu_temp'])
				smi_info['temp_slow'] = round(temp['gpu_temp_slow_threshold'])
				smi_info['temp_crit'] = round(temp['gpu_temp_max_threshold'])
				smi_info['temp_unit'] = f"\xB0{temp['unit']}"
				smi_info['gpu_util'] = round(utilise['gpu_util'])
				smi_info['mem_util'] = round(utilise['memory_util'])
				smi_info['mem_used'] = round(memory['used'])
				smi_info['mem_free'] = round(memory['total'] - memory['used'])
				smi_info['mem_total'] = round(memory['total'])

				print(f"Nvidia SMI: Temp {smi_info['temp']}/{smi_info['temp_crit']}{smi_info['temp_unit']} (P{'?' if smi_info['pstate'] is None else smi_info['pstate']}), Memory {smi_info['mem_used']}/{smi_info['mem_total']}{memory['unit']}{our_memory}, Mem util {smi_info['mem_util']}{utilise['unit']}, GPU util {smi_info['gpu_util']}{utilise['unit']}", file=file)

				return smi_info

		except (TypeError, LookupError, KeyError, util.nvsmi.NVMLError) as e:
			print(f"Nvidia SMI: Status check of CUDA device failed with {e.__class__.__name__}: {e}", file=file)

	return None

#
# Training utility functions
#

# Evaluate the reference losses of a model and criterion
def eval_reference_losses(device, model, criterion, data_loaders, enabled=True):
	# device = Device to evaluate the model on
	# model = Model to calculate the reference losses for (netmodel.NetModel)
	# criterion = Criterion to use to evaluate the model losses (should match the model)
	# data_loaders = Data loaders to calculate the reference losses for (dataset.DatasetTuple of torch.utils.data.DataLoader)
	# enabled = Whether the reference losses should actually be calculated
	# Return List of Tuple(string describing reference, losses), where losses is a dataset.DatasetTuple of float, a dataset.DatasetTuple of float corresponding to the minimum losses across all reference losses, and the time taken for reference evaluation

	print("Evaluating reference losses:")
	if enabled:
		start_time = timeit.default_timer()
		ref_losses = eval_reference_losses_impl(device, model, criterion, data_loaders)
		ref_time = timeit.default_timer() - start_time
		if ref_losses:
			min_ref_losses = dataset.DatasetTuple(train=min(losses.train for ref_name, losses in ref_losses), valid=min(losses.valid for ref_name, losses in ref_losses), test=min(losses.test for ref_name, losses in ref_losses))
			for ref_name, losses in ref_losses:
				print(f"  {ref_name} (train) = {losses.train:{loss_fmt}}")
				print(f"  {ref_name} (valid) = {losses.valid:{loss_fmt}}")
				print(f"  {ref_name} (test)  = {losses.test:{loss_fmt}}")
			print(f"  Minimum reference losses: Train {min_ref_losses.train:{loss_fmt}}, Valid {min_ref_losses.valid:{loss_fmt}}, Test {min_ref_losses.test:{loss_fmt}}")
		else:
			min_ref_losses = dataset.DatasetTuple(train=math.inf, valid=math.inf, test=math.inf)
			print("  No reference outputs are available for the model in use")
		print(f"  Evaluation took {ref_time:.1f}s")
	else:
		ref_losses = []
		ref_time = 0
		min_ref_losses = dataset.DatasetTuple(train=math.inf, valid=math.inf, test=math.inf)
		print("  Reference loss calculation is disabled")
	print()

	return ref_losses, min_ref_losses, ref_time

# Implementation of evaluating the reference losses of a model and criterion
def eval_reference_losses_impl(device, model, criterion, data_loaders):
	# device = Device to evaluate the model on
	# model = Model to calculate the reference losses for (netmodel.NetModel)
	# criterion = Criterion to use to evaluate the model losses (should match the model)
	# data_loaders = Data loaders to calculate the reference losses for (dataset.DatasetTuple of torch.utils.data.DataLoader)
	# Return List of Tuple(string describing reference, losses), where losses is a dataset.DatasetTuple of float

	model.eval()

	ref_names = model.reference_output_names()
	num_ref = len(ref_names)
	if num_ref < 1:
		return []

	dict_losses = [{} for _ in range(num_ref)]

	for dt, data_loader in data_loaders.items():

		model_loss = [0.0] * num_ref
		num_samples = 0

		with torch.inference_mode():
			for data, target in data_loader:

				num_in_batch = data[0].shape[0]

				data = tuple(d.to(device, non_blocking=data_loader.pin_memory) for d in data)
				target = tuple(t.to(device, non_blocking=data_loader.pin_memory) for t in target)

				ref_outputs = model.reference_outputs(data)
				if len(ref_outputs) != num_ref:
					raise ValueError(f"Model returned {len(ref_outputs)} reference output(s), but previously returned {num_ref} reference output name(s) => Lengths must always match")

				for i in range(num_ref):
					avg_loss = criterion(ref_outputs[i], target).item()
					model_loss[i] += avg_loss * num_in_batch

				num_samples += num_in_batch

		for i in range(num_ref):
			dict_losses[i][dt] = model_loss[i] / num_samples

	return [(name, dataset.DatasetTuple(**dict_loss)) for name, dict_loss in zip(ref_names, dict_losses)]

# Evaluate the loss of a model on a particular dataset
def eval_model_loss(model, criterion, data_loader):
	# model = Model to evaluate on a particular dataset (netmodel.NetModel)
	# criterion = Criterion to use for evaluation
	# data_loader = Data loader for the dataset to evaluate the model on
	# Return mean sample loss, number of samples, number of batches

	model.eval()

	model_loss = 0.0
	num_samples = 0
	num_batches = len(data_loader)

	with torch.inference_mode():
		for data, target in data_loader:

			num_in_batch = data[0].shape[0]

			data = tuple(d.to(model.device, non_blocking=data_loader.pin_memory) for d in data)
			target = tuple(t.to(model.device, non_blocking=data_loader.pin_memory) for t in target)

			output = model(data)
			avg_loss = criterion(output, target).item()

			model_loss += avg_loss * num_in_batch
			num_samples += num_in_batch

	model_loss /= num_samples

	return model_loss, num_samples, num_batches
# EOF
