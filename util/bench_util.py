# PNN Library: Benchmarking utilities

# Imports
import timeit
import torch

# Time torch operations (uses timeit timer)
class TimeTorch:

	def __init__(self, device=None):
		self.device = device if device is not None and device.type == 'cuda' else None
		self.start_time = None

	def start(self):
		if self.device:
			torch.cuda.synchronize(self.device)
		self.start_time = timeit.default_timer()
		return self.start_time

	def stop(self):
		if self.device:
			torch.cuda.synchronize(self.device)
		return timeit.default_timer() - self.start_time

# Time torch operations (uses CUDA events)
class TimeTorchCUDA:

	def __init__(self, device=None):
		self.device = device if device is not None and device.type == 'cuda' else None
		self.start_event = self.stop_event = None

	def start(self):
		if self.device:
			torch.cuda.synchronize(self.device)
		self.start_event = torch.cuda.Event(enable_timing=True)
		self.stop_event = torch.cuda.Event(enable_timing=True)
		self.start_event.record(stream=torch.cuda.current_stream(self.device))

	def stop(self):
		self.stop_event.record(stream=torch.cuda.current_stream(self.device))
		if self.device:
			torch.cuda.synchronize(self.device)
		return self.start_event.elapsed_time(self.stop_event) / 1000
# EOF
