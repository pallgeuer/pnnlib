# PNN Library: Neural network model

# Imports
import inspect
from typing import Dict, Tuple
import torch
import torch.nn as nn
from pnnlib import dataset

# Constants
META_OPTIMISED_PARAMS = 'OptimisedParams'

#
# Classes
#

# Network model class
class NetModel(nn.Module):

	reqd_inputs_raw: dataset.DataSpec
	reqd_targets_raw: dataset.DataSpec
	reqd_outputs_raw: dataset.DataSpec
	reqd_inputs: dataset.DataSpec
	reqd_targets: dataset.DataSpec
	reqd_outputs: dataset.DataSpec
	reqd_input_map: Dict[dataset.ChannelTypeBase, dataset.ChannelIndex]
	reqd_target_map: Dict[dataset.ChannelTypeBase, dataset.ChannelIndex]
	reqd_output_map: Dict[dataset.ChannelTypeBase, dataset.ChannelIndex]
	reqd_input_group_channels: Tuple[int]
	reqd_target_group_channels: Tuple[int]
	reqd_output_group_channels: Tuple[int]

	def __init__(self, C):
		super().__init__()
		self.C = C
		self.device = torch.device('cpu')

		self.reqd_inputs_raw = self.get_required_inputs()
		self.reqd_inputs = dataset.resolve_data_spec_groups(self.reqd_inputs_raw)
		self.reqd_input_map = dataset.generate_channel_map(self.reqd_inputs)
		self.reqd_input_group_channels = tuple(sum(self.reqd_inputs.channels[channel].type.count for channel in group) for group in self.reqd_inputs.groups)

		self.reqd_targets_raw = self.get_required_targets()
		self.reqd_targets = dataset.resolve_data_spec_groups(self.reqd_targets_raw)
		self.reqd_target_map = dataset.generate_channel_map(self.reqd_targets)
		self.reqd_target_group_channels = tuple(sum(self.reqd_targets.channels[channel].type.count for channel in group) for group in self.reqd_targets.groups)

		self.reqd_outputs_raw = self.get_required_outputs()
		self.reqd_outputs = dataset.resolve_data_spec_groups(self.reqd_outputs_raw)
		self.reqd_output_map = dataset.generate_channel_map(self.reqd_outputs)
		self.reqd_output_group_channels = tuple(sum(self.reqd_outputs.channels[channel].type.count for channel in group) for group in self.reqd_outputs.groups)

	def to(self, *args, **kwargs):
		selff = super().to(*args, **kwargs)
		try:
			selff.device = next(selff.parameters()).device
		except StopIteration:
			selff.device = torch.device('cpu')
		return selff

	def get_required_inputs(self):
		# Return a dataset.DataSpec of information specifying the required format and properties of the model inputs
		# This is the data that the model forward function should be called with
		# Beware that this method is called already from the NetModel constructor, so the derived constructor will only have executed until super().__init__() at that point (using self.C is guaranteed to be okay though)
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def get_required_targets(self):
		# Return a dataset.DataSpec of information specifying the required format and properties of the model targets
		# This is the ground truth data that should that should be used to evaluate the model criterion
		# Beware that this method is called already from the NetModel constructor, so the derived constructor will only have executed until super().__init__() at that point (using self.C is guaranteed to be okay though)
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def get_required_outputs(self):
		# Return a dataset.DataSpec of information specifying the required format and properties of the model outputs
		# This is the data that should result from the model forward function, and that should be used to evaluate the model criterion
		# Beware that this method is called already from the NetModel constructor, so the derived constructor will only have executed until super().__init__() at that point (using self.C is guaranteed to be okay though)
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	# noinspection PyMethodMayBeStatic
	def reference_output_names(self):
		# Return an iterable of the string names of the implemented reference outputs
		# Note: The returned order must match between the reference_output_names() and reference_outputs() functions!
		return ()

	# noinspection PyMethodMayBeStatic, PyUnusedLocal
	def reference_outputs(self, input_data):
		# input_data = Model input to return the reference outputs for
		# Return an iterable of reference outputs (each identical in format to what the model would output for the given model input)
		# Note: The returned order must match between the reference_output_names() and reference_outputs() functions!
		return ()

	def get_optimizer(self):
		# Return an optimizer for the model (torch.optim.Optimizer)
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def get_criterion(self):
		# Return a criterion for the model (Callable(sample_outputs, sample_targets) -> Scalar torch tensor)
		# Note: Use get_criterion_moved() instead for a criterion that's already moved to the same device as the model!
		# Note: The criterion must calculate the MEAN loss of the batch, not the SUM!
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def get_criterion_moved(self):
		# Return a criterion for the model that has been moved to the same device as the model (Callable(sample_outputs, sample_targets) -> Scalar torch tensor)
		return self.get_criterion().to(self.device)

	def parameters_to_optimize(self):
		# Return an iterator over the model parameters to optimize
		return self.parameters()  # By default try to optimize all parameters... (see optimizable_parameters() for a possible alternative)

	def optimizable_parameters(self):
		# Return an iterator over the optimizable model parameters (parameters with requires_grad = True)
		return filter(lambda p: p.requires_grad, self.parameters())

	def annotate_output(self, channel, value, ungrouped_data, derived_data, dset=None):
		# channel = Channel that should be annotated (dataset.ChannelSpec)
		# value = Value of the specified channel to modify in-place (will generally be a copy or modified/converted version of the corresponding ungrouped value in ungrouped_data, or of corresponding transformed/tensorized values)
		# ungrouped_data = All ungrouped data for the sample (source data for the annotation)
		# derived_data = All derived data for the sample (source data for the annotation)
		# dset = Dataset that this request originated from (None => self)
		pass

	def ungroup_output_data(self, output_data, sample_index):
		# output_data = Output data from the model in the required output data specification format
		# sample_index = Index/slice to select for ungrouping (None => No batch dimension, first dimension is already the channels dimension, Negative int => Everything)
		# Return the ungrouped output data in the format Dict[ChannelSpec (flattened), Tensor]
		return dataset.ungroup_data(self.reqd_outputs, output_data, sample_index)

	# noinspection PyMethodMayBeStatic, PyUnusedLocal
	def derive_output_data(self, ungrouped_data, perf_params=None):
		# ungrouped_data = Ungrouped data (must correspond to the data for a SINGLE sample)
		# perf_params = Performance evaluation parameters to use for deriving output data
		# Return the derived output data in the format Dict[ChannelSpec (flattened), Any]
		return {}

	def get_performance_params(self):
		# Return a modifiable dict of the expected performance parameters and their default values (not including META_OPTIMISED_PARAMS)
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def resolve_performance_params(self, perf_params):
		# perf_params = Explicit performance parameters to resolve (or None for default)
		# Return a clean dict of exactly the performance parameters and their values (including META_OPTIMISED_PARAMS)
		params = self.get_performance_params()
		if perf_params is not None and params.keys() <= perf_params.keys():
			params[META_OPTIMISED_PARAMS] = False
			params.update(item for item in perf_params.items() if item[0] in params)
		else:
			params[META_OPTIMISED_PARAMS] = False
		return params

	def evaluate_performance(self, data_loader, perf_params=None, optimise_params=False):
		# data_loader = Dataset to evaluate the model performance on (or tuple of multiple datasets, each is torch.utils.data.DataLoader or list of batchified get_sample_grouped() outputs)
		# perf_params = Explicit performance evaluation parameters to use, or start optimisation with (None or Dict[str, Any] that must contain (at minimum) all parameters except possibly META_OPTIMISED_PARAMS)
		# optimise_params = Whether to optimise the performance evaluation parameters in order to achieve the best possible result
		# Return scalar float model performance (higher is better), model performance details Dict[str, float], used/optimised performance evaluation parameters (Dict[str, Any] of exactly only the performance parameters, including META_OPTIMISED_PARAMS)
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")
# EOF
