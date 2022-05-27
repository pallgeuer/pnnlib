# PNN Library: Dataset classes

# Standard library imports
import os
import re
import random
import inspect
import itertools
from enum import Enum, Flag, auto
from collections import namedtuple, Counter
from typing import Callable, Optional, NamedTuple, Tuple, Sequence, Union, Dict, Any

# Third-party imports
import PIL.Image
import torch.utils.data

# Local imports
from util.classes import EnumFI
from util.contextman import ReentrantMeta

####################
### Helper types ###
####################

# Dataset type enumeration
class DatasetType(Enum):
	Train = auto()  # Training dataset
	Valid = auto()  # Validation dataset
	Test = auto()   # Test dataset

# Dataset type tuple base class
__DatasetTuple = namedtuple('DatasetTuple', 'train valid test')  # Format: Arbitrary but should be the same for each field

# Dataset type tuple class
class DatasetTuple(__DatasetTuple):

	@classmethod
	def keys(cls):
		return cls._fields

	def items(self):
		return self._asdict().items()

	def get(self, dataset_type):
		if dataset_type == DatasetType.Train:
			return self.train
		elif dataset_type == DatasetType.Valid:
			return self.valid
		elif dataset_type == DatasetType.Test:
			return self.test
		else:
			raise ValueError(f"Unrecognised dataset type value (should be of type DatasetType enum): {dataset_type}")

# Load stage enumeration
class LoadStage(Enum):
	LoadRaw = auto()
	Transform = auto()
	Tensorize = auto()

# Load stage tuple base class
__LoadStageTuple = namedtuple('LoadStageTuple', 'raw tfrm tensor')  # Format: Arbitrary but should be the same for each field

# Load stage tuple class
class LoadStageTuple(__LoadStageTuple):

	@classmethod
	def keys(cls):
		return cls._fields

	def items(self):
		return self._asdict().items()

	def get(self, load_stage):
		if load_stage == LoadStage.LoadRaw:
			return self.raw
		elif load_stage == LoadStage.Transform:
			return self.tfrm
		elif load_stage == LoadStage.Tensorize:
			return self.tensor
		else:
			raise ValueError(f"Unrecognised load stage value (should be of type LoadStage enum): {load_stage}")

# Resource type enumeration base
class ResourceTypeBase(namedtuple('ResourceType', 'value ext required'), EnumFI):
	pass

# Channel type enumeration base
class ChannelTypeBase(namedtuple('ChannelType', 'value count tfrm_raw_deps tfrm_tfrm_deps tensor_tfrm_deps tensor_tensor_deps tensor_range'), EnumFI):

	count: int

	def __new__(cls, *args, **kwargs):
		subbed_args = list(args[:3])
		for arg in args[3:]:
			if not arg:
				subbed_args.append(arg)
			else:
				subbed_arg = []
				for value_tuple in arg:
					if isinstance(value_tuple, tuple) and not isinstance(value_tuple, Enum):
						value = value_tuple[0]
						if isinstance(value, auto):
							value = value.value
						subbed_arg.append(cls._value2member_map_[value])
					else:
						subbed_arg.append(value_tuple)
				subbed_args.append(tuple(subbed_arg))
		return super().__new_member__(cls, *subbed_args, **kwargs)

# Channel options enumeration base
class ChannelOptBase(namedtuple('ChannelOpt', 'value stage'), EnumFI):
	pass

# Channel specification class
class ChannelSpec(NamedTuple):
	type: ChannelTypeBase
	opts: Union[Dict[ChannelOptBase, Any], Tuple]

# Data specification class
class DataSpec(NamedTuple):
	channels: Sequence[ChannelSpec]
	groups: Union[bool, Sequence[Tuple[int]]]  # False (no groups), True (all in one group), or sequence of Tuple[int] specifying channel indices to group together

# Channel index class
class ChannelIndex(NamedTuple):
	group: int      # First group index in which the channel occurs regardless of options
	channel: int    # Channel index within the group
	slice: int      # Slice start index within the group
	slice_end: int  # Slice end index within the group (1 greater than the last index that is actually part of the channel)
	count: int      # Number of slices that the given channel takes up

# Include-exclude specification tuple
IncludeExclude = namedtuple('IncludeExclude', 'include exclude')  # Format: None or Set (falsey => include all found), None or Set (falsey => exclude none)
INCLUDE_EXCLUDE_NONE = IncludeExclude(None, None)

# Input/target kind flag
class IOKind(Flag):
	NoKind = 0
	Input = auto()
	Target = auto()
	InputDep = auto()
	TargetDep = auto()

# Agenda item tuple classes
RawAgendaItem = namedtuple('RawAgendaItem', 'resource kind opts func')  # Format: ChannelSpec[Enum, Tuple], IOKind, Dict[Enum, Any], Callable
TfrmAgendaItem = namedtuple('TfrmAgendaItem', 'channel kind opts raw_deps tfrm_deps func')  # Format: ChannelSpec[Enum, Tuple], IOKind, Dict[Enum, Any], Tuple[ChannelSpec[Enum, Tuple]], Tuple[ChannelSpec[Enum, Tuple]], Callable
TensorizeAgendaItem = namedtuple('TensorizeAgendaItem', 'channel kind opts tfrm_deps tensor_deps func')  # Format: ChannelSpec[Enum, Tuple], IOKind, Dict[Enum, Any], Tuple[ChannelSpec[Enum, Tuple]], Tuple[ChannelSpec[Enum, Tuple]], Callable

###########################
### Data spec functions ###
###########################

# Resolve the grouping of a data specification to a standardised List of Tuple[int], where every channel occurs in at least one group, and there is at least one channel
def resolve_data_spec_groups(data_spec):

	if data_spec.groups is False:
		actual_groups = [(i,) for i in range(len(data_spec.channels))]
	elif data_spec.groups is True:
		actual_groups = [tuple(range(len(data_spec.channels)))]
	else:
		missing_indices = set(range(len(data_spec.channels)))
		for group in data_spec.groups:
			missing_indices -= set(group)
		actual_groups = list(tuple(group) for group in data_spec.groups)
		actual_groups.extend((i,) for i in sorted(missing_indices))

	resolved_spec = DataSpec(channels=data_spec.channels, groups=actual_groups)
	check_data_spec_resolved(resolved_spec)
	return resolved_spec

# Check a data specification against the strict channel and resolved groups conditions
def check_data_spec_resolved(data_spec, exception_if_bad=True):

	issues = []

	if not isinstance(data_spec, DataSpec):
		issues.append(f"Object is not of type DataSpec: {data_spec}")
	else:

		if not data_spec.channels:
			issues.append("There are no channels => Need at least one")

		for c, channel in enumerate(data_spec.channels):
			if not isinstance(channel, ChannelSpec):
				issues.append(f"Channel {c} is not of type ChannelSpec: {channel}")
			else:
				if not isinstance(channel.type, ChannelTypeBase):
					issues.append(f"Channel {c} is not of type ChannelTypeBase: {channel.type}")
				elif channel.type.tfrm_raw_deps is not None and not all(isinstance(dep, ResourceTypeBase) for dep in channel.type.tfrm_raw_deps):
					issues.append(f"Channel {c} has raw-to-transformed dependencies that are not of type ResourceTypeBase: {channel.type.tfrm_raw_deps}")
				elif channel.type.tfrm_tfrm_deps is not None and not all(isinstance(dep, ChannelTypeBase) for dep in channel.type.tfrm_tfrm_deps):
					issues.append(f"Channel {c} has transformed-to-transformed dependencies that are not of type ChannelTypeBase: {channel.type.tfrm_tfrm_deps}")
				elif channel.type.tensor_tfrm_deps is not None and not all(dep is None or isinstance(dep, ChannelTypeBase) for dep in channel.type.tensor_tfrm_deps):
					issues.append(f"Channel {c} has transformed-to-tensor dependencies that are not of type ChannelTypeBase: {channel.type.tensor_tfrm_deps}")
				elif channel.type.tensor_tensor_deps is not None and not all(isinstance(dep, ChannelTypeBase) for dep in channel.type.tensor_tensor_deps):
					issues.append(f"Channel {c} has tensor-to-tensor dependencies that are not of type ChannelTypeBase: {channel.type.tensor_tensor_deps}")
				if not isinstance(channel.opts, dict):
					issues.append(f"Channel {c} options is not a dict: {channel.opts}")
				elif not all(isinstance(opt, ChannelOptBase) for opt in channel.opts):
					issues.append(f"Channel {c} has options that are not of type ChannelOptBase: {channel.opts}")

		if not isinstance(data_spec.groups, list):
			issues.append(f"Resolved groups specification is not a list: {data_spec.groups}")
		elif not data_spec.groups:
			issues.append(f"Resolved groups specification is empty")
		else:
			groups_union = set()
			for g, group in enumerate(data_spec.groups):
				if not isinstance(group, tuple):
					issues.append(f"Group {g} is not a tuple: {group}")
				elif not group:
					issues.append(f"Group {g} is empty")
				else:
					if not all(isinstance(c, int) for c in group):
						issues.append(f"Group {g} has non-int entries: {group}")
					groups_union |= set(group)
			if groups_union != set(range(len(data_spec.channels))):
				issues.append(f"Group specifications do not include all channels, or include non-existent channels: {groups_union}")

	if issues and exception_if_bad:
		raise ValueError("DataSpec does not meet the strict channel and resolved groups conditions:\n - " + '\n - '.join(issues))

	return issues

# Generate a channel map from a resolved data specification
def generate_channel_map(data_spec):
	# data_spec = Resolved data specification
	# Return a Dict[dataset.ChannelTypeBase, dataset.ChannelIndex] that maps every specified channel to its location in the final group
	channel_map = {}
	for g, group in enumerate(data_spec.groups):
		slice_index = 0
		for c, channel in enumerate(group):
			channel_type = data_spec.channels[channel].type
			if channel_type not in channel_map:
				channel_map[channel_type] = ChannelIndex(group=g, channel=c, slice=slice_index, slice_end=slice_index + channel_type.count, count=channel_type.count)
			slice_index += channel_type.count
	return channel_map

# Ungroup data based on a resolved data specification
def ungroup_data(data_spec, data, index):
	# data_spec = Resolved data specification
	# data = Data that follows the given data specification
	# index = Index/slice to select for the first dimension of each encountered tensor (None => First dimension is already the channel dimension, Negative int => Select everything)
	# Return the ungrouped data in the format Dict[ChannelSpec (flattened), Tensor]
	ungrouped_data = {}
	for g, group_indices in enumerate(data_spec.groups):
		group_data = data[g]
		slice_index = 0
		for channel_index in group_indices:
			channel_spec = data_spec.channels[channel_index]
			channel_slice = slice(slice_index, slice_index + channel_spec.type.count)
			slice_index += channel_spec.type.count
			if index is None:
				channel_data = group_data[channel_slice, ...]
			elif index < 0:
				channel_data = group_data[:, channel_slice, ...]
			else:
				channel_data = group_data[index, channel_slice, ...]
			channel_spec_flat = ChannelSpec(channel_spec.type, tuple(sorted(channel_spec.opts.items())))
			ungrouped_data[channel_spec_flat] = channel_data
	return ungrouped_data

######################
### Staged dataset ###
######################

# Staged dataset class
class StagedDataset(torch.utils.data.Dataset):

	#
	# Construction
	#

	def __init__(self, root_dir, dataset_type, reqd_inputs, reqd_targets, limit_to=None, limit_hard=False, limit_sorted=False, limit_seed=0, load_kwargs=None):
		# root_dir = Dataset root directory path (ROOT)
		# dataset_type = Type of dataset (DatasetType enum = Train, Valid, Test)
		# reqd_inputs = Required model inputs (expects resolved dataset.DataSpec)
		# reqd_targets = Required model targets (expects resolved dataset.DataSpec)
		# limit_to = If a positive integer is provided, limit the dataset by allowing only limit_to samples (see the remaining limit_* arguments for options as to how this is to be implemented)
		# limit_hard = If true, truncate the dataset to limit_to samples, otherwise repeat samples as often as necessary to still have the same original dataset size
		# limit_sorted = Whether to preserve sample order when limiting the dataset
		# limit_seed = Integer seed to use for the deterministic selection of samples to limit to
		# load_kwargs = Extra keyword arguments to pass to the load_samples method

		self.root = os.path.expanduser(root_dir)
		if not os.path.isdir(self.root):
			raise NotADirectoryError(f"Specified root path is not a valid directory: {self.root}")
		self.dataset_type = dataset_type
		self.reqd_inputs = reqd_inputs
		self.reqd_targets = reqd_targets

		self.agenda, self.input_group_specs, self.target_group_specs, reqd_res_types = self.process_reqd_channels(self.reqd_inputs, self.reqd_targets, LoadStageTuple(raw=self.get_raw_func, tfrm=self.get_tfrm_func, tensor=self.get_tensorize_func))
		self.target_tensor_types = {item.channel.type for item in self.agenda.tensor if IOKind.Target in item.kind}
		self.samples, self.load_details = self.load_samples(reqd_res_types, **(load_kwargs or {}))
		self.total_samples = len(self.samples)
		self._extra_data_indices = None

		self.limit_to = min(int(limit_to), self.total_samples) if limit_to and limit_to >= 1 else 0
		self.limit_hard = limit_hard
		self.limit_sorted = limit_sorted
		self.limit_random = random.Random(limit_seed)
		if self.limit_to:
			self.limited_indices = self.limit_random.sample(range(self.total_samples), self.limit_to)
			if self.limit_sorted:
				self.limited_indices.sort()
		else:
			self.limited_indices = None

		self.sample_size = self.limit_to if self.limit_to and self.limit_hard else self.total_samples
		self.init_dataset(self)

	@staticmethod
	def process_reqd_channels(reqd_inputs: DataSpec, reqd_targets, get_funcs):
		# reqd_inputs = Required model inputs (expects resolved dataset.DataSpec)
		# reqd_targets = Required model targets (expects resolved dataset.DataSpec)
		# get_funcs = LoadStageTuple of Callables used to get the required stage functions
		# Return LoadStageTuple agenda, input groups specification, target groups specification, set of required resource types

		input_group_specs = tuple(tuple(ChannelSpec(reqd_inputs.channels[index].type, tuple(sorted(reqd_inputs.channels[index].opts.items()))) for index in group) for group in reqd_inputs.groups)
		target_group_specs = tuple(tuple(ChannelSpec(reqd_targets.channels[index].type, tuple(sorted(reqd_targets.channels[index].opts.items()))) for index in group) for group in reqd_targets.groups)

		input_tensor_channels, input_dep_tensor_channels = StagedDataset._resolve_tensor_channels(input_group_specs)
		target_tensor_channels, target_dep_tensor_channels = StagedDataset._resolve_tensor_channels(target_group_specs)
		reqd_tensor_channels = input_dep_tensor_channels | target_dep_tensor_channels

		tensorize_agenda = []
		tensorized_channels = set()
		tfrm_channel_kind = {}
		while reqd_tensor_channels:
			num_left = len(reqd_tensor_channels)
			for channel in sorted(reqd_tensor_channels):
				if channel.type.tensor_tfrm_deps is None or channel.type.tensor_tensor_deps is None:
					raise ValueError(f"A specified channel type is invalid as a tensorized value: {channel.type}")
				tfrm_opts = tuple(opt for opt in channel.opts if opt[0].stage.value <= LoadStage.Transform.value)
				tensor_deps = tuple(ChannelSpec(dep, channel.opts) for dep in channel.type.tensor_tensor_deps)
				if all(dep in tensorized_channels for dep in tensor_deps):
					reqd_tensor_channels.remove(channel)
					tensorized_channels.add(channel)
					tfrm_deps = tuple(ChannelSpec(dep or channel.type, tfrm_opts) for dep in channel.type.tensor_tfrm_deps)
					kind_dep = (IOKind.InputDep if channel in input_dep_tensor_channels else IOKind.NoKind) | (IOKind.TargetDep if channel in target_dep_tensor_channels else IOKind.NoKind)
					kind = (IOKind.Input if channel in input_tensor_channels else IOKind.NoKind) | (IOKind.Target if channel in target_tensor_channels else IOKind.NoKind) | kind_dep
					tensorize_agenda.append(TensorizeAgendaItem(channel=channel, kind=kind, opts=dict(channel.opts), tfrm_deps=tfrm_deps, tensor_deps=tensor_deps, func=get_funcs.tensor(channel.type)))
					for dep_channel in tfrm_deps:
						tfrm_channel_kind[dep_channel] = tfrm_channel_kind.get(dep_channel, IOKind.NoKind) | kind_dep
			if len(reqd_tensor_channels) == num_left:
				raise ValueError("There is some cycle or bad value(s) in the tensor-to-tensor dependencies")
		if not tensorize_agenda:
			raise ValueError("Should have at least one tensorize agenda step")

		reqd_tfrm_channels = set()
		tfrm_channels_to_process = set(tfrm_channel_kind.keys())
		while tfrm_channels_to_process:
			tfrm_channel = tfrm_channels_to_process.pop()
			if tfrm_channel not in reqd_tfrm_channels:
				reqd_tfrm_channels.add(tfrm_channel)
				if tfrm_channel.type.tfrm_tfrm_deps:
					for channel_type in tfrm_channel.type.tfrm_tfrm_deps:
						dep_channel = ChannelSpec(channel_type, tfrm_channel.opts)
						tfrm_channels_to_process.add(dep_channel)
						tfrm_channel_kind[dep_channel] = tfrm_channel_kind.get(dep_channel, IOKind.NoKind) | tfrm_channel_kind[tfrm_channel]

		tfrm_agenda = []
		tfrmed_channels = set()
		raw_resource_kind = {}
		while reqd_tfrm_channels:
			num_left = len(reqd_tfrm_channels)
			for channel in sorted(reqd_tfrm_channels):
				if channel.type.tfrm_raw_deps is None or channel.type.tfrm_tfrm_deps is None:
					raise ValueError(f"A specified channel type is invalid as a transformed value: {channel.type}")
				raw_opts = tuple(opt for opt in channel.opts if opt[0].stage.value <= LoadStage.LoadRaw.value)
				tfrm_deps = tuple(ChannelSpec(dep, channel.opts) for dep in channel.type.tfrm_tfrm_deps)
				if all(dep in tfrmed_channels for dep in tfrm_deps):
					reqd_tfrm_channels.remove(channel)
					tfrmed_channels.add(channel)
					raw_deps = tuple(ChannelSpec(dep, raw_opts) for dep in channel.type.tfrm_raw_deps)
					kind = tfrm_channel_kind[channel]
					tfrm_agenda.append(TfrmAgendaItem(channel=channel, kind=kind, opts=dict(channel.opts), raw_deps=raw_deps, tfrm_deps=tfrm_deps, func=get_funcs.tfrm(channel.type)))
					for dep_resource in raw_deps:
						raw_resource_kind[dep_resource] = raw_resource_kind.get(dep_resource, IOKind.NoKind) | kind
			if len(reqd_tfrm_channels) == num_left:
				raise ValueError("There is some cycle or bad value(s) in the transformed-to-transformed dependencies")
		if not tfrm_agenda:
			raise ValueError("Should have at least one transform agenda step")

		raw_agenda = []
		for resource, kind in sorted(raw_resource_kind.items()):
			raw_agenda.append(RawAgendaItem(resource=resource, kind=kind, opts=dict(resource.opts), func=get_funcs.raw(resource.type)))
		if not raw_agenda:
			raise ValueError("Should have at least one raw agenda step")

		reqd_res_types = set(resource.type for resource in raw_resource_kind.keys())
		if not reqd_res_types:
			raise ValueError("Should have at least one required resource type")

		return LoadStageTuple(raw_agenda, tfrm_agenda, tensorize_agenda), input_group_specs, target_group_specs, reqd_res_types

	@staticmethod
	def _resolve_tensor_channels(group_specs):
		tensor_channels = {channel for channel_tuple in group_specs for channel in channel_tuple}
		tensor_channels_to_process = tensor_channels.copy()
		dep_tensor_channels = set()
		while tensor_channels_to_process:
			tensor_channel = tensor_channels_to_process.pop()
			if tensor_channel not in dep_tensor_channels:
				dep_tensor_channels.add(tensor_channel)
				if tensor_channel.type.tensor_tensor_deps:
					for channel_type in tensor_channel.type.tensor_tensor_deps:
						tensor_channels_to_process.add(ChannelSpec(channel_type, tensor_channel.opts))
		return tensor_channels, dep_tensor_channels

	def get_agenda(self, stage=None):
		if stage is None:
			return self.agenda
		else:
			return self.agenda.get(stage)

	def load_samples(self, reqd_res_types, **kwargs):
		# reqd_res_types = Set of required resource types
		# kwargs = Further required keyword arguments
		# Return a deterministically sorted list of the samples, where each entry in the list is Tuple[sample_key, Dict[res_type, path]], and a list of string details about the loaded samples
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def get_load_details(self):
		return self.load_details

	#
	# Get samples
	#

	def __getitem__(self, index):
		# index = Index of the sample to retrieve
		# Return same as get_sample() for the required sample index
		return self.get_sample(index=index)

	def __len__(self):
		# Return number of samples
		return self.sample_size

	def rand_index(self):
		# Return a valid random sample index
		return random.randrange(self.sample_size)

	def resolve_index(self, index):
		# index = Sample index to wrap to valid range, or if None choose a valid random index
		# Return an index guaranteed to be valid
		if index is None:
			return self.rand_index()
		elif isinstance(index, int):
			return index % self.sample_size
		else:
			return self.resolve_key(index)

	def resolve_key(self, sample_key):
		# sample_key = Sample key to search for in the samples list (raises an exception if not found)
		try:
			return next(index for index, entry in enumerate(self.samples) if entry[0] == sample_key)
		except StopIteration:
			raise KeyError(f"Sample key not found: {sample_key}")

	def get_sample_entry(self, index, dset=None):
		# index = Index of the sample entry to retrieve
		# dset = Dataset that this request originated from
		# Return the sample entry in the format Tuple[sample_key, Dict[res_type, path]]
		dset_is_self = dset is None or dset is self
		if index is None or index < 0 or index >= self.total_samples or (dset_is_self and index >= self.sample_size):
			raise IndexError()
		elif dset_is_self and self.limited_indices:
			index = self.limited_indices[index % self.limit_to]
		return self.samples[index]

	def get_sample_raw(self, index=None, sample_entry=None, dset=None, **kwargs):
		# index = Index of the raw sample to retrieve
		# sample_entry = Sample entry to load the raw data for
		# dset = Dataset that this request originated from
		# kwargs = Keyword arguments to pass up the chain of get methods
		# Return same as get_sample_entry(), raw data as Dict[resource, value]
		if dset is None:
			dset = self
		if sample_entry is None:
			# noinspection PyArgumentList
			sample_entry = self.get_sample_entry(index=index, dset=dset, **kwargs)
		self.prepare_raw_funcs(dset, sample_entry)
		sample_key, res_path_dict = sample_entry
		raw_data = {item.resource: item.func(dset=dset, key=sample_key, path=res_path_dict.get(item.resource.type), opts=item.opts) for item in self.agenda.raw}
		return sample_entry, raw_data

	def get_sample_tfrmed(self, index=None, sample_raw=None, dset=None, **kwargs):
		# index = Index of the transformed sample to retrieve
		# sample_raw = Raw sample to transform
		# dset = Dataset that this request originated from
		# kwargs = Keyword arguments to pass up the chain of get methods (e.g. sample_entry)
		# Return same as get_sample_raw(), transformed data as Dict[channel, value]
		if dset is None:
			dset = self
		if sample_raw is None:
			sample_raw = self.get_sample_raw(index=index, dset=dset, **kwargs)
		self.prepare_tfrm_funcs(dset, sample_raw)
		raw_data = sample_raw[-1]
		tfrmed_data = {}
		for item in self.agenda.tfrm:
			tfrmed_data[item.channel] = item.func(dset=dset, raw_data={resource.type: raw_data[resource] for resource in item.raw_deps}, tfrmed_data={channel.type: tfrmed_data[channel] for channel in item.tfrm_deps}, opts=item.opts)
		return *sample_raw, tfrmed_data

	def get_sample_tensor(self, index=None, sample_tfrmed=None, dset=None, **kwargs):
		# index = Index of the tensorized sample to retrieve
		# sample_tfrmed = Transformed sample to tensorize
		# dset = Dataset that this request originated from
		# kwargs = Keyword arguments to pass up the chain of get methods (e.g. sample_raw)
		# Return same as get_sample_tfrmed(), tensor data as Dict[channel, value]
		if dset is None:
			dset = self
		if sample_tfrmed is None:
			sample_tfrmed = self.get_sample_tfrmed(index=index, dset=dset, **kwargs)
		self.prepare_tensorize_funcs(dset, sample_tfrmed)
		tfrmed_data = sample_tfrmed[-1]
		tensor_data = {}
		for item in self.agenda.tensor:
			tensor_data[item.channel] = item.func(dset=dset, tfrmed_data={channel.type: tfrmed_data[channel] for channel in item.tfrm_deps}, tensor_data={channel.type: tensor_data[channel] for channel in item.tensor_deps}, opts=item.opts)
		return *sample_tfrmed, tensor_data

	def get_sample_grouped(self, index=None, sample_tensor=None, dset=None, batchify=False, **kwargs):
		# index = Index of the grouped sample to retrieve
		# sample_tensor = Tensorized sample to finalise
		# dset = Dataset that this request originated from
		# batchify = Whether to return sample data as a size 1 batch
		# kwargs = Keyword arguments to pass up the chain of get methods (e.g. sample_tfrmed)
		# Return same as get_sample_tensor(), grouped input data as tuple of tensor values, grouped target data as tuple of tensor values
		if sample_tensor is None:
			sample_tensor = self.get_sample_tensor(index=index, dset=dset, **kwargs)
		tensor_data = sample_tensor[-1]
		input_data = self._group_sample(tensor_data, self.input_group_specs, batchify)
		target_data = self._group_sample(tensor_data, self.target_group_specs, batchify)
		if batchify:
			return *((data,) for data in sample_tensor), input_data, target_data
		else:
			return *sample_tensor, input_data, target_data

	def get_sample(self, index=None, sample_grouped=None, dset=None, **kwargs):
		# index = Index of the sample to retrieve
		# sample_grouped = Grouped sample to return the input/target data for
		# dset = Dataset that this request originated from
		# kwargs = Keyword arguments to pass up the chain of get methods (e.g. sample_tensor, batchify)
		# Return sample input data, sample target data, and the data associated with any configured extra indices
		if sample_grouped is None:
			sample_grouped = self.get_sample_grouped(index=index, dset=dset, **kwargs)
		# noinspection PyProtectedMember
		extra_data_indices = dset._extra_data_indices if dset is not None else self._extra_data_indices
		if extra_data_indices is None:
			return sample_grouped[-2:]
		else:
			return tuple(sample_grouped[index] for index in itertools.chain((-2, -1), extra_data_indices))

	@staticmethod
	def _group_sample(tensor_data, group_specs, batchify):
		# tensor_data = Dict of tensor values
		# group_specs = Tuple[Tuple[ChannelSpec[Enum, Tuple]]]
		# batchify = Whether to return the grouped sample as a size 1 batch
		group_generator = (torch.cat(tuple(tensor_data[channel] for channel in group_spec), dim=0) if len(group_spec) > 1 else tensor_data[group_spec[0]] for group_spec in group_specs)
		return tuple(tensor.unsqueeze(0) for tensor in group_generator) if batchify else tuple(group_generator)

	def get_extra_data_indices(self):
		return self._extra_data_indices

	def set_extra_data_indices(self, extra_data_indices):
		self._extra_data_indices = extra_data_indices

	def extra_data_indices(self, extra_data_indices):
		return self.ExtraDataIndicesCM(self, extra_data_indices)

	class ExtraDataIndicesCM(metaclass=ReentrantMeta):

		def __init__(self, dataset, extra_data_indices):
			self._dataset = dataset
			self._old_extra_data_indices = self._dataset.get_extra_data_indices()
			self._new_extra_data_indices = extra_data_indices

		def __enter__(self):
			self._dataset.set_extra_data_indices(self._new_extra_data_indices)
			return self._dataset

		def __exit__(self, exc_type, exc_val, exc_tb):
			self._dataset.set_extra_data_indices(self._old_extra_data_indices)
			return False

	#
	# Override interface
	#

	def init_dataset(self, dset):
		# dset = Dataset that this request originated from
		pass

	def get_res_type(self, name):
		# name = Name to retrieve the corresponding resource type of
		# Return the corresponding resource type or None if no such resource type exists
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def get_raw_func(self, res_type):
		# res_type = Desired resource type to load
		# Return a raw loader callable to be used like raw_func(dset=StagedDataset, key=Any, path=str, opts=Dict[Enum]) that returns the raw loaded data for the given resource type
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def get_tfrm_func(self, channel_type):
		# channel_type = Desired channel type to transform
		# Return a transform callable to be used like tfrm_func(dset=StagedDataset, raw_data=Dict[Enum], tfrmed_data=Dict[Enum], opts=Dict[Enum]) that returns the transformed data for the given channel type
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def get_tensorize_func(self, channel_type):
		# channel_type = Desired channel type to tensorize
		# Return a tensorize callable to be used like tensorize_func(dset=StagedDataset, tfrmed_data=Dict[Enum], tensor_data=Dict[Enum], opts=Dict[Enum]) that returns the tensor data for the given channel type
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	def prepare_raw_funcs(self, dset, sample_entry):
		# dset = Dataset that this request originated from
		# sample_entry = Sample entry that needs to be loaded by the raw funcs
		pass

	def prepare_tfrm_funcs(self, dset, sample_raw):
		# dset = Dataset that this request originated from
		# sample_raw = Raw sample data that needs to be transformed by the tfrm funcs
		pass

	def prepare_tensorize_funcs(self, dset, sample_tfrmed):
		# dset = Dataset that this request originated from
		# sample_tfrmed = Transformed sample data that needs to be tensorized by the tensorize funcs
		pass

	def annotate_raw(self, resource, value, raw_data, dset=None):
		# resource = Resource that should be annotated (ChannelSpec)
		# value = Value of the specified resource to modify in-place (will generally be a copy or modified/converted version of the corresponding raw value in raw_data)
		# raw_data = All raw data for the sample (source data for the annotation)
		# dset = Dataset that this request originated from (None => self)
		pass

	def annotate_tfrmed(self, channel, value, tfrmed_data, dset=None):
		# channel = Channel that should be annotated (ChannelSpec)
		# value = Value of the specified channel to modify in-place (will generally be a copy or modified/converted version of the corresponding transformed value in tfrmed_data)
		# tfrmed_data = All transformed data for the sample (source data for the annotation)
		# dset = Dataset that this request originated from (None => self)
		pass

	def annotate_tensor(self, channel, value, tfrmed_data, tensor_data, dset=None):
		# channel = Channel that should be annotated (ChannelSpec)
		# value = Value of the specified channel to modify in-place (will generally be a copy or modified/converted version of the corresponding transformed/tensorized value in tfrmed_data/tensor_data)
		# tfrmed_data = All transformed data for the sample (source data for the annotation)
		# tensor_data = All tensorized data for the sample (source data for the annotation)
		# dset = Dataset that this request originated from (None => self)
		pass

# Viewable staged dataset class
# noinspection PyAbstractClass
class ViewableStagedDataset(StagedDataset):

	#
	# Construction
	#

	def __init__(self, *args, partition_limit_to=None, partition_ratios=DatasetTuple(0.7, 0.15, 0.15), partition_seed=0, **kwargs):
		# args = See StagedDataset.__init__()
		# partition_limit_to = DatasetTuple of non-negative sample count limits (see the remaining limit_* arguments for options as to how this is to be implemented)
		# partition_ratios = DatasetTuple of ratios in which to partition the dataset into TVS parts (does not need to sum to 1, but each component should be non-negative)
		# partition_seed = Integer seed to use for the deterministic random partitioning into TVS parts of the dataset
		# kwargs = See StagedDataset.__init__()
		super().__init__(*args, **kwargs)
		self.partition_limit_to = DatasetTuple(
			train=int(partition_limit_to.train) if partition_limit_to and partition_limit_to.train >= 1 else 0,
			valid=int(partition_limit_to.valid) if partition_limit_to and partition_limit_to.valid >= 1 else 0,
			test=int(partition_limit_to.test) if partition_limit_to and partition_limit_to.test >= 1 else 0,
		)
		self.partition_indices, self.partition_sizes = self.partition_samples(partition_ratios, partition_seed)

	def partition_samples(self, partition_ratios, partition_seed):
		# partition_ratios = DatasetTuple of ratios in which to partition the dataset into TVS parts (does not need to sum to 1, but each component should be non-negative)
		# partition_seed = Integer seed to use for the deterministic random partitioning into TVS parts of the dataset
		# Return DatasetTuple of partition indices, DatasetTuple of partition sizes (after limiting)

		if partition_ratios.train < 0 or partition_ratios.valid < 0 or partition_ratios.test < 0:
			raise ValueError("Partition ratios must all be non-negative")

		ratio_sum = partition_ratios.train + partition_ratios.valid + partition_ratios.test
		if ratio_sum <= 0:
			raise ValueError("Partition ratio sum must be positive")

		ratio_valid = partition_ratios.valid / ratio_sum
		ratio_test = partition_ratios.test / ratio_sum

		num_valid = round(ratio_valid * self.total_samples)
		num_test = round(ratio_test * self.total_samples)
		num_train = self.total_samples - num_valid - num_test

		if num_train < 0:
			num_train = 0
			num_test = self.total_samples - num_valid

		assert num_train + num_valid + num_test == self.total_samples and num_train >= 0 and num_valid >= 0 and num_test >= 0

		indices_train = []
		indices_valid = []
		indices_test = []
		num_train_left = num_train
		num_valid_left = num_valid
		num_test_left = num_test

		partition_random = random.Random(partition_seed)
		for index in range(self.total_samples):
			prob = partition_random.random()
			num_samples_left = self.total_samples - index
			if prob < num_train_left / num_samples_left:
				indices_train.append(index)
				num_train_left -= 1
			elif prob < (num_train_left + num_valid_left) / num_samples_left:
				indices_valid.append(index)
				num_valid_left -= 1
			else:
				indices_test.append(index)
				num_test_left -= 1

		assert len(indices_train) == num_train and len(indices_valid) == num_valid and len(indices_test) == num_test

		if self.partition_limit_to.train >= 1:
			indices_train = self.limit_random.sample(indices_train, min(self.partition_limit_to.train, num_train))
			if self.limit_sorted:
				indices_train.sort()
			if self.limit_hard:
				num_train = len(indices_train)

		if self.partition_limit_to.valid >= 1:
			indices_valid = self.limit_random.sample(indices_valid, min(self.partition_limit_to.valid, num_valid))
			if self.limit_sorted:
				indices_valid.sort()
			if self.limit_hard:
				num_valid = len(indices_valid)

		if self.partition_limit_to.test >= 1:
			indices_test = self.limit_random.sample(indices_test, min(self.partition_limit_to.test, num_test))
			if self.limit_sorted:
				indices_test.sort()
			if self.limit_hard:
				num_test = len(indices_test)

		return DatasetTuple(indices_train, indices_valid, indices_test), DatasetTuple(num_train, num_valid, num_test)

	def create_views(self):
		# Return a DatasetTuple of newly created views
		return DatasetTuple(
			self.create_view_by_indices(DatasetType.Train, self.partition_indices.train, sample_size=self.partition_sizes.train),
			self.create_view_by_indices(DatasetType.Valid, self.partition_indices.valid, sample_size=self.partition_sizes.valid),
			self.create_view_by_indices(DatasetType.Test, self.partition_indices.test, sample_size=self.partition_sizes.test)
		)

	def create_view(self, dataset_type):
		# dataset_type = Required view
		return self.create_view_by_indices(dataset_type, self.partition_indices.get(dataset_type), sample_size=self.partition_sizes.get(dataset_type))

	def create_view_by_indices(self, dataset_type, indices, sample_size=None):
		# dataset_type = Required dataset view type (DatasetType enum = Train, Valid, Test)
		# indices = List of indices to include in the view
		# sample_size = Number of indices to truncate or extend the view to
		# Return the newly created view
		dset = StagedDatasetView(self, dataset_type, indices, sample_size=sample_size)
		self.init_dataset(dset)
		return dset

# Staged dataset view class
# noinspection PyAbstractClass
class StagedDatasetView(StagedDataset):

	#
	# Construction
	#

	# noinspection PyMissingConstructor
	def __init__(self, dataset, dataset_type, sample_indices, sample_size=None):
		# dataset = StagedDataset instance this is a view of
		# dataset_type = Required dataset view type (DatasetType enum = Train, Valid, Test)
		# sample_indices = List[int] specifying the required sample indices in the main dataset
		# sample_size = Number of indices to truncate or extend the view to

		self.dataset = dataset

		self.root = self.dataset.root
		self.dataset_type = dataset_type
		self.reqd_inputs = self.dataset.reqd_inputs
		self.reqd_targets = self.dataset.reqd_targets
		self.input_group_specs = self.dataset.input_group_specs
		self.target_group_specs = self.dataset.target_group_specs
		self.agenda = self.dataset.agenda
		self.target_tensor_types = self.dataset.target_tensor_types
		self.load_details = self.dataset.load_details
		self._extra_data_indices = None

		self.sample_indices = sample_indices
		self.sample_size = len(self.sample_indices) if sample_size is None else sample_size
		if self.sample_size < len(self.sample_indices):
			self.sample_indices = self.sample_indices[:self.sample_size]
		self.actual_size = len(self.sample_indices)

	#
	# Get samples
	#

	def resolve_key(self, sample_key):
		try:
			return next(view_index for view_index, main_index in enumerate(self.sample_indices) if self.dataset.samples[main_index][0] == sample_key)
		except StopIteration:
			raise KeyError(f"Sample key not found: {sample_key}")

	def get_sample_entry(self, index, dset=None):
		return self.dataset.get_sample_entry(self._convert_index(index), dset=dset or self)

	def get_sample_raw(self, index=None, sample_entry=None, dset=None, **kwargs):
		return self.dataset.get_sample_raw(index=self._convert_index(index), sample_entry=sample_entry, dset=dset or self, **kwargs)

	def get_sample_tfrmed(self, index=None, sample_raw=None, dset=None, **kwargs):
		return self.dataset.get_sample_tfrmed(index=self._convert_index(index), sample_raw=sample_raw, dset=dset or self, **kwargs)

	def get_sample_tensor(self, index=None, sample_tfrmed=None, dset=None, **kwargs):
		return self.dataset.get_sample_tensor(index=self._convert_index(index), sample_tfrmed=sample_tfrmed, dset=dset or self, **kwargs)

	def get_sample_grouped(self, index=None, sample_tensor=None, dset=None, batchify=False, **kwargs):
		return self.dataset.get_sample_grouped(index=self._convert_index(index), sample_tensor=sample_tensor, dset=dset or self, batchify=batchify, **kwargs)

	def get_sample(self, index=None, sample_grouped=None, dset=None, **kwargs):
		return self.dataset.get_sample(index=self._convert_index(index), sample_grouped=sample_grouped, dset=dset or self, **kwargs)

	def _convert_index(self, index):
		if index is None:
			return None
		elif index < 0 or index >= self.sample_size:
			raise IndexError()
		return self.sample_indices[index % self.actual_size]

	#
	# Override interface
	#

	def init_dataset(self, dset):
		self.dataset.init_dataset(dset)

	def get_res_type(self, name):
		return self.dataset.get_res_type(name)

	def get_raw_func(self, res_type):
		return self.dataset.get_raw_func(res_type)

	def get_tfrm_func(self, channel_type):
		return self.dataset.get_tfrm_func(channel_type)

	def get_tensorize_func(self, channel_type):
		return self.dataset.get_tensorize_func(channel_type)

	def prepare_raw_funcs(self, dset, sample_entry):
		self.dataset.prepare_raw_funcs(dset, sample_entry)

	def prepare_tfrm_funcs(self, dset, sample_raw):
		self.dataset.prepare_tfrm_funcs(dset, sample_raw)

	def prepare_tensorize_funcs(self, dset, sample_tfrmed):
		self.dataset.prepare_tensorize_funcs(dset, sample_tfrmed)

	def annotate_raw(self, resource, value, raw_data, dset=None):
		self.dataset.annotate_raw(resource, value, raw_data, dset=dset or self)

	def annotate_tfrmed(self, channel, value, tfrmed_data, dset=None):
		self.dataset.annotate_tfrmed(channel, value, tfrmed_data, dset=dset or self)

	def annotate_tensor(self, channel, value, tfrmed_data, tensor_data, dset=None):
		self.dataset.annotate_tensor(channel, value, tfrmed_data, tensor_data, dset=dset or self)

####################
### Flat dataset ###
####################

# Flat dataset class
# noinspection PyAbstractClass
class FlatDataset(StagedDataset):

	#
	# Construction
	#

	def __init__(self, root_dir, dataset_type, res_type, reqd_inputs, reqd_targets, limit_to=None, limit_hard=False, limit_sorted=False, limit_seed=0):
		# root_dir = Path of directory to load data from
		# dataset_type = Type of dataset (DatasetType enum = Train, Valid, Test)
		# res_type = Resource type of the files to load from the root directory
		# reqd_inputs = Required model inputs (expects resolved dataset.DataSpec)
		# reqd_targets = Required model targets (expects resolved dataset.DataSpec)
		# limit_to = If a positive integer is provided, limit the dataset by allowing only limit_to samples (see the remaining limit_* arguments for options as to how this is to be implemented)
		# limit_hard = If True, truncate the dataset to limit_to samples, otherwise repeat samples as often as necessary to still have the same original dataset size
		# limit_sorted = Whether to preserve sample order when limiting the dataset
		# limit_seed = Integer seed to use for the deterministic selection of samples to limit to
		self.res_type = res_type
		self.preloaded_data = {}
		super().__init__(root_dir, dataset_type, reqd_inputs, reqd_targets, limit_to=limit_to, limit_hard=limit_hard, limit_sorted=limit_sorted, limit_seed=limit_seed)

	# noinspection PyMethodOverriding
	def load_samples(self, reqd_res_types):
		# reqd_res_types = Set of required resource types
		# Return a deterministically sorted list of the samples, where each entry in the list is Tuple[sample_key, Dict[res_type, path]], and a list of string details about the loaded samples
		check_file_func = self.get_check_file_func(self.res_type)
		samples = sorted((entry.name, {self.res_type: entry.path}) for entry in os.scandir(self.root) if entry.is_file() and entry.name.lower().endswith(self.res_type.ext) and (check_file_func is None or check_file_func(entry)))
		for res_type in reqd_res_types:
			if res_type != self.res_type:
				self.preloaded_data[res_type] = self.preload_resource(res_type)
		return samples, []

	#
	# Override interface
	#

	# noinspection PyMethodMayBeStatic, PyUnusedLocal
	def get_check_file_func(self, res_type) -> Optional[Callable]:
		# res_type = Resource type that the file should be checked for
		# Return None or a check file callable to be used like check_file(file=os.DirEntry)
		return None

	def preload_resource(self, res_type):
		# res_type = Resource type that should be preloaded
		# Return the preloaded resource value
		raise NotImplementedError(f"Class {self.__class__.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

############################
### Subset-based dataset ###
############################

# Subset-based dataset class
# noinspection PyAbstractClass
class SubsetDataset(ViewableStagedDataset):
	# The assumed format of the dataset folder is:
	#  - Samples:     ROOT/SAMPLES_DIR/RESOURCE/SUBSET/*.EXT
	#  - Blacklists:  ROOT/BLACKLISTS_DIR/BLACKLIST/SUBSET/*BLACKLIST_EXT

	SAMPLES_DIR = 'samples'
	BLACKLISTS_DIR = 'blacklists'
	BLACKLIST_EXT = '.list'

	#
	# Construction
	#

	def __init__(self, root_dir, reqd_inputs, reqd_targets, limit_to=None, limit_hard=False, limit_sorted=False, limit_seed=0, partition_limit_to=None, partition_ratios=DatasetTuple(0.7, 0.15, 0.15), partition_seed=0, res_suffix_inex_map=None, subsets_inex=INCLUDE_EXCLUDE_NONE, blacklist_type_inex=INCLUDE_EXCLUDE_NONE, blacklist_name_inex=INCLUDE_EXCLUDE_NONE, sample_regex_inex=INCLUDE_EXCLUDE_NONE):
		# root_dir = Dataset root directory path (ROOT)
		# reqd_inputs = Required model inputs (expects resolved dataset.DataSpec)
		# reqd_targets = Required model targets (expects resolved dataset.DataSpec)
		# limit_to = If a positive integer is provided, limit the dataset by allowing only limit_to samples (see the remaining limit_* arguments for options as to how this is to be implemented)
		# limit_hard = If True, truncate the dataset to limit_to samples, otherwise repeat samples as often as necessary to still have the same original dataset size
		# limit_sorted = Whether to preserve sample order when limiting the dataset
		# limit_seed = Integer seed to use for the deterministic selection of samples to limit to
		# partition_limit_to = DatasetTuple of non-negative sample count limits (see the remaining limit_* arguments for options as to how this is to be implemented)
		# partition_ratios = DatasetTuple of ratios in which to partition the dataset into TVS parts (does not need to sum to 1, but each component should be non-negative)
		# partition_seed = Integer seed to use for the deterministic random partitioning into TVS parts of the dataset
		# res_suffix_inex_map = Dict[res_type, IncludeExclude] of resource directory suffixes to load
		# subsets_inex = IncludeExclude of subset strings to load
		# blacklist_type_inex = IncludeExclude of blacklist types to consider during loading
		# blacklist_name_inex = IncludeExclude of blacklist file names to consider during loading (no extension)
		# sample_regex_inex = IncludeExclude of sample name regex patterns to restrict the loaded samples to
		super().__init__(root_dir, DatasetType.Train, reqd_inputs, reqd_targets, limit_to=limit_to, limit_hard=limit_hard, limit_sorted=limit_sorted, limit_seed=limit_seed, partition_limit_to=partition_limit_to, partition_ratios=partition_ratios, partition_seed=partition_seed, load_kwargs={'res_suffix_inex_map': res_suffix_inex_map, 'subsets_inex': subsets_inex, 'blacklist_type_inex': blacklist_type_inex, 'blacklist_name_inex': blacklist_name_inex, 'sample_regex_inex': sample_regex_inex})

	# noinspection PyMethodOverriding
	def load_samples(self, reqd_res_types, res_suffix_inex_map, subsets_inex, blacklist_type_inex, blacklist_name_inex, sample_regex_inex):
		# reqd_res_types = Set of required resource types
		# res_suffix_inex_map = Dict[res_type, IncludeExclude] of resource directory suffixes to load
		# subsets_inex = IncludeExclude of subset strings to load
		# blacklist_type_inex = IncludeExclude of blacklist types to consider during loading
		# blacklist_name_inex = IncludeExclude of blacklist file names to consider during loading (no extension)
		# sample_regex_inex = IncludeExclude of sample name regex patterns to restrict the loaded samples to
		# Return a deterministically sorted list of the samples, where each entry in the list is Tuple[sample_key, Dict[res_type, path]], and a list of string details about the loaded samples

		sorted_reqd_res_types = sorted(reqd_res_types)
		details = [
			"Required resource types: " + ', '.join(res_type.name for res_type in sorted_reqd_res_types if res_type.required),
			"Optional resource types: " + ', '.join(res_type.name for res_type in sorted_reqd_res_types if not res_type.required)
		]

		def res_type_from_dir(rdir_name):
			rdir_name_parts = rdir_name.split('#', maxsplit=1)
			if len(rdir_name_parts) <= 1:
				rtype_name = rdir_name
				rtype_suffix = '?'
			else:
				rtype_name, rtype_suffix = rdir_name_parts
			return self.get_res_type(rtype_name), rtype_suffix

		res_dir_names = set()
		res_type_res_dirs = {res_type: [] for res_type in reqd_res_types}
		res_type_subsets_map = {}
		samples_path = os.path.join(self.root, self.SAMPLES_DIR)
		for entry in os.scandir(samples_path):
			if entry.is_dir():
				res_type, res_type_suffix = res_type_from_dir(entry.name)
				if res_type:
					if res_type in reqd_res_types:
						res_suffix_inex = res_suffix_inex_map.get(res_type, INCLUDE_EXCLUDE_NONE) if res_suffix_inex_map else INCLUDE_EXCLUDE_NONE
						if (not res_suffix_inex.exclude or res_type_suffix not in res_suffix_inex.exclude) and (not res_suffix_inex.include or res_type_suffix in res_suffix_inex.include):
							res_dir_names.add(entry.name)
							subset_entries = {subentry for subentry in os.scandir(entry.path) if subentry.is_dir()}
							res_type_res_dirs[res_type].append((res_type_suffix, entry.name, {(subentry.name, subentry.path) for subentry in subset_entries}))
							res_dir_subsets = (subentry.name for subentry in subset_entries)
							res_type_subsets = res_type_subsets_map.get(res_type, None)
							if res_type_subsets is None:
								res_type_subsets_map[res_type] = set(res_dir_subsets)
							else:
								res_type_subsets.update(res_dir_subsets)
				else:
					raise ValueError(f"Unrecognised resource directory (could lead to erroneous blacklisting): {entry.name}")
		for res_dir_list in res_type_res_dirs.values():
			res_dir_list.sort()
		details.append("Relevant available resource dirs: " + ', '.join(sorted(res_dir_names)))

		subsets = None
		for res_type in reqd_res_types:
			if not res_type.required:
				continue
			res_type_subsets = res_type_subsets_map.get(res_type, None)
			if res_type_subsets is None:
				raise NotADirectoryError(f"Failed to find required resource type directory: {res_type.name}")
			if subsets is None:
				subsets = res_type_subsets
			else:
				subsets &= res_type_subsets
		if not subsets:
			raise ValueError("No common subsets were found")
		details.append("Common subsets: " + ', '.join(sorted(subsets)))

		if subsets_inex.include:
			if not subsets_inex.include <= subsets:
				raise NotADirectoryError(f"Failed to find these required subsets in ALL of the required resource type directories: {', '.join(subsets_inex.include - subsets)}")
			subsets &= subsets_inex.include
		if subsets_inex.exclude:
			subsets -= subsets_inex.exclude
		details.append("Chosen subsets: " + ', '.join(sorted(subsets)))

		res_dir_blacklists = {}
		other_blacklist_keys = set()
		blacklists_path = os.path.join(self.root, self.BLACKLISTS_DIR)
		blacklist_types_included = blacklist_type_inex.include and blacklist_type_inex.include.copy()
		blacklist_names_included = blacklist_name_inex.include and blacklist_name_inex.include.copy()

		def typed_blacklists_gen(blacklist_type_path):
			for subset_entry in os.scandir(blacklist_type_path):
				if subset_entry.is_dir() and subset_entry.name in subsets:
					for list_entry in os.scandir(subset_entry.path):
						if list_entry.is_file():
							list_name, list_ext = os.path.splitext(list_entry.name)
							if list_ext == self.BLACKLIST_EXT:
								blacklist_names_included.discard(list_name)
								if (not blacklist_name_inex.exclude or list_name not in blacklist_name_inex.exclude) and (not blacklist_name_inex.include or list_name in blacklist_name_inex.include):
									with open(list_entry.path, 'r') as file:
										line_count = 0
										for line in file.readlines():
											sample_name = line.strip()
											if sample_name:
												yield subset_entry.name, sample_name
												line_count += 1
										details.append(f"Retrieved {line_count} sample names from blacklist: {os.path.join(blacklist_type, subset_entry.name, list_entry.name)}")

		for type_entry in os.scandir(blacklists_path):
			if type_entry.is_dir():
				blacklist_type = type_entry.name
				blacklist_types_included.discard(blacklist_type)
				if not (blacklist_type_inex.exclude and blacklist_type in blacklist_type_inex.exclude):
					if blacklist_type in res_dir_names:
						res_dir_blacklists[blacklist_type] = set(typed_blacklists_gen(type_entry.path))
					else:
						res_type = res_type_from_dir(blacklist_type)[0]
						if (blacklist_type_inex.include and blacklist_type in blacklist_type_inex.include) or (not blacklist_type_inex.include and not res_type):
							other_blacklist_keys.update(typed_blacklists_gen(type_entry.path))

		if blacklist_types_included:
			raise ValueError(f"Failed to find these explicitly included blacklist types: {', '.join(sorted(blacklist_types_included))}")
		if blacklist_names_included:
			raise ValueError(f"Failed to find these explicitly included blacklist names: {', '.join(sorted(blacklist_names_included))}")

		sample_dicts = {}
		for res_type, res_dirs in res_type_res_dirs.items():
			if res_type.ext:
				res_type_ext = res_type.ext.lower()
				if res_type_ext[0] != '.':
					res_type_ext = '.' + res_type_ext
			else:
				res_type_ext = None
			sample_dict = {}
			sample_subsets = set()
			for res_dir_suffix, res_dir_name, res_dir_subsets in res_dirs:
				res_dir_sample_dict = {}
				for subset, subset_path in res_dir_subsets:
					if subset in subsets:
						sample_subsets.add(subset)
						for entry in os.scandir(subset_path):
							if entry.name and entry.is_file():
								name, ext = os.path.splitext(entry.name)
								if not res_type_ext or ext.lower() == res_type_ext:
									sample_key = (subset, name)
									if sample_key not in res_dir_sample_dict or res_dir_sample_dict[sample_key] > entry.name:
										res_dir_sample_dict[sample_key] = entry.path
				dict_size = len(res_dir_sample_dict)
				res_dir_blacklist = res_dir_blacklists.get(res_dir_name)
				if res_dir_blacklist:
					for blacklisted_key in res_dir_blacklist:
						res_dir_sample_dict.pop(blacklisted_key, None)
					details.append(f"Removed {dict_size - len(res_dir_sample_dict)} samples from {res_dir_name} resource directory based on {len(res_dir_blacklist)} blacklist entries")
				sample_dict.update(res_dir_sample_dict)
			if res_type.required and sample_subsets != subsets:
				raise ValueError(f"Could not find all chosen subsets for the resource type: {res_type.name}")
			sample_dicts[res_type] = sample_dict

		sample_dict_list = [sample_dict.keys() for res_type, sample_dict in sample_dicts.items() if res_type.required]
		common_samples = set(sample_dict_list[0]).intersection(*sample_dict_list[1:])
		common_sample_size = len(common_samples)

		subset_counts = Counter(sample_key[0] for sample_key in common_samples)
		for subset, count in sorted(subset_counts.items()):
			details.append(f"Found {count} common samples in {subset} subset")
		details.append(f"Found {common_sample_size} common samples in total")

		common_samples -= other_blacklist_keys
		details.append(f"Removed {common_sample_size - len(common_samples)} common samples based on {len(other_blacklist_keys)} unioned blacklist entries")

		regex_keys = set()
		if sample_regex_inex.include or sample_regex_inex.exclude:
			sample_regex_include = tuple(re.compile(sample_regex) if isinstance(sample_regex, str) else sample_regex for sample_regex in sample_regex_inex.include) if sample_regex_inex.include else ()
			sample_regex_exclude = tuple(re.compile(sample_regex) if isinstance(sample_regex, str) else sample_regex for sample_regex in sample_regex_inex.exclude) if sample_regex_inex.exclude else ()
			for sample_key in common_samples:
				any_include_matches = False
				for sample_regex in sample_regex_include:
					if sample_regex.fullmatch(sample_key[1]):
						any_include_matches = True
				if any(sample_regex.fullmatch(sample_key[1]) for sample_regex in sample_regex_exclude) or (sample_regex_include and not any_include_matches):
					regex_keys.add(sample_key)

		common_samples -= regex_keys
		details.append(f"Removed {len(regex_keys)} common samples due to sample name regexes")

		subset_counts = Counter(sample_key[0] for sample_key in common_samples)
		for subset, count in sorted(subset_counts.items()):
			details.append(f"Loaded {count} samples from {subset} subset")
		details.append(f"Loaded {len(common_samples)} samples in total")

		samples = [(sample_key, {res_type: sample_dict[sample_key] if res_type.required else sample_dict.get(sample_key, None) for res_type, sample_dict in sample_dicts.items()}) for sample_key in common_samples]
		samples.sort()
		if not samples:
			raise ValueError("No suitable samples were found")

		return samples, details

########################
### Helper functions ###
########################

# Image loader function
def image_loader(path, reqd_mode='RGB'):
	# path = Path of image to load
	# reqd_mode = Required image mode (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)
	with open(path, 'rb') as f:
		image = PIL.Image.open(f)
		image.load()  # We wish to guarantee that the image is loaded before continuing (otherwise 'return image' below would leave it up to later lazy loading which could lead to unexpected problems)
		if image.mode == reqd_mode:
			return image
		else:
			return image.convert(reqd_mode)

# Load raw RGB image
# noinspection PyUnusedLocal
def load_raw_image_rgb(dset, key, path, opts):
	return image_loader(path, reqd_mode='RGB')

# Load raw greyscale image
# noinspection PyUnusedLocal
def load_raw_image_l(dset, key, path, opts):
	return image_loader(path, reqd_mode='L')

# Parse a string sample specifier to tuple form
def parse_sample_spec(string, allow_key=False):
	# string = String sample specifier (e.g. T, V10, S0)
	# allow_key = Allow parsing of key specifications like 'T@samplename' and 'V@subset@name' (returned sample_index is then 'samplename' or ('subset', 'name'))

	if not string:
		raise ValueError("Sample specifier shouldn't be an empty string")

	dtype = string[0]
	if dtype == 'T':
		dataset_type = DatasetType.Train
	elif dtype == 'V':
		dataset_type = DatasetType.Valid
	elif dtype == 'S':
		dataset_type = DatasetType.Test
	else:
		raise ValueError(f"Failed to parse dataset type from sample specifier: {string}")

	if allow_key and len(string) >= 2 and string[1] == '@':
		part_list = string.split('@')
		if len(part_list) == 2:
			sample_index = part_list[1]
		else:
			sample_index = tuple(part_list[1:])
	else:
		number_string = string[1:]
		if not number_string:
			sample_index = None
		else:
			try:
				sample_index = int(number_string)
			except (ValueError, TypeError):
				raise ValueError(f"Failed to parse sample index from sample specifier: {string}")

	return dataset_type, sample_index

# Convert a sample specifier to a human-readable string (e.g. 'Train 40')
def sample_spec_string(sample):
	# sample = Sample specifier in tuple form (dataset_type, sample_index)
	return f"{sample[0].name} {sample[1] if sample[1] is not None else '<rand>'}"

# Resolve a sample specifier to tuple form
def resolve_sample_spec(sample, default_dataset_type=DatasetType.Train, allow_key=False):
	# sample = Sample specifier in any form (None = default, str = string sample specifier, (dataset.DatasetType, int) = dataset type and sample index)
	if sample is None:
		return default_dataset_type, None
	elif isinstance(sample, str):
		return parse_sample_spec(sample, allow_key=allow_key)
	elif isinstance(sample, tuple) and len(sample) == 2:
		return sample
	else:
		raise TypeError(f"Cannot resolve sample specifier: {sample}")

# Resolve a list of sample specifiers to tuple form
def resolve_sample_spec_list(sample_list, default_dataset_type=DatasetType.Train, allow_key=False):
	# sample_list = List of sample specifiers in any form (see resolve_sample_spec() function)
	if sample_list is None:
		return [resolve_sample_spec(None, default_dataset_type=default_dataset_type, allow_key=allow_key)]
	else:
		return [resolve_sample_spec(sample, default_dataset_type=default_dataset_type, allow_key=allow_key) for sample in sample_list]

# Data loader collate function that only tensor-collates the last two elements of each sample (input/target tensors) => For use with the StagedDataset class and extra data indices
def collate_staged_samples(batch):
	elem = batch[0]
	elem_size = len(elem)  # Note: Assumes that each element in the uncollated batch is a sequence, e.g. tuple, which will always be the case for StagedDataset
	if elem_size <= 2:
		return torch.utils.data.dataloader.default_collate(batch)
	elif not all(len(elem) == elem_size for elem in batch):
		raise RuntimeError("Each element in list of batch should be of equal size")
	collated_batch = [list(field) for field in zip(*batch)]
	collated_batch[0] = torch.utils.data.dataloader.default_collate(collated_batch[0])
	collated_batch[1] = torch.utils.data.dataloader.default_collate(collated_batch[1])
	return collated_batch
# EOF
